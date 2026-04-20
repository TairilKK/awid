using BenchmarkTools
using MLDatasets
using Random
using Printf

# Data ------------------------------------------------------------------------
train_data = MLDatasets.FashionMNIST(split=:train)
test_data = MLDatasets.FashionMNIST(split=:test)

function prepare_data(data)
    N = length(data)
    x_data = Vector{Array{Float32,3}}(undef, N)
    d_data = Vector{Vector{Float32}}(undef, N)
    for i in 1:N
        x_data[i] = reshape(Float32.(data.features[:, :, i]), 1, 28, 28)
        d = zeros(Float32, 10)
        d[data.targets[i]+1] = 1.0f0
        d_data[i] = d
    end
    return x_data, d_data
end

x_train, d_train = prepare_data(train_data)
x_test, d_test = prepare_data(test_data)

# Model -----------------------------------------------------------------------
include("base.jl")
include("losses.jl")
include("graphNodes.jl")
include("operators.jl")
include("operatorPasses.jl")
include("operatorDiscretization.jl")

function build_model()
    net = chain((
        conv((3, 3), 1 => 6, pad=1, bias=false),
        maxpool((2, 2)),
        conv((3, 3), 6 => 16, pad=1, bias=false),
        maxpool((2, 2)),
        flatten(),
        dense(784 => 84, relu),
        dropout(0.),
        dense(84 => 10),
    ))

    input = tensor(1, 28, 28)
    target = tensor(10)
    output = net(input)
    loss = lce(output, target)
    model = graph(loss)

    return (; net, input, target, output, loss, model)
end

function find_nodes(model, ::Val{op}) where {op}
    return [node for node in model if node isa GraphNode{op}]
end

function op_symbol(node::GraphNode{OP,N}) where {OP,N}
    return OP
end

function summary_line(name, trial)
    t = minimum(trial).time / 1e6
    gctime = minimum(trial).gctime / 1e6
    mem = minimum(trial).memory / 1024
    allocs = minimum(trial).allocs
    @printf("%-28s %10.3f ms   gc %8.3f ms   mem %10.1f KiB   allocs %d\n", name, t, gctime, mem, allocs)
end

function print_cache_state(model)
    convs = find_nodes(model, Val(:conv))
    println("\nCache state for conv nodes:")
    for (i, node) in enumerate(convs)
        println("  conv[$i] keys = ", collect(keys(node.cache)))
        for (k, v) in node.cache
            println("    ", k, " => size ", size(v), ", eltype ", eltype(v))
        end
    end
end

function benchmark_whole_model(model, input, target, x, d)
    println("\n=== Whole-model benchmarks ===")

    forward!(model, input => x, target => d)
    zerograd!(model)
    backward!(model)

    b_forward = @benchmark forward!($model, $input => $x, $target => $d)
    summary_line("forward!", b_forward)

    forward!(model, input => x, target => d)
    zerograd!(model)
    b_backward = @benchmark begin
        zerograd!($model)
        forward!($model, $input => $x, $target => $d)
        backward!($model)
    end
    summary_line("forward! + backward!", b_backward)

    b_train_step = @benchmark begin
        zerograd!($model)
        forward!($model, $input => $x, $target => $d)
        backward!($model)
        optimize!($model, 1f-2)
    end
    summary_line("train step", b_train_step)

    println("\nAllocated bytes (single call):")
    println("  forward!             : ", @allocated forward!(model, input => x, target => d))
    println("  forward!+backward!   : ", @allocated begin
        zerograd!(model)
        forward!(model, input => x, target => d)
        backward!(model)
    end)
end

function benchmark_selected_nodes(model, input, target, x, d)
    println("\n=== Selected node benchmarks ===")

    # Prepare valid forward activations / caches.
    forward!(model, input => x, target => d)
    zerograd!(model)
    backward!(model)

    convs = find_nodes(model, Val(:conv))
    maxpools = find_nodes(model, Val(:maxpool))
    muls = find_nodes(model, Val(:mul))
    lces = find_nodes(model, Val(:lce))

    for (i, node) in enumerate(convs)
        b = @benchmark primal!($node)
        summary_line("conv[$i] primal!", b)
    end

    for (i, node) in enumerate(convs)
        kernels, xin, _ = node.args
        b = @benchmark begin
            fill!($kernels.grad, zero(eltype($kernels.grad)))
            fill!($xin.grad, zero(eltype($xin.grad)))
            adjoint!($node)
        end
        summary_line("conv[$i] adjoint!", b)
    end

    for (i, node) in enumerate(maxpools)
        b = @benchmark primal!($node)
        summary_line("maxpool[$i] primal!", b)
    end

    for (i, node) in enumerate(maxpools)
        xnode = node.args[1]
        b = @benchmark begin
            fill!($xnode.grad, zero(eltype($xnode.grad)))
            adjoint!($node)
        end
        summary_line("maxpool[$i] adjoint!", b)
    end

    for (i, node) in enumerate(muls)
        b = @benchmark primal!($node)
        summary_line("mul[$i] primal!", b)
    end

    for (i, node) in enumerate(muls)
        W, xnode = node.args
        b = @benchmark begin
            fill!($W.grad, zero(eltype($W.grad)))
            fill!($xnode.grad, zero(eltype($xnode.grad)))
            adjoint!($node)
        end
        summary_line("mul[$i] adjoint!", b)
    end

    for (i, node) in enumerate(lces)
        b = @benchmark primal!($node)
        summary_line("lce[$i] primal!", b)
    end

    for (i, node) in enumerate(lces)
        xnode, _ = node.args
        b = @benchmark begin
            fill!($xnode.grad, zero(eltype($xnode.grad)))
            adjoint!($node)
        end
        summary_line("lce[$i] adjoint!", b)
    end
end

function main()
    Random.seed!(1234)

    built = build_model()
    model = built.model
    input = built.input
    target = built.target

    x = x_train[1]
    d = d_train[1]

    println("Benchmarking on one FashionMNIST sample with the current model from main.jl")
    println("Model nodes: ", length(model))
    println("Conv nodes : ", length(find_nodes(model, Val(:conv))))
    println("Mul nodes  : ", length(find_nodes(model, Val(:mul))))
    println("Pool nodes : ", length(find_nodes(model, Val(:maxpool))))

    # Warmup compilation
    zerograd!(model)
    forward!(model, input => x, target => d)
    backward!(model)
    optimize!(model, 1f-2)

    print_cache_state(model)
    benchmark_whole_model(model, input, target, x, d)
    benchmark_selected_nodes(model, input, target, x, d)
end

main()
