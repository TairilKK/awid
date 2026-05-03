println("Program started.\n")

using MLDatasets

function data()
    println("Loading FashionMNIST...")
    function prepare_data(data)
        N = length(data)
        x_data = Vector{Array{Float32,3}}(undef, N)
        d_data = Vector{Vector{Float32}}(undef, N)

        for i in 1:N
            x_data[i] = reshape(Float32.(data.features[:, :, i]), 1, 28, 28)
            d = zeros(Float32, 10)
            d[data.targets[i]+1] = 1.0
            d_data[i] = d
        end

        return (x_data, d_data)
    end

    train_data = MLDatasets.FashionMNIST(split=:train)
    test_data = MLDatasets.FashionMNIST(split=:test)

    x_train, d_train = prepare_data(train_data)
    x_test, d_test = prepare_data(test_data)

    println("Data ready.")
    return (x_train, d_train, x_test, d_test)
end

println("Including project files...")
include("base.jl")
include("losses.jl")
include("graphNodes.jl")
include("operators.jl")
include("operatorPasses.jl")
include("operatorDiscretization.jl")
println("All files included.\n")

x_train, d_train, x_test, d_test = data()

println("\nBuilding network...")
net = chain((
    conv((3, 3), 1 => 6, pad=1, bias=false),
    maxpool((2, 2)),
    conv((3, 3), 6 => 16, pad=1, bias=false),
    maxpool((2, 2)),
    flatten(),
    dense(784 => 84, relu),
    dropout(0.4f0),
    dense(84 => 10),
))
println("Network built.\n")

println("Creating input/target tensors and computation graph...")
input = tensor(1, 28, 28)
target = tensor(10)
output = net(input)
loss = lce(output, target)
model = graph(loss)
println("Graph created. Number of nodes: ", length(model))

settings = (
    epoch=3,
    indices=collect(1:60_000),
    batch_size=10,
    learning_rate=1e-2
)
println("\nSettings:")
println("  epochs = ", settings.epoch)
println("  batch_size = ", settings.batch_size)
println("  learning_rate = ", settings.learning_rate)
println("  number of training samples = ", length(settings.indices))

using Random

function test(model, x_test, d_test)
    correct = 0
    total = length(x_test)

    for (x, d) in zip(x_test, d_test)
        forward!(model, input => x, target => d)
        result = output.data

        pred_class = argmax(result)
        true_class = argmax(d)
        if pred_class == true_class
            correct += 1
        end
    end

    acc = round(correct / total * 100, digits=2)
    return acc
end

function train_sgd!(model, incdices, x_train, d_train)
    shuffle!(incdices)
    L = 0.0f0

    for sample in incdices
        zerograd!(model)
        forward!(model,
            input => x_train[sample],
            target => d_train[sample])
        backward!(model)
        L += model[end].data[1]
        optimize!(model, settings.learning_rate)
    end
    return L / length(incdices)
end

using ProgressMeter

function train_minibatch!(model, indices, batch_size, x_train, d_train, learning_rate)
    shuffle!(indices)
    n_samples = length(indices)
    n_batches = cld(n_samples, batch_size)
    prog = Progress(n_batches; desc="Epoch progress:", barlen=40)

    total_loss = 0.0f0
    n_samples = length(indices)
    for batch_start in 1:batch_size:n_samples
        batch_end = min(batch_start + batch_size - 1, n_samples)
        current_batch_size = batch_end - batch_start + 1
        zerograd!(model)
        for pos in batch_start:batch_end
            zero_nonweight_grads!(model)
            sample_idx = indices[pos]
            forward!(model,
                input => x_train[sample_idx],
                target => d_train[sample_idx])
            backward!(model)
            total_loss += model[end].data[1]
        end
        optimize!(model, learning_rate)

        next!(prog; showvalues=[
            (:batch, batch_end ÷ batch_size),
            (:loss, round(total_loss / (batch_start + batch_size), digits=4))
        ])
    end
    return total_loss / n_samples
end

println("\nRunning initial test...")
test(model, x_test, d_test)

total_time = 0.0
total_allocs = 0

println("\nRunning training...")
for epoch in 1:settings.epoch
    global IS_TRAINING = true
    stats = @timed train_minibatch!(model, settings.indices, settings.batch_size, x_train, d_train, settings.learning_rate)
    L = stats.value
    global total_time += stats.time
    global total_allocs += stats.bytes

    global IS_TRAINING = false
    train_acc = test(model, x_train, d_train)
    test_acc = test(model, x_test, d_test)

    println("Epoch ", epoch, ": train_acc = ", train_acc, "%, test_acc = ", test_acc, "%")
end

println("\n--- Training Stats for train_minibatch! ---")
println("Total Time: ", round(total_time, digits=3), " seconds")
println("Total Allocations: ", round(total_allocs / 1024^3, digits=2), " GB")

println("\nProgram finished.")
