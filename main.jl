using MLDatasets
train_data = MLDatasets.FashionMNIST(split=:train)
test_data = MLDatasets.FashionMNIST(split=:test)

function prepare_data(data)
    N = length(data)
    x_data = Vector{Array{Float32,3}}(undef, N)
    d_data = Vector{Vector{Float32}}(undef, N)
    for i in 1:N
        x_data[i] = reshape(Float32.(data.features[:, :, i]) ./ 255., 1, 28, 28)
        d = zeros(Float32, 10)
        d[data.targets[i]+1] = 1.0
        d_data[i] = d
    end
    return x_data, d_data
end

x_train, d_train = prepare_data(train_data)
x_test, d_test = prepare_data(test_data)

# %% 
include("base.jl")
include("losses.jl")
include("graphNodes.jl")
include("operators.jl")
include("operatorPasses.jl")
include("operatorDiscretization.jl")

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

settings = (
    epoch=3,
    batch=collect(1:60000),
    learning_rate=1e-2
)

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
        rr = zeros(Int64, 10)
        rr[pred_class] = 1
    end
    acc = correct / total
    println("Accuracy = ", round(acc * 100, digits=2),
        "% (", correct, "/", total, ")")
    return acc * 100
end

function train!(model, batch, x_train, d_train)
    shuffle!(batch)
    L = 0.0
    for sample in batch
        zerograd!(model)
        forward!(model,
            input => x_train[sample],
            target => d_train[sample])
        for (i, node) in enumerate(model)
            if any(isnan, node.data) || any(isinf, node.data)
                println("PREV NODE DATA")
                println("i = ", i - 1)
                println("node = ", model[i-1])
                println("size = ", size(model[i-1].data))
                println("data = ", model[i-1].data)

                println("BAD NODE DATA")
                println("i = ", i)
                println("node = ", node)
                println("size = ", size(node.data))
                println("data = ", node.data)
                error("node data exploded")
            end
        end
        backward!(model)
        for (i, node) in enumerate(model)
            if any(isnan, node.grad) || any(isinf, node.grad)
                println("BAD GRAD in node ", i, " -> ", node)
                println("data = ", node.data)
                println("grad = ", node.grad)
                error("gradient exploded")
            end
        end
        L += model[end].data[1]
        optimize!(model, settings.learning_rate)
    end
    return L
end

for _ in 1:settings.epoch
    @time L = train!(model, settings.batch, x_train, d_train)
    print("Loss: ")
    println(L)
end

test(model, x_test, d_test)
