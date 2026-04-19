include("cafe.jl")
include("bad.jl")

net = chain((
  conv((3, 3), 1 => 6, pad=1, bias=false), # 3*3*6 = 54
  maxpool((2, 2)),                         # 54 / 2 = 27
  flatten(),
  # dense(4 => 8, relu),
  dense(24 => 2, sigmoid),
))
input = tensor(1, 4, 4)
target = tensor(2)
output = net(input)
loss = bce(output, target)
model = graph(loss)

# %%
using Random

function data(;
  size=(1, 2, 2),
  n_train_0=10,
  n_train_1=10,
  n_test_0=5,
  n_test_1=5,
  seed=123
)
  Random.seed!(seed)

  function sample(class)
    if class == 1
      return rand(Float64, size...) .* 0.5 .+ 0.5
    else
      return rand(Float64, size...) .* 0.5
    end
  end

  x_train = vcat(
    [sample(0) for _ in 1:n_train_0],
    [sample(1) for _ in 1:n_train_1]
  )

  d_train = vcat(
    [[1, 0] for _ in 1:n_train_0],
    [[0, 1] for _ in 1:n_train_1]
  )

  x_test = vcat(
    [sample(0) for _ in 1:n_test_0],
    [sample(1) for _ in 1:n_test_1]
  )

  d_test = vcat(
    [[1, 0] for _ in 1:n_test_0],
    [[0, 1] for _ in 1:n_test_1]
  )

  return (x_test, d_test, x_train, d_train)
end
# %%


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
    # debug print
    print(d)
    print("; ")
    print(round.(result, digits=3))
    print("; ")
    rr = zeros(Int64, 2)
    rr[pred_class] = 1
    print(rr)
    println(";")
  end
  acc = correct / total
  println("Accuracy = ", round(acc * 100, digits=2), "% (", correct, "/", total, ")")
  return acc
end

function train!(model, batch, x_train, d_train)
  shuffle!(batch)
  L = 0.0
  for sample in batch
    zerograd!(model)
    forward!(model,
      input => x_train[sample],
      target => d_train[sample])
    backward!(model)
    L += model[end].data[1]
    optimize!(model, 1e-1)
  end
  return L
end

# %%
x_test, d_test, x_train, d_train = data(
  size=(1, 4, 4),
  n_train_0=10,
  n_train_1=10
)

test(model, x_test, d_test)

for _ in 1:5
  train!(model, collect(1:20), x_train, d_train)
end

test(model, x_test, d_test)

