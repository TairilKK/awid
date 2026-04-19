include("cafe.jl")
include("bad.jl")

using Random

net =
  chain((
    dense(2 => 16, relu),
    dense(16 => 1, sigmoid),
  ))

input = tensor(2)
target = tensor(1)
output = net(input)
loss = bce(output, target)
model = graph(loss)
# %%

function data(N)
  c = ([-1, -1], [-1, +1], [+1, -1], [+1, +1])
  y = (0, 1, 1, 0)
  xs = zeros(2, N)
  ys = zeros(N)

  for i in 1:N
    j = rand(1:4)
    xs[:, i] .= 0.1randn(2) + c[j]
    ys[i] = y[j]
  end

  return xs, ys
end

function test(model)
  for (x, y) in zip(([+1, +1], [+1, -1], [-1, +1], [-1, -1]), ([0], [1], [1], [0]))
    forward!(model, input => x, target => y)
    result = round(output.data[1], digits=3)
    print(x[1] > 0 ? "1" : "0", " xor ", x[2] > 0 ? "1" : "0", " = ")
    println(result)
  end
end

inputs, targets = data(1000)

function train!(model, batch, inputs, targets)
  shuffle!(batch)
  L = 0.0
  for sample in batch
    zerograd!(model)
    forward!(model,
      input => inputs[:, sample],
      target => targets[sample, :])
    backward!(model)
    L += model[end].data[1]
    optimize!(model, 1e-1)
  end
  return L
end

batch = collect(1:500)
println("[x] Random model:")
test(model)
println("[x] Training...")
for _ in 1:5
  L = train!(model, batch, inputs, targets)
  println("[+] Loss: ", L)
end
println("[x] Final model:")
test(model)


