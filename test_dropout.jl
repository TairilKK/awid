include("bad.jl")

input = tensor(1, 4, 4)

net = chain((
  maxpool((2, 2)),
  flatten(),
  dropout(0.)
))

model = graph(net(input))

x_data = zeros(Float64, 1, 4, 4)
x_data[1, :, :] = [
  1 2 5 6
  3 4 7 8
  9 10 13 14
  11 12 15 16
]

forward!(model, input => x_data)

println("output.data = ")
println(output.data)
println("size(output.data) = ", size(output.data))
