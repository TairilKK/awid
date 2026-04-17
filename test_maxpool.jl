include("bad.jl")

input = tensor(1, 4, 4)

net = chain((
  maxpool((2, 2)),
))

output = net(input)
model = graph(output)

x_data = zeros(Float64, 1, 4, 4)
x_data[1, :, :] = [
  1 2 5 8
  3 4 6 7
  12 10 19 34
  11 8 15 16
]

forward!(model, input => x_data)

println("output.data = ")
println(output.data)
