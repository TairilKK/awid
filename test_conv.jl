include("bad.jl")

input = tensor(1, 3, 3)
net = chain((
  conv((2, 2), 1 => 1, pad=0),
))
output = net(input)
model = graph(output)

conv_node = nothing
kernels = nothing
for node in model
  if node isa GraphNode{:conv,3}
    conv_node = node
    kernels, _, _ = node.args
    break
  end
end


x_data = zeros(Float64, 1, 3, 3)
x_data[1, :, :] = [
  1 2 3
  4 5 6
  7 8 9
]

kernels.data .= 0
kernels.data[1, 1, :, :] = [
  1 0
  0 1
]

forward!(model, input => x_data)

println("output size = ", size(output.data))
println("output = ")
println(output.data)

expected = zeros(Float64, 1, 2, 2)
expected[1, :, :] = [
  6 8
  12 14
]

println("expected = ")
println(expected)

println("max error = ", maximum(abs.(output.data .- expected)))


