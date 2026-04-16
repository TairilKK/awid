include("cafe.jl")
include("bad.jl")

using LinearAlgebra
#=============================#
#=  operator discretization  =#
#=============================#
function (x::Tensor{N})() where N
  data = zeros(x.outsize...)
  return GraphNode(data)
end

function (E::BinaryCrossEntropy)(x, y)
  return GraphNode(:bce, (x, y), zeros(1))
end

function (y::Sigmoid)(x)
  sz = length(x.data)
  return GraphNode(:sigmoid, (x,), zeros(sz))
end

function (y::Dense)(x)
  n = y.insize
  m = y.outsize
  W = GraphNode(randn(m, n), true)
  b = GraphNode(randn(m), true)
  mul = GraphNode(:mul, (W, x), zeros(m))
  add = GraphNode(:add, (mul, b), zeros(m))
  return add
end

function (y::ReLU)(x)
  sz = length(x.data)
  return GraphNode(:relu, (x,), zeros(sz))
end

#=====================#
#=  operator passes  =#
#=====================#
function primal!(z::GraphNode{:bce,2})
  x, y = z.args
  z.data = -(y.data .* log.(x.data) + (1 .- y.data) .* log.(1 .- x.data))
  return nothing
end

function adjoint!(z::GraphNode{:bce,2})
  x, y = z.args
  x.grad -= y.data ./ x.data .* z.grad
  x.grad += (1 .- y.data) ./ (1 .- x.data) .* z.grad
  return nothing
end

function primal!(y::GraphNode{:mul,2})
  W, x = y.args
  y.data = W.data * x.data
  return nothing
end

function adjoint!(y::GraphNode{:mul,2})
  W, x = y.args
  W.grad += y.grad * x.data'
  x.grad += W.data' * y.grad
  return nothing
end

function primal!(y::GraphNode{:relu,1})
  x, = y.args
  y.data .= max.(0, x.data)
  return nothing
end

function adjoint!(y::GraphNode{:relu,1})
  x, = y.args
  for i in 1:length(x.data)
    if x.data[i] == y.data[i]
      x.grad[i] += y.grad[i]
    end
  end
  return nothing
end

function primal!(z::GraphNode{:add,2})
  x, y = z.args
  z.data = x.data .+ y.data
  return nothing
end

function adjoint!(z::GraphNode{:add,2})
  x, y = z.args
  x.grad += z.grad
  y.grad += z.grad
  return nothing
end

function primal!(z::GraphNode{:dot,2})
  x, y = z.args
  z.data = dot(x.data, y.data)
  return nothing
end

function adjoint!(z::GraphNode{:dot,2})
  x, y = z.args
  x.grad += y.data .* z.grad
  y.grad += x.data .* z.grad
  return nothing
end

function primal!(y::GraphNode{:sum,1})
  x, = y.args
  y.data = sum(x.data)
  return nothing
end

function adjoint!(y::GraphNode{:sum,1})
  x, = y.args
  x.grad += y.grad
  return nothing
end

function primal!(y::GraphNode{:sigmoid,1})
  x, = y.args
  y.data = 1 ./ (1 .+ exp.(-x.data))
  return nothing
end

function adjoint!(y::GraphNode{:sigmoid,1})
  x, = y.args
  x.grad += exp.(-x.data) ./ (1 .+ exp.(-x.data)) .^ 2 .* y.grad
  return nothing
end

