# cafe: Convolutional Architecture for Fast Evaluation
abstract type Operator end
const Chain = Vector{Operator}
struct Sigmoid <: Operator end
struct ReLU <: Operator end
struct Dense <: Operator
  insize::Int64
  outsize::Int64
end

relu() = ReLU()
sigmoid() = Sigmoid()
dense(pair::Pair{Int64,Int64}) =
  Dense(first(pair), last(pair))
dense(pair::Pair{Int64,Int64}, activation) =
  tuple(dense(pair), activation())

abstract type Loss end
struct BinaryCrossEntropy <: Loss end
bce(output, target) = BinaryCrossEntropy()(output, target)

struct Tensor{N}
  outsize::NTuple{N,Int64}
end
tensor(sz...) = Tensor(sz)()

function chain(operators)
  function flatten(x::Tuple)
    y = Vector{Operator}()
    for v in x
      if v isa Tuple
        push!(y, v...)
      else
        push!(y, v)
      end
    end
    return y
  end

  result = Vector{Operator}()
  for operator in flatten(operators)
    push!(result, operator)
  end
  return result
end

function (chain::Chain)(x)
  node = x
  for op in chain
    node = op(node)
  end
  return node
end

# %%

# bad: Backward Automatic Differentiation
mutable struct GraphNode{OP,N}
  args::NTuple{N,GraphNode}
  grad
  data
end

const GraphWeight = GraphNode{:weight,0}
const GraphTensor = GraphNode{:tensor,0}
function GraphNode(data::T, trainable=false) where T
  if trainable
    return GraphNode{:weight,0}((), zero(data), data)
  else
    return GraphNode{:tensor,0}((), zero(data), data)
  end
end

function GraphNode(op::Symbol, args::Tuple, data::T) where T
  N = length(args)
  grad = similar(data)
  return GraphNode{op,N}(args, grad, data)
end

function graph(node)
  function visit!(node::GraphNode, visited, ordered)
    if node in visited
    else
      push!(visited, node)
      for arg in node.args
        visit!(arg, visited, ordered)
      end
      push!(ordered, node)
    end
    return nothing
  end
  ordered = Vector{GraphNode}()
  visited = Set{GraphNode}()
  visit!(node, visited, ordered)
  return ordered
end

function zerograd!(order::Vector{GraphNode})
  for node in order
    node.grad .= 0
  end
end

function primal!(tensor::GraphTensor) end
function primal!(weight::GraphWeight) end
function tangent!(tensor::GraphTensor) end
function tangent!(weight::GraphWeight) end
function forward!(order::Vector{GraphNode}, pairs...)
  for pair in pairs
    tensor, data = pair
    tensor.data .= data
  end

  for node in order
    primal!(node)
  end
end
function forwardd!(order::Vector{GraphNode}, pairs...)
  for pair in pairs
    tensor, grad = pair
    tensor.grad .= grad
  end

  for node in order
    tangent!(node)
  end
end

function adjoint!(::GraphTensor) end
function adjoint!(::GraphWeight) end
function backward!(order::Vector{GraphNode})
  seed = last(order)
  seed.grad .= 1

  for node in reverse(order)
    adjoint!(node)
  end
end

# %%

import Base: show
show(io::IO, x::GraphNode{OP,N}) where {OP,N} =
  print(io, "layer ", OP, " with ", N, " arg(s)")
show(io::IO, x::GraphWeight) = print(io, "weight")
show(io::IO, x::GraphTensor) = print(io, "tensor")

# %%

function optimize!(graph, η)
  for node in graph
    if node isa GraphWeight
      node.data .-= η * node.grad
    end
  end
end

# %%

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

