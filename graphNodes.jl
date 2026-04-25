# bad: Backward Automatic Differentiation
mutable struct GraphNode{OP,N}
  args::NTuple{N,GraphNode}
  grad::Array{Float32}
  data::Array{Float32}
  cache::Dict{Symbol,Any}
end

const GraphWeight = GraphNode{:weight,0}
const GraphTensor = GraphNode{:tensor,0}
function GraphNode(data::T, trainable=false) where T
  cache = Dict{Symbol,Any}()
  if trainable
    return GraphNode{:weight,0}((), zero(data), data, cache)
  else
    return GraphNode{:tensor,0}((), zero(data), data, cache)
  end
end

function GraphNode(op::Symbol, args::Tuple, data::T) where T
  N = length(args)
  grad = similar(data)
  cache = Dict{Symbol,Any}()
  return GraphNode{op,N}(args, grad, data, cache)
end

function graph(node::GraphNode)
  function visit!(node::GraphNode, visited::Set{GraphNode}, ordered::Vector{GraphNode})
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
  return nothing
end

function zero_nonweight_grads!(order::Vector{GraphNode})
  for node in order
    if !(node isa GraphWeight)
      node.grad .= 0
    end
  end
  return nothing
end

function primal!(tensor::GraphTensor) end
function primal!(weight::GraphWeight) end

# NOTE: === [ unused ] ===
function tangent!(tensor::GraphTensor) end
function tangent!(weight::GraphWeight) end
# NOTE: === ] unused [ ===

function forward!(order::Vector{GraphNode}, pairs...)
  for pair in pairs
    tensor, data = pair
    tensor.data .= data
  end

  for node in order
    primal!(node)
  end
  return nothing
end

# NOTE: === [ unused ] ===
function forwardd!(order::Vector{GraphNode}, pairs...)
  for pair in pairs
    tensor, grad = pair
    tensor.grad .= grad
  end

  for node in order
    tangent!(node)
  end
  return nothing
end
# NOTE: === ] unused [ ===

function adjoint!(::GraphTensor) end
function adjoint!(::GraphWeight) end
function backward!(order::Vector{GraphNode})
  seed = last(order)
  seed.grad .= 1f0

  for node in reverse(order)
    adjoint!(node)
  end
  return nothing
end


import Base: show
show(io::IO, x::GraphNode{OP,N}) where {OP,N} =
  print(io, "layer ", OP, " with ", N, " arg(s)")
show(io::IO, x::GraphWeight) = print(io, "weight")
show(io::IO, x::GraphTensor) = print(io, "tensor")


function optimize!(graph, η)
  for node in graph
    if node isa GraphWeight
      node.data .-= η * node.grad
    end
  end
  return nothing
end
