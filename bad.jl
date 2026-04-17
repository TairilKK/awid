include("cafe.jl")
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
end

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

# INFO: My work: 
# =========================
# ========= START =========
# =========================
function (y::Conv)(x)
  in_channels = y.in_channels
  out_channels = y.out_channels
  kh, kw = y.kernel_size
  pad = y.pad

  padnode = GraphNode([pad])
  kernels = GraphNode(randn(out_channels, in_channels, kh, kw), true)

  _, H, W = size(x.data)
  out_h = H + 2pad - kh + 1
  out_w = W + 2pad - kw + 1

  return GraphNode(
    :conv,
    (kernels, x, padnode),
    zeros(out_channels, out_h, out_w)
  )
end

function (y::MaxPool)(x)
  kh, kw = y.kernel_size

  C, H, W = size(x.data)

  @assert H % kh == 0
  @assert W % kw == 0

  out_h = H ÷ kh
  out_w = W ÷ kw

  println((C, out_h, out_w))
  return GraphNode(:maxpool, (x,), zeros(C, out_h, out_w))
end

function (y::Flatten)(x)
  sz = length(x.data)
  return GraphNode(:flatten, (x,), zeros(sz))
end

function (y::Dropout)(x)
  probnode = GraphNode([y.prob])
  masknode = GraphNode(zeros(size(x.data)...))
  return GraphNode(:dropout, (x, probnode, masknode), zeros(size(x.data)...))
end
# INFO: My work: 
# ========================= 
# =========  END  ========= 
# =========================

#=====================#
#=  operator passes  =#
#=====================#

## BinaryCrossEntropy
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

# Mul
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

# ReLU
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

# Add
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

# Dot
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

# Sum
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

# Sigmoid
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

# INFO: My work: 
# =========================
# ========= START =========
# =========================
function primal!(y::GraphNode{:conv,3})
  kernels, x, padnode = y.args
  pad = padnode.data[1]

  W = kernels.data
  X = x.data

  OC, IC, KH, KW = size(W)
  ICx, H, WW = size(X)

  @assert IC == ICx

  out_h = H + 2 * pad - KH + 1
  out_w = WW + 2 * pad - KW + 1

  Xpad = zeros(eltype(X), IC, H + 2 * pad, WW + 2 * pad)
  Xpad[:, pad+1:pad+H, pad+1:pad+WW] .= X

  fill!(y.data, 0)

  for oc in 1:OC
    for oh in 1:out_h
      for ow in 1:out_w
        s = 0.0
        for ic in 1:IC
          for i in 1:KH
            for j in 1:KW
              s += W[oc, ic, i, j] * Xpad[ic, oh+i-1, ow+j-1]
            end
          end
        end
        y.data[oc, oh, ow] = s
      end
    end
  end

  return nothing
end

function adjoint!(y::GraphNode{:conv,3})
  kernels, x, padnode = y.args
  pad = padnode.data[1]

  W = kernels.data
  X = x.data
  GY = y.grad

  OC, IC, KH, KW = size(W)
  _, H, WW = size(X)
  _, out_h, out_w = size(y.data)

  Xpad = zeros(eltype(X), IC, H + 2 * pad, WW + 2 * pad)
  Xpad[:, pad+1:pad+H, pad+1:pad+WW] .= X

  GXpad = zeros(eltype(X), IC, H + 2 * pad, WW + 2 * pad)

  for oc in 1:OC
    for oh in 1:out_h
      for ow in 1:out_w
        gy = GY[oc, oh, ow]
        for ic in 1:IC
          for i in 1:KH
            for j in 1:KW
              kernels.grad[oc, ic, i, j] += Xpad[ic, oh+i-1, ow+j-1] * gy
              GXpad[ic, oh+i-1, ow+j-1] += W[oc, ic, i, j] * gy
            end
          end
        end
      end
    end
  end

  x.grad .+= GXpad[:, pad+1:pad+H, pad+1:pad+WW]

  return nothing
end

function primal!(y::GraphNode{:maxpool,1})
  x, = y.args
  X = x.data

  C, H, W = size(X)
  _, out_h, out_w = size(y.data)

  kh = H ÷ out_h
  kw = W ÷ out_w

  fill!(y.data, 0)

  for c in 1:C
    for oh in 1:out_h
      for ow in 1:out_w
        h_start = (oh - 1) * kh + 1
        h_end = oh * kh
        w_start = (ow - 1) * kw + 1
        w_end = ow * kw

        m = -Inf
        for i in h_start:h_end
          for j in w_start:w_end
            if X[c, i, j] > m
              m = X[c, i, j]
            end
          end
        end

        y.data[c, oh, ow] = m
      end
    end
  end

  return nothing
end

function adjoint!(y::GraphNode{:maxpool,1})
  x, = y.args
  X = x.data
  GY = y.grad

  C, H, W = size(X)
  _, out_h, out_w = size(y.data)

  kh = H ÷ out_h
  kw = W ÷ out_w

  for c in 1:C
    for oh in 1:out_h
      for ow in 1:out_w
        h_start = (oh - 1) * kh + 1
        h_end = oh * kh
        w_start = (ow - 1) * kw + 1
        w_end = ow * kw

        max_i = h_start
        max_j = w_start
        max_val = X[c, h_start, w_start]

        for i in h_start:h_end
          for j in w_start:w_end
            if X[c, i, j] > max_val
              max_val = X[c, i, j]
              max_i = i
              max_j = j
            end
          end
        end

        x.grad[c, max_i, max_j] += GY[c, oh, ow]
      end
    end
  end

  return nothing
end

function primal!(y::GraphNode{:flatten,1})
  x, = y.args
  y.data .= reshape(x.data, :)
  return nothing
end

function adjoint!(y::GraphNode{:flatten,1})
  x, = y.args
  x.grad .+= reshape(y.grad, size(x.data))
  return nothing
end

function primal!(y::GraphNode{:dropout,3})
  x, probnode, masknode = y.args
  p = probnode.data[1]

  @assert 0.0 <= p < 1.0

  scale = 1.0 / (1.0 - p)

  for i in eachindex(x.data)
    if rand() < p
      masknode.data[i] = 0.0
    else
      masknode.data[i] = scale
    end
  end

  y.data .= x.data .* masknode.data
  return nothing
end

function adjoint!(y::GraphNode{:dropout,3})
  x, probnode, masknode = y.args
  x.grad .+= y.grad .* masknode.data
  return nothing
end
# INFO: My work: 
# =========================
# =========  END  =========
# =========================
