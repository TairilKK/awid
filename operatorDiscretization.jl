include("base.jl")
include("losses.jl")
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
  W = GraphNode(0.01f0 * randn(m, n), true)
  # W = GraphNode(2 * rand(m, n) .- 1, true)
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
function (y::Conv)(x)
  in_channels = y.in_channels
  out_channels = y.out_channels
  kh, kw = y.kernel_size
  pad = y.pad

  padnode = GraphNode([pad])
  kernels = GraphNode(0.01f0 * randn(out_channels, in_channels, kh, kw), true)
  # kernels = GraphNode(2 * rand(out_channels, in_channels, kh, kw) .- 1, true)

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

  return GraphNode(:maxpool, (x,), zeros(C, out_h, out_w))
end
function (y::Flatten)(x)
  sz = length(x.data)
  return GraphNode(:flatten, (x,), zeros(sz))
end
function (y::DropOut)(x)
  probnode = GraphNode([y.prob])
  masknode = GraphNode(zeros(size(x.data)...))
  return GraphNode(:dropout, (x, probnode, masknode), zeros(size(x.data)...))
end
function (E::LogitCrossEntropy)(x, y)
  return GraphNode(:lce, (x, y), zeros(1))
end
function (E::SoftMax)(x)
  sz = length(x.data)
  return GraphNode(:softmax, (x,), zeros(sz))
end
# INFO: ^^^^^^^^^^^^^^^^^^^^^^^^
