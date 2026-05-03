include("base.jl")
include("losses.jl")

#=============================#
#=  operator discretization  =#
#=============================#
function (x::Tensor{N})() where N
  data = zeros(Float32, x.outsize...)
  return GraphNode(data)
end

function (E::BinaryCrossEntropy)(x, y)
  return GraphNode(:bce, (x, y), zeros(Float32, 1))
end

function (y::Sigmoid)(x)
  sz = length(x.data)
  return GraphNode(:sigmoid, (x,), zeros(Float32, sz))
end

function (y::Dense)(x)
  n = y.insize
  m = y.outsize
  W = GraphNode(0.01f0 * randn(Float32, m, n), true)
  b = GraphNode(randn(Float32, m), true)
  mul = GraphNode(:mul, (W, x), zeros(Float32, m))
  add = GraphNode(:add, (mul, b), zeros(Float32, m))
  return add
end

function (y::ReLU)(x)
  sz = length(x.data)
  return GraphNode(:relu, (x,), zeros(Float32, sz))
end

function (y::Conv)(x)
  in_channels = y.in_channels
  out_channels = y.out_channels
  kh, kw = y.kernel_size
  pad = y.pad

  padnode = GraphNode(Float32[pad])
  kernels = GraphNode(0.01f0 * randn(Float32, out_channels, in_channels, kh, kw), true)

  _, H, W = size(x.data)
  out_h = H + 2pad - kh + 1
  out_w = W + 2pad - kw + 1

  return GraphNode(
    :conv,
    (kernels, x, padnode),
    zeros(Float32, out_channels, out_h, out_w)
  )
end

function (y::MaxPool)(x)
  kh, kw = y.kernel_size

  C, H, W = size(x.data)

  out_h = H ÷ kh
  out_w = W ÷ kw

  return GraphNode(:maxpool, (x,), zeros(Float32, C, out_h, out_w))
end

function (y::Flatten)(x)
  sz = length(x.data)
  return GraphNode(:flatten, (x,), zeros(Float32, sz))
end

function (y::DropOut)(x)
  probnode = GraphNode(Float32[y.prob])
  masknode = GraphNode(zeros(Float32, size(x.data)...))
  return GraphNode(:dropout, (x, probnode, masknode), zeros(Float32, size(x.data)...))
end

function (E::LogitCrossEntropy)(x, y)
  return GraphNode(:lce, (x, y), zeros(Float32, 1))
end
