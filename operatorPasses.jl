include("graphNodes.jl")
using LinearAlgebra
#=====================#
#=  operator passes  =#
#=====================#

function primal!(z::GraphNode{:bce,2})
  x, y = z.args
  z.data .= .- (y.data .* log.(x.data) .+ (1 .- y.data) .* log.(1 .- x.data))
  return nothing
end
function adjoint!(z::GraphNode{:bce,2})
  x, y = z.args
  x.grad .+= ((1 .- y.data) ./ (1 .- x.data) .- (y.data ./ x.data)) .* z.grad
  return nothing
end

function primal!(y::GraphNode{:mul,2})
  W, x = y.args
  mul!(y.data, W.data, x.data)
  return nothing
end
function adjoint!(y::GraphNode{:mul,2})
  W, x = y.args
  mul!(W.grad, y.grad, x.data', 1, 1) 
  mul!(x.grad, W.data', y.grad, 1, 1)
  return nothing
end

function primal!(y::GraphNode{:relu,1})
  x, = y.args
  y.data .= max.(0, x.data)
  return nothing
end
function adjoint!(y::GraphNode{:relu,1})
  x, = y.args
  x.grad .+= y.grad .* (x.data .== y.data)
  return nothing
end

function primal!(z::GraphNode{:add,2})
  x, y = z.args
  z.data .= x.data .+ y.data
  return nothing
end
function adjoint!(z::GraphNode{:add,2})
  x, y = z.args
  x.grad .+= z.grad 
  y.grad .+= z.grad
  return nothing
end

function primal!(z::GraphNode{:dot,2})
  x, y = z.args
  z.data = dot(x.data, y.data)
  return nothing
end
function adjoint!(z::GraphNode{:dot,2})
  x, y = z.args
  x.grad .+= y.data .* z.grad
  y.grad .+= x.data .* z.grad
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
  y.data .= 1 ./ (1 .+ exp.(.-x.data))
  return nothing
end
function adjoint!(y::GraphNode{:sigmoid,1})
  x, = y.args
  x.grad .+= exp.(.-x.data) ./ (1 .+ exp.(.-x.data)).^2 .* y.grad
  return nothing
end

function get_cache_matrix!(cache::Dict{Symbol,Any}, key::Symbol, T, dims::Tuple)
  if !haskey(cache, key) || size(cache[key]) != dims || eltype(cache[key]) != T
    cache[key] = Matrix{T}(undef, dims...)
  end
  return cache[key]
end
function im2col_pad!(cols, X, KH, KW, pad)
  IC, H, W = size(X)
  Hp = H + 2 * pad
  Wp = W + 2 * pad
  out_h = Hp - KH + 1
  out_w = Wp - KW + 1

  col = 1
  for ow in 1:out_w
    for oh in 1:out_h
      idx = 1
      for j in 0:KW-1
        # Calculate the actual horizontal index in X
        iw = ow + j - pad
        for i in 0:KH-1
          # Calculate the actual vertical index in X
          ih = oh + i - pad
          
          # If we are in the "pad zone", value is 0. 
          # Otherwise, pull from X.
          in_bounds = (ih >= 1 && ih <= H && iw >= 1 && iw <= W)
          
          for ic in 1:IC
            if in_bounds
                cols[idx, col] = X[ic, ih, iw]
            else
                cols[idx, col] = zero(eltype(X))
            end
            idx += 1
          end
        end
      end
      col += 1
    end
  end
  return cols
end
function primal!(y::GraphNode{:conv,3})
  kernels, x, padnode = y.args
  pad = Int32(padnode.data[1])
  W = kernels.data
  X = x.data
  Y = y.data

  OC, IC, KH, KW = size(W)
  ICx, H, WW = size(X)
  @assert IC == ICx

  out_h = H + 2 * pad - KH + 1
  out_w = WW + 2 * pad - KW + 1
  @assert size(Y) == (OC, out_h, out_w)

  T = promote_type(eltype(X), eltype(W))

  Xcol = get_cache_matrix!(y.cache, :Xcol, T, (IC * KH * KW, out_h * out_w))

  im2col_pad!(Xcol, X, KH, KW, pad)

  Wcol = reshape(W, OC, IC * KH * KW)
  Ycol = reshape(Y, OC, out_h * out_w)
  mul!(Ycol, Wcol, Xcol)

  return nothing
end

function col2im_pad!(dx, cols, KH, KW, pad)
  IC, H, W = size(dx)

  Hp = H + 2 * pad
  Wp = W + 2 * pad
  out_h = Hp - KH + 1
  out_w = Wp - KW + 1

  col = 1
  for ow in 1:out_w, oh in 1:out_h
    idx = 1
    for j in 1:KW, i in 1:KH
      hi = oh + i - pad - 1
      wi = ow + j - pad - 1
      for ic in 1:IC
        if 1 <= hi <= H && 1 <= wi <= W
          dx[ic, hi, wi] += cols[idx, col]
        end
        idx += 1
      end
    end
    col += 1
  end

  return dx
end
function adjoint!(y::GraphNode{:conv,3})
  kernels, x, padnode = y.args
  pad = Int64(padnode.data[1])

  W = kernels.data
  X = x.data
  GY = y.grad

  OC, IC, KH, KW = size(W)
  ICx, H, WW = size(X)
  OCy, out_h, out_w = size(GY)

  @assert IC == ICx
  @assert OC == OCy

  T = promote_type(eltype(W), eltype(X), eltype(GY))

  Xcol = get_cache_matrix!(y.cache, :Xcol, T, (IC * KH * KW, out_h * out_w))
  im2col_pad!(Xcol, X, KH, KW, pad)

  Wcol = reshape(W, OC, IC * KH * KW)
  GYcol = reshape(GY, OC, out_h * out_w)

  dWcol = get_cache_matrix!(y.cache, :dWcol, T, (OC, IC * KH * KW))
  mul!(dWcol, GYcol, Xcol')

  dXcol = get_cache_matrix!(y.cache, :dXcol, T, (IC * KH * KW, out_h * out_w))
  mul!(dXcol, Wcol', GYcol)

  kernels.grad .+= reshape(dWcol, size(W)...)
  col2im_pad!(x.grad, dXcol, KH, KW, pad)

  return nothing
end

function primal!(y::GraphNode{:maxpool,1})
  x, = y.args
  _maxpool_primal!(y.data, x.data)
  return nothing
end
function _maxpool_primal!(Y::Array{T,3}, X::Array{T,3}) where {T}
  C, H, W = size(X)
  _, out_h, out_w = size(Y)

  kh = H ÷ out_h
  kw = W ÷ out_w

  for c in 1:C, oh in 1:out_h
    h_start = (oh - 1) * kh + 1
    h_end = oh * kh

    for ow in 1:out_w
      w_start = (ow - 1) * kw + 1
      w_end = ow * kw

      best = X[c, h_start, w_start]

      for i in h_start:h_end, j in w_start:w_end
        v = X[c, i, j]
        if v > best
          best = v
        end
      end

      Y[c, oh, ow] = best
    end
  end

  return nothing
end
function adjoint!(y::GraphNode{:maxpool,1})
  x, = y.args
  _maxpool_adjoint!(x.grad, y.grad, x.data, y.data)
  return nothing
end

function _maxpool_adjoint!(
  XG::AbstractArray{T,3},
  GY::AbstractArray{T,3},
  X::AbstractArray{T,3},
  Y::AbstractArray{T,3},
) where {T}
  C, H, W = size(X)
  _, out_h, out_w = size(Y)

  kh = H ÷ out_h
  kw = W ÷ out_w

  for c in 1:C, oh in 1:out_h
    h_start = (oh - 1) * kh + 1
    h_end = oh * kh

    for ow in 1:out_w
      w_start = (ow - 1) * kw + 1
      w_end = ow * kw

      best_i = h_start
      best_j = w_start
      best = X[c, h_start, w_start]

      for i in h_start:h_end, j in w_start:w_end
        v = X[c, i, j]
        if v > best
          best = v
          best_i = i
          best_j = j
        end
      end

      XG[c, best_i, best_j] += GY[c, oh, ow]
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
    x, mask, p_node = y.args
    p = p_node.data[1]
    global IS_TRAINING

    if IS_TRAINING[]
        mask.data = rand(size(x.data)...) .> p

        y.data .= (x.data .* mask.data) ./ (1.0 - p)
    else
        y.data .= x.data
    end
end
function adjoint!(y::GraphNode{:dropout,3})
    x, mask, p_node = y.args
    p = p_node.data[1]
    global IS_TRAINING

    if IS_TRAINING[]
        # Gradient only flows through "active" neurons,
        x.grad .+= (y.grad .* mask.data) ./ (1.0 - p)
    else
        # Even though we are not training in adjoint!,
        # the gradient flows through normally
        x.grad .+= y.grad
    end
end


function primal!(z::GraphNode{:lce})
  x, y = z.args

  m = maximum(x.data)
  lse = m + log(sum(exp.(x.data .- m)))
  cls = argmax(y.data)

  z.data[1] = -(x.data[cls] - lse)
  return nothing
end
function adjoint!(z::GraphNode{:lce})
  x, y = z.args
  ex = exp.(x.data .- maximum(x.data))
  soft = ex ./ sum(ex)
  x.grad .+= z.grad[1] .* (soft .- y.data)
  return nothing
end
