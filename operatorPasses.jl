include("graphNodes.jl")
using LinearAlgebra
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

# NOTE: [ Conv Helpers ] ------------------------------------------------------
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
  @inbounds for ow in 1:out_w
    for oh in 1:out_h

      idx = 1
      for j in 1:KW
        for i in 1:KH
          hi = oh + i - pad - 1
          wi = ow + j - pad - 1
          for ic in 1:IC
            if 1 <= hi <= H && 1 <= wi <= W
              cols[idx, col] = X[ic, hi, wi]
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
# NOTE: ] Conv Helpers [ ------------------------------------------------------
function primal!(y::GraphNode{:conv,3})
  kernels, x, padnode = y.args
  pad = padnode.data[1]

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
### NOTE: [ Conv Helpers ] ----------------------------------------------------
function col2im_pad!(dx, cols, KH, KW, pad)
  IC, H, W = size(dx)

  Hp = H + 2 * pad
  Wp = W + 2 * pad
  out_h = Hp - KH + 1
  out_w = Wp - KW + 1

  fill!(dx, zero(eltype(dx)))

  col = 1
  @inbounds for ow in 1:out_w
    for oh in 1:out_h
      idx = 1
      for j in 1:KW
        for i in 1:KH
          hi = oh + i - pad - 1
          wi = ow + j - pad - 1
          for ic in 1:IC
            if 1 <= hi <= H && 1 <= wi <= W
              dx[ic, hi, wi] += cols[idx, col]
            end
            idx += 1
          end
        end
      end
      col += 1
    end
  end

  return dx
end
function col2im_pad_add!(dx, cols, KH, KW, pad)
  IC, H, W = size(dx)

  Hp = H + 2 * pad
  Wp = W + 2 * pad
  out_h = Hp - KH + 1
  out_w = Wp - KW + 1

  col = 1
  @inbounds for ow in 1:out_w
    for oh in 1:out_h
      idx = 1
      for j in 1:KW
        for i in 1:KH
          hi = oh + i - pad - 1
          wi = ow + j - pad - 1
          for ic in 1:IC
            if 1 <= hi <= H && 1 <= wi <= W
              dx[ic, hi, wi] += cols[idx, col]
            end
            idx += 1
          end
        end
      end
      col += 1
    end
  end

  return dx
end
### NOTE: ] Conv Helpers [ ----------------------------------------------------
function adjoint!(y::GraphNode{:conv,3})
  kernels, x, padnode = y.args
  pad = padnode.data[1]

  W = kernels.data
  X = x.data
  GY = y.grad

  OC, IC, KH, KW = size(W)
  ICx, H, WW = size(X)
  OCy, out_h, out_w = size(GY)

  @assert IC == ICx
  @assert OC == OCy

  T = promote_type(eltype(W), eltype(X), eltype(GY))

  # to samo Xcol co w primal
  Xcol = get_cache_matrix!(y.cache, :Xcol, T, (IC * KH * KW, out_h * out_w))
  im2col_pad!(Xcol, X, KH, KW, pad)

  Wcol = reshape(W, OC, IC * KH * KW)
  GYcol = reshape(GY, OC, out_h * out_w)

  # dW = GYcol * Xcol'
  dWcol = get_cache_matrix!(y.cache, :dWcol, T, (OC, IC * KH * KW))
  mul!(dWcol, GYcol, Xcol')

  # dxcol = Wcol' * GYcol
  dXcol = get_cache_matrix!(y.cache, :dXcol, T, (IC * KH * KW, out_h * out_w))
  mul!(dXcol, Wcol', GYcol)

  # akumulacja do gradients
  kernels.grad .+= reshape(dWcol, size(W)...)
  col2im_pad_add!(x.grad, dXcol, KH, KW, pad)

  return nothing
end
# NOTE: CONVOLUTION END

# TODO: Optimize
function primal!(y::GraphNode{:maxpool,1})
  x, = y.args
  X = x.data
  Y = y.data

  C, H, W = size(X)
  _, out_h, out_w = size(Y)

  kh = H ÷ out_h
  kw = W ÷ out_w

  @inbounds for c in 1:C
    for oh in 1:out_h
      h_start = (oh - 1) * kh + 1
      h_end = oh * kh

      for ow in 1:out_w
        w_start = (ow - 1) * kw + 1
        w_end = ow * kw

        best = X[c, h_start, w_start]

        for i in h_start:h_end
          for j in w_start:w_end
            v = X[c, i, j]
            if v > best
              best = v
            end
          end
        end

        Y[c, oh, ow] = best
      end
    end
  end

  return nothing
end
function adjoint!(y::GraphNode{:maxpool,1})
  x, = y.args
  X = x.data
  GY = y.grad
  XG = x.grad

  C, H, W = size(X)
  _, out_h, out_w = size(y.data)

  kh = H ÷ out_h
  kw = W ÷ out_w

  @inbounds for c in 1:C
    for oh in 1:out_h
      h_start = (oh - 1) * kh + 1
      h_end = oh * kh

      for ow in 1:out_w
        w_start = (ow - 1) * kw + 1
        w_end = ow * kw

        best_i = h_start
        best_j = w_start
        best = X[c, h_start, w_start]

        for i in h_start:h_end
          for j in w_start:w_end
            v = X[c, i, j]
            if v > best
              best = v
              best_i = i
              best_j = j
            end
          end
        end

        XG[c, best_i, best_j] += GY[c, oh, ow]
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

# TODO: Implement zeros for eval
function primal!(y::GraphNode{:dropout,3})
  x, probnode, masknode = y.args
  p = probnode.data[1]

  T = eltype(x.data)
  scale = inv(one(T) - p)
  zeroT = zero(T)

  xd = x.data
  yd = y.data
  md = masknode.data

  @inbounds for i in eachindex(xd, yd, md)
    if rand() < p
      md[i] = zeroT
      yd[i] = zeroT
    else
      md[i] = scale
      yd[i] = xd[i] * scale
    end
  end

  return nothing
end
function adjoint!(y::GraphNode{:dropout,3})
  x, probnode, masknode = y.args
  xg = x.grad
  yg = y.grad
  md = masknode.data

  @inbounds for i in eachindex(xg, yg, md)
    xg[i] += yg[i] * md[i]
  end

  return nothing
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

function primal!(y::GraphNode{:softmax})
  x = y.args[1]
  xd = x.data
  yd = y.data
  m = maximum(xd)
  @. yd = exp(xd - m)
  s = sum(yd)
  @. yd = yd / s
  return nothing
end
function adjoint!(y::GraphNode{:softmax})
  x = y.args[1]
  s = y.data
  g = y.grad
  xg = x.grad
  gs = sum(g .* s)
  @. xg += s * (g - gs)
  return nothing
end
