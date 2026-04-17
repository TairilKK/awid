# cafe: Convolutional Architecture for Fast Evaluation
abstract type Operator end
const Chain = Vector{Operator}
struct Sigmoid <: Operator end
struct ReLU <: Operator end
struct Dense <: Operator
  insize::Int64
  outsize::Int64
end

# INFO: My work: 
# =========================
# ========= START =========
# =========================
struct Conv <: Operator
  kernel_size::Tuple{Int64,Int64}
  in_channels::Int64
  out_channels::Int64
  pad::Int64
  bias::Bool
end

struct MaxPool
  kernel_size::Tuple{Int64,Int64}
end

struct Flatten <: Operator end

struct Dropout <: Operator
  prob::Float64
end
# INFO: My work: 
# =========================
# =========  END  =========
# =========================

relu() = ReLU()
sigmoid() = Sigmoid()
dense(pair::Pair{Int64,Int64}) =
  Dense(first(pair), last(pair))
dense(pair::Pair{Int64,Int64}, activation) =
  tuple(dense(pair), activation())

# INFO: My work: 
# =========================
# ========= START =========
# =========================
conv(kernel, pair; pad=0, bias=true) =
  Conv(kernel, first(pair), last(pair), pad, bias)

maxpool(kernel) = MaxPool(kernel)

flatten() = Flatten()

function dropout(p::Float64)
  @assert 0.0 <= p <= 1.0
  return Dropout(p)
end
# INFO: My work: 
# =========================
# =========  END  =========
# =========================

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
