abstract type Operator end
const Chain = Vector{Operator}

struct Sigmoid <: Operator end
struct ReLU <: Operator end
struct Dense <: Operator
  insize::Int64
  outsize::Int64
end
struct Conv <: Operator
  kernel_size::Tuple{Int64,Int64}
  in_channels::Int64
  out_channels::Int64
  pad::Int64
  bias::Bool
end
struct MaxPool <: Operator
  kernel_size::Tuple{Int64,Int64}
end
struct Flatten <: Operator end
struct DropOut <: Operator
  prob::Float32
end

relu() = ReLU()
sigmoid() = Sigmoid()
dense(pair::Pair{Int64,Int64}) =
  Dense(first(pair), last(pair))
dense(pair::Pair{Int64,Int64}, activation) =
  tuple(dense(pair), activation())
conv(kernel, pair; pad=0, bias=true) =
  Conv(kernel, first(pair), last(pair), pad, bias)
maxpool(kernel) = MaxPool(kernel)
flatten() = Flatten()
function dropout(p::Float32)
  @assert 0.0f0 <= p <= 1.0f0
  return DropOut(p)
end
