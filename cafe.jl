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
