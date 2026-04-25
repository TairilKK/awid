# cafe: Convolutional Architecture for Fast Evaluation
include("operators.jl")

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

const IS_TRAINING = Ref(true)
trainmode!() = (IS_TRAINING[] = true)
evalmode!() = (IS_TRAINING[] = false)
