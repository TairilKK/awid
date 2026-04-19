abstract type Loss end
struct BinaryCrossEntropy <: Loss end
struct LogitCrossEntropy <: Loss end

bce(output, target) = BinaryCrossEntropy()(output, target)
lce(output, target) = LogitCrossEntropy()(output, target)
