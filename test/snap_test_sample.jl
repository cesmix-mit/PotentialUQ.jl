import PotentialUQ
using LinearAlgebra: I
using Random
using Turing
Random.seed!(1234);

n = 100
β = zeros(10)
A = 1e-1*randn(n, 10)
A[1:10, 1:10] += I(10)
b = randn(n)

Q = Matrix(1e-3*I(n))

snap = PotentialUQ.SNAP(A, b, β)
dsnap = PotentialUQ.SNAPDistribution(snap, Q)

samples = PotentialUQ.sample(dsnap; num_samples = 1000)

println("MAP Solution, Sampling ", dsnap.x.β)

# summaries = describe(samples)
# show(Base.stdout, "text/plain", summaries[1])
# show(Base.stdout, "text/plain", summaries[2])
