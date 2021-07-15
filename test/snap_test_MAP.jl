import PotentialUQ
using LinearAlgebra: I
using Random
Random.seed!(1234);

n = 100
β = ones(10)
A = 1e-1*randn(n, 10)
A[1:10, 1:10] += I(10)
b = randn(n)

Q = Matrix(1e-3*I(n))

snap = PotentialUQ.SNAP(A, b, β)
dsnap = PotentialUQ.SNAPDistribution(snap, Q)

@time fitsnap = PotentialUQ.MAP(dsnap)
dsnap = PotentialUQ.SNAPDistribution(snap, Q)
@time fitsnap = PotentialUQ.MAP(dsnap)

println("Solution ", A\b)
println("MAP Solution, Optimization ", fitsnap.x.β)