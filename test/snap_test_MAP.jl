import PotentialUQ
using LinearAlgebra: I
using Random
println("Starting test of MAP method")
Random.seed!(1234);

n = 100
β = zeros(10)
A = 1e-1*randn(n, 10)
A[1:10, 1:10] += I(10)
b = randn(n)
β = A\b
σ = 1e-3 .* ones(n)
logσ = log.(σ)
snap = PotentialUQ.Potentials.SNAP(β, 0.9, 6, 1)
dsnap = PotentialUQ.SNAPQDistribution(snap, A, b, logσ)
println(dsnap( (β = β, σ = logσ)))
println( dsnap( (β = A\b, σ = logσ)) )

@time PotentialUQ.MAP(dsnap)
println(dsnap.x)

snap = PotentialUQ.Potentials.SNAP(β, 0.9, 6, 1)
dsnap = PotentialUQ.SNAPQDistribution(snap, A, b, logσ)
println(dsnap.x)
@time PotentialUQ.MAP(dsnap)

println("Solution ", A\b)
println("MAP Solution, Optimization β =  ") 
show(stdout, "text/plain", dsnap.x[1])
println("MAP Solution, Optimization σ =  ") 
show(stdout, "text/plain", exp.(dsnap.x[2]))
println(" ")

println("End of test.")