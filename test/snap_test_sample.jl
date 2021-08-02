# include("../../Potentials.jl/src/Potentials.jl")
# using .Potentials
import PotentialUQ
using LinearAlgebra: I, diagm
using Statistics: mean, std
using Random
println("Starting test of sampler")
Random.seed!(1234);

n = 10
β = zeros(10)
A = 1e-1*randn(n, 10)
A[1:10, 1:10] += I(10)
b = randn(n)

β = A\b
σ = 1e-3 .* ones(n)
logσ = log.(σ)
snap = PotentialUQ.Potentials.SNAP(β, 0.9, 6, 1)
dsnap = PotentialUQ.SNAPQDistribution(snap, A, b, logσ)
# dsnap = PotentialUQ.SNAPDistribution(snap, A, b, diagm(σ))
@time samp = PotentialUQ.Sample(dsnap; num_adapts = 100)
println("Samples ", size(samp))
println("Samples ", samp[1, :, 1])
m = mean(samp[:, :, 1], dims = 1)
s = std(samp[:, :, 1], dims = 1)
println("Mean ", m, "\t")
println("Std ", s)
# show(stdout, "text/plain", dsnap.samples[1])
# println(" ")
println("Solution ", A\b)
println("MAP Solution, Optimization β =  ") 
show(stdout, "text/plain", m[1:10])
println("MAP Solution, Optimization σ =  ") 
show(stdout, "text/plain", exp.(m[11:end]))

