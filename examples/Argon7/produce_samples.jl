using LinearAlgebra: svd, svdvals, cond, pinv, I, diagm
using Statistics
using Distributions, TransformVariables
using Random
using JLD

include("../../src/PotentialUQ.jl")

num_configs = 40
r = Vector{PotentialUQ.Potentials.Configuration}(undef, num_configs)

for j = 1:num_configs
    file_path = "examples/Argon7/DATA/DATA$j"
    c = PotentialUQ.Potentials.Configuration(file_path; atom_names = [:Ar], 
                    rcutoff = [0.5], neighbor_weight = [1.0])
    r[j] = c
end

num_configs_test = 20
rtest = Vector{PotentialUQ.Potentials.Configuration}(undef, num_configs_test)
file_path = "examples/Argon7"
for j = num_configs+1:num_configs+num_configs_test
    file_path = "examples/Argon7/DATA/DATA$j"
    c = PotentialUQ.Potentials.Configuration(file_path; atom_names = [:Ar], 
                    rcutoff = [0.5], neighbor_weight = [1.0])
    rtest[j-num_configs] = c
end

ϵ = 1.0
σ = 1.0
lj = PotentialUQ.Potentials.LennardJones(ϵ, σ)
ljtest = PotentialUQ.Potentials.LennardJones(ϵ, σ)

e = PotentialUQ.Potentials.potential_energy(r, lj)
etest = PotentialUQ.Potentials.potential_energy(rtest, ljtest)

fo = PotentialUQ.Potentials.force(r, lj)
fotest = PotentialUQ.Potentials.force(rtest, ljtest)

v = PotentialUQ.Potentials.virial(r, lj)
vtest = PotentialUQ.Potentials.virial(rtest, ljtest)

v_tensor = PotentialUQ.Potentials.virial_stress(r, lj)
v_tensortest = PotentialUQ.Potentials.virial_stress(rtest, ljtest)


# SNAP
b = vcat(e, reduce(vcat, reduce(vcat, fo)), reduce(vcat, v_tensor))

rcutoff = 5.0
twojmax = 3
snap = PotentialUQ.Potentials.SNAP(rcutoff, twojmax, r[1].num_atom_types)
A = PotentialUQ.Potentials.get_snap(r, snap)
println("Shape of A: ", size(A))
show(stdout, "text/plain", A[1:20, 1:9])
println(" ")

# Fit
snap.β = A \ b 
println("Fitted β = ")
show(stdout, "text/plain", snap.β)
println(" ")
# Test 
e_snap = PotentialUQ.Potentials.potential_energy(rtest, snap)
println("SNAP potential_energy = ", e_snap)
println("Relative Error = ", mean(abs.(etest - e_snap)./abs.(etest)))
v_snap = PotentialUQ.Potentials.virial(rtest, snap)
println("SNAP Virial = ", v_snap)
println("Virial Error = ", mean(abs.(vtest - v_snap)./abs.(vtest)))

# Regularize 
println("Regularizing")
AA = A' * A 
println(" ")
snap.β = pinv(AA, 1e-6) * (A'*b)
println("Regularized Fitted β = ")
show(stdout, "text/plain", snap.β)
println(" ")
e_snap = PotentialUQ.Potentials.potential_energy(rtest, snap)
println("Regularized SNAP potential_energy = ", e_snap)
println("Regularized Relative Error = ", mean(abs.(etest - e_snap)./abs.(etest)))
v_snap = PotentialUQ.Potentials.virial(rtest, snap)
println("Regularized SNAP Virial = ", v_snap)
println("Regularized Virial Error = ", mean(abs.(vtest - v_snap)./abs.(vtest)))


############### UQ ######################
# Unknowns
n, m = size(A)

# β = pinv(A' * A, 1e-6)*(A'*b)
β = A\b
nt = (β = β, )
# println("v = ", nt)
t = PotentialUQ.namedtp_to_vec(nt)
x = TransformVariables.inverse(t, nt)

# # Prior
F = svd(A)
Σ = diagm(F.S)
scale_A = vec(std(A, dims = 2))
Q =  F.U' * ( diagm(scale_A) ) * (F.U)
Q = I(m) + 0.5*Q + 0.5*Q'
println("Q = ")
show(stdout, "text/plain", Q)
println(" ")
println(" y = ")
y =  F.U' * b
show(stdout, "text/plain", y)
println(" ")
sampler(rng::Random.AbstractRNG, d::ContinuousMultivariateDistribution) = β + 1e-1*randn(length(d))
logpdf(x::Vector) = loglikelihood(MvNormal(β, 1e-1*I(m)), x)
snap_prior = PotentialUQ.SNAP_Prior_Distribution(A, b, x, t, sampler, logpdf)
println("logpdf = ", Distributions.logpdf(snap_prior, x))
snap_likelihood(x::Vector) = loglikelihood( MvNormal( Σ * F.Vt*x, Q ),  y )
println("Likelihood = ", snap_likelihood(x))
dsnap = PotentialUQ.PotentialDistribution(snap, nt, snap_prior, snap_likelihood)

@time samp = PotentialUQ.Sample(dsnap; num_samples = 1000, num_adapts = 500, verbose = true)

snap.β = dsnap.x.β
println("MAP Fitted β = ")
show(stdout, "text/plain", snap.β)
println(" ")
e_snap = PotentialUQ.Potentials.potential_energy(rtest, snap)
println("MAP SNAP potential_energy = ", e_snap)
println("MAP Relative Error = ", mean(abs.(etest - e_snap)./abs.(etest)))
v_snap = PotentialUQ.Potentials.virial(rtest, snap)
println("MAP SNAP Virial = ", v_snap)
println("MAP Virial Error = ", mean(abs.(vtest - v_snap)./abs.(vtest)))


save("examples/Argon7/samples_snap.jld", "samples", samp)
