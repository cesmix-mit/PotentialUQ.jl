using LinearAlgebra: svd, svdvals, cond, pinv, I, diagm, norm
using Statistics
using Distributions, TransformVariables
using Random
using JLD

include("../../../src/PotentialUQ.jl")

######################################################
##################### Setup ##########################
######################################################

ϵ = 1.0
σ = 1.0
Temp = 0.25
seed = randperm(100)
ljtrue = PotentialUQ.Potentials.LennardJones(ϵ, σ)
num_configs = 40
r_train = Vector{PotentialUQ.Potentials.Configuration}(undef, num_configs)
radii = 3.4
for j = 1:num_configs
    file_path = "examples/Argon7/DATA_training/TEMP_$(Int(Temp*100))/seed_1/DATA.$(Int(10000*j*2.5))"
    c_temp = PotentialUQ.Potentials.Configuration(file_path; atom_names = [:Ar], 
                    radii = [radii], weights = [1.0])
    r_train[j] = c_temp
end


num_configs_test = 25
r_test = Vector{PotentialUQ.Potentials.Configuration}(undef, num_configs_test)
for j = 1:num_configs_test
    file_path = "examples/Argon7/DATA_training/TEMP_$(Int(Temp*100))/seed_1/DATA.$(Int(10000*j*3))"
    c_temp = PotentialUQ.Potentials.Configuration(file_path; atom_names = [:Ar], 
                    radii = [radii], weights = [1.0])
    r_test[j] = c_temp 
end


etrain = PotentialUQ.Potentials.potential_energy(r_train, ljtrue)
etest = PotentialUQ.Potentials.potential_energy(r_test, ljtrue)

ftrain = PotentialUQ.Potentials.force(r_train, ljtrue)
ftest = PotentialUQ.Potentials.force(r_test, ljtrue)

vtrain = PotentialUQ.Potentials.virial(r_train, ljtrue)
vtest = PotentialUQ.Potentials.virial(r_test, ljtrue)

vttrain = PotentialUQ.Potentials.virial_stress(r_train, ljtrue)
vttest = PotentialUQ.Potentials.virial_stress(r_train, ljtrue)


###################### Setup SNAP ####################
# SNAP
b = vcat(etrain, reduce(vcat, reduce(vcat, ftrain)), reduce(vcat, vttrain))
# b = etrain
twojmax = 6
rcutfac = 0.75/radii
snap = PotentialUQ.Potentials.SNAP(rcutfac, twojmax, r_train[1].num_atom_types)
A = PotentialUQ.Potentials.get_snap(r_train, snap)

# Fit
snap.β = A\b
# show(stdout, "text/plain", snap.β)
# println(" ")
# Test 
e_snap = PotentialUQ.Potentials.potential_energy(r_test, snap)
error = mean(abs.(etest - e_snap)./abs.(etest))

println("Relative Error = ", error)

f_snap = PotentialUQ.Potentials.force(r_test, snap)
ff = vcat( (ftest - f_snap)... ) 
fff = vcat( ftest... )
error = mean( norm.(ff)./norm.(fff)  )
println("Force Relative Error = ",  error)

v_snap = PotentialUQ.Potentials.virial(r_test, snap)
error = mean(abs.(vtest - v_snap)./abs.(vtest))
println("Virial Error = ", error)
println(" ")


######################################################
##################### UQ #############################
######################################################

# Unknowns
n, m = size(A)

# β = pinv(A' * A, 1e-6)*(A'*b)
β = snap.β
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
y =  copy(F.U' * b)

# sampler(rng::Random.AbstractRNG, d::ContinuousMultivariateDistribution) = β + 1e-1*randn(length(d))
# logpdf(x::Vector) = loglikelihood(MvNormal(β, 1e-1*I(m)), x)
# snap_prior = PotentialUQ.SNAP_Prior_Distribution(A, b, x, t, sampler, logpdf)
snap_prior = PotentialUQ.SNAP_Prior_Distribution(A, b, nt)
println("logpdf = ", Distributions.logpdf(snap_prior, x))
snap_likelihood(x::Vector) = loglikelihood( MvNormal( Σ*F.V'*x - y, Q ),  zeros(m) )
println("Likelihood = ", snap_likelihood(x))
dsnap = PotentialUQ.PotentialDistribution(snap, nt, snap_prior, snap_likelihood)

@time chain, samp = PotentialUQ.Sample(dsnap; return_chain = true, num_samples = 1000, num_adapts = 500, verbose = true)

snap.β = dsnap.x.β
println("MAP Fitted β = ")
show(stdout, "text/plain", snap.β)
println(" ")
e_snap = PotentialUQ.Potentials.potential_energy(r_test, snap)
println("MAP SNAP potential_energy = ", e_snap)
println("MAP Relative Error = ", mean(abs.(etest - e_snap)./abs.(etest)))
v_snap = PotentialUQ.Potentials.virial(r_test, snap)
f_snap = PotentialUQ.Potentials.force(r_test, snap)
ff = vcat( (ftest - f_snap)... )
fff = vcat( ftest... )
println("Force Relative Error = ", mean( norm.(ff)./norm.(fff)  ) )
println("MAP SNAP Virial = ", v_snap)
println("MAP Virial Error = ", mean(abs.(vtest - v_snap)./abs.(vtest)))


save("examples/Argon7/snap_estimation/TEMP_25/samples_snap.jld", "samples", samp, "map", snap.β, "A", A, "mle", A\b)
