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
seed = 3
ljtrue = PotentialUQ.Potentials.LennardJones(ϵ, σ)

## Set up training and test data
num_configs = 20
r_train = Vector{PotentialUQ.Potentials.Configuration}(undef, num_configs)
for j = 1:num_configs
    file_path = "examples/Argon7/DATA_training/TEMP_$(Int(100*Temp))/seed_$seed/DATA.$(10000*j)"
    c_temp = PotentialUQ.Potentials.Configuration(file_path; atom_names = [:Ar], 
                    radii = [3.4], weights = [1.0])
    r_train[j] = c_temp
end

seed = 20
num_configs_test = 10
r_test = Vector{PotentialUQ.Potentials.Configuration}(undef, num_configs_test)
for j = 1:num_configs_test
    file_path = "examples/Argon7/DATA_training/TEMP_$(Int(100*Temp))/seed_$seed/DATA.$(100000*j)"
    c_temp = PotentialUQ.Potentials.Configuration(file_path; atom_names = [:Ar], 
                    radii = [3.4], weights = [1.0])
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
twojmax = 6
rcutfac = 1.0
snap = PotentialUQ.Potentials.SNAP(rcutfac, twojmax, r_train[1].num_atom_types)
A = PotentialUQ.Potentials.get_snap(r_train, snap)

# Fit
AA = A'*A
snap.β = pinv(AA, 1e-8)*(A'*b)
# show(stdout, "text/plain", snap.β)
# println(" ")
# Test 
e_snap = PotentialUQ.Potentials.potential_energy(r_train, snap)
println("Relative Error = ", mean(abs.(etrain - e_snap)./abs.(etrain)))

f_snap = PotentialUQ.Potentials.force(r_train, snap)
ff = vcat( (ftrain - f_snap)... ) 
fff = vcat( ftrain... )
println("Force Relative Error = ", mean( norm.(ff)./norm.(fff)  ) )

v_snap = PotentialUQ.Potentials.virial(r_train, snap)
println("Virial Error = ", mean(abs.(vtrain - v_snap)./abs.(vtrain)))
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
println(size(y))
sampler(rng::Random.AbstractRNG, d::ContinuousMultivariateDistribution) = β + 1e-1*randn(length(d))
logpdf(x::Vector) = loglikelihood(MvNormal(β, 1e-1*I(m)), x)
snap_prior = PotentialUQ.SNAP_Prior_Distribution(A, b, x, t, sampler, logpdf)
println("logpdf = ", Distributions.logpdf(snap_prior, x))
snap_likelihood(x::Vector) = loglikelihood( MvNormal( Σ*F.V'*x - y, Q ),  zeros(m) )
println("Likelihood = ", snap_likelihood(x))
dsnap = PotentialUQ.PotentialDistribution(snap, nt, snap_prior, snap_likelihood)

@time chain, samp = PotentialUQ.Sample(dsnap; return_chain = true, num_samples = 100, num_adapts = 50, verbose = true)

# snap.β = dsnap.x.β
# println("MAP Fitted β = ")
# show(stdout, "text/plain", snap.β)
# println(" ")
# e_snap = PotentialUQ.Potentials.potential_energy(r_train, snap)
# println("MAP SNAP potential_energy = ", e_snap)
# println("MAP Relative Error = ", mean(abs.(etest - e_snap)./abs.(etest)))
# v_snap = PotentialUQ.Potentials.virial(r_train, snap)
# f_snap = PotentialUQ.Potentials.force(r_train, snap)
# ff = vcat( (ftrain - f_snap)... )
# fff = vcat( ftrain... )
# println("Force Relative Error = ", mean( norm.(ff)./norm.(fff)  ) )
# println("MAP SNAP Virial = ", v_snap)
# println("MAP Virial Error = ", mean(abs.(vtest - v_snap)./abs.(vtest)))


# save("examples/Argon7/samples_snap.jld", "samples", samp, "map", snap.β)
