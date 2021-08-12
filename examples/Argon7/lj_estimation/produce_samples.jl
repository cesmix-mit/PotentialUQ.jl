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
seed = 10
ljtrue = PotentialUQ.Potentials.LennardJones(ϵ, σ)

## Set up training and test data
num_configs = 50
r_train = Vector{PotentialUQ.Potentials.Configuration}(undef, num_configs)
for j = 1:num_configs
    file_path = "examples/Argon7/DATA_training/TEMP_$(Int(100*Temp))/seed_$seed/DATA.$(5*1000*j)"
    c = PotentialUQ.Potentials.Configuration(file_path; atom_names = [:Ar], 
                    radii = [3.4], weights = [1.0])
    r_train[j] = c
end

seed = 20
num_configs_test = 50
r_test = Vector{PotentialUQ.Potentials.Configuration}(undef, num_configs_test)
for j = num_configs+1:num_configs+num_configs_test
    file_path = "examples/Argon7/DATA_training/TEMP_$(Int(100*Temp))/seed_$seed/DATA.$(5*1000*j)"
    c = PotentialUQ.Potentials.Configuration(file_path; atom_names = [:Ar], 
                    radii = [3.4], weights = [1.0])
    r_test[j-num_configs] = c
end


etrain = PotentialUQ.Potentials.potential_energy(r_train, ljtrue)
etest = PotentialUQ.Potentials.potential_energy(r_test, ljtrue)

ftrain = PotentialUQ.Potentials.force(r_train, ljtrue)
ftest = PotentialUQ.Potentials.force(r_test, ljtrue)

vtrain = PotentialUQ.Potentials.virial(r_train, ljtrue)
vtest = PotentialUQ.Potentials.virial(r_test, ljtrue)

######################################################
##################### UQ #############################
######################################################
# Unknowns

ϵ = 0.9
σ = 1.1
lj_train = PotentialUQ.Potentials.LennardJones(ϵ, σ)
nt = (ϵ = ϵ, σ = σ)
t = PotentialUQ.namedtp_to_vec(nt)
x = TransformVariables.inverse(t, nt)

# # Prior
function sampler(rng::Random.AbstractRNG, d::ContinuousMultivariateDistribution)
    x1 = 0.5+rand() 
    x2 = 0.5+rand()
    return [x1, x2]
end
logpdf(x::Vector) = loglikelihood(Uniform(0.5, 1.5), x[1]) + loglikelihood(Uniform(0.5, 1.5), x[2])
lj_prior = PotentialUQ.LJ_Prior_Distribution(r_train, x, t, sampler, logpdf)

function lj_likelihood(x::Vector)
    lj_temp = PotentialUQ.Potentials.LennardJones(x[1], x[2])
    ehat = PotentialUQ.Potentials.potential_energy(r_train, lj_temp)
    fhat = PotentialUQ.Potentials.force(r_train, lj_temp)
    vhat = PotentialUQ.Potentials.virial(r_train, lj_temp)
    
    error = 0.5*sqrt(mean( (ehat - etrain).^2 ))
    ff = vcat( (fhat - ftrain)... )
    error += 0.5* mean( norm.(ff) )
    error += 0.5*sqrt( mean( (vhat - vtrain).^2 ) )
    return -error
end

println("prior = ", logpdf(x))
println("Likelihood = ", lj_likelihood(x))

dlj = PotentialUQ.PotentialDistribution(lj_train, nt, lj_prior, lj_likelihood)

@time chain, samp = PotentialUQ.Sample(dlj; return_chain = true, num_samples = 1000, num_adapts = 500, verbose = true)

# corner(chain)


save("examples/Argon7/lj_estimation/samples_lj_TEMP_$(Int(100*Temp)).jld", "samples", samp, "map", dlj.x)
