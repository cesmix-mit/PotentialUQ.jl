########################################################################################################
## Details
# This file will create the structure for the UQ inference problems.
# This package is designed to be used with a family of julia packages,
#           - Potential.jl (*)
#               - Creates potential types, has abstract 'Potential' type (*), let p <: 'Potential'
#               - Assume mutable struct Potential{T<:AbstractFloat}
#                            x       # Potential trainable parameters
#                            p       # Potential nontrainable parameters 
#                        end
#               - Assume mutable struct SNAP{T<:AbstractFloat}
#                            A 
#                            b 
#                            β 
#                        end
#           - PotentialLearning.jl 
#               - Fits data-driven potential
#               - We assume that potential learning updates parameters p.x
#           - PotentialUQ.jl (this package)
#               - Defines distribution over potential parameters.
#               - Initially must provide distribution over p.x (assume user provides distribution, in form of log likelihood) 
#               - In case of SNAP, potential is predetermined.
#               - Allow users to modify distribution with priors by adding to log-likelihood.
#
#########################################################################################################
module PotentialUQ

using Base: NamedTuple
using LinearAlgebra: tr, logdet, transpose, Symmetric, diag, I, pinv, diagm, svd, dot
using Statistics: cov, std, mean
using TransformVariables, Bijectors
using Distributions
using Random
import InteratomicPotentials as Potentials


include("types/types.jl")
include("samplers/samplers.jl")
include("utilities/utils.jl")
# include("optimizers/map.jl")

export state, chain, chain_diagnostics, mcmc_step, mcmc, hmc_step, hmc, Sample

end


