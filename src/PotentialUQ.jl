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
using LinearAlgebra: tr, logdet, transpose, Symmetric, diag, I, pinv, diagm, svd
using Statistics: cov, std, mean
using SciMLBase, GalacticOptim, Optim 
using TransformVariables, Turing, Bijectors
# using InteractiveUtils: @code_warntype
# using DynamicHMC, LogDensityProblems
# using AdvancedHMC, ForwardDiff
using Distributions, UnPack
using Random
import InteratomicPotentials as Potentials
include("distribution_types.jl")
include("distributions.jl")
include("utils.jl")
include("map.jl")
include("sampling.jl")
end


