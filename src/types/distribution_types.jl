using Base: AbstractFloat
########################################################################################################
## Details
# This file will create the type structures for the UQ inference problems.
# 
#########################################################################################################

abstract type ArbitraryDistribution end

mutable struct PotentialDistribution <: ArbitraryDistribution
    potential::Potentials.ArbitraryPotential          # potential defined in Potential.jl
    prior :: ContinuousMultivariateDistribution
    distribution::Function                   
    x::NamedTuple
    t::TransformVariables.AbstractTransform
    samples::Vector
    function PotentialDistribution(potential::Potentials.ArbitraryPotential, x::NamedTuple, prior::ContinuousMultivariateDistribution, likelihood::Function)
        t = namedtp_to_vec(x)
        distribution(x) = likelihood(x) + prior.logpdf(x)
        new(potential, prior, distribution, x, t, zeros(0))
    end
end

function (potential_dist::PotentialDistribution)(θ)
    x = inverse(potential_dist.t, θ)
    potential_dist.distribution(x) + potential_dist.prior.logpdf(x)
end

#########################################################################################################
########################################### Prior ####################################################### 
#########################################################################################################

# Define an ArbitraryPrior type 
abstract type ArbitraryPrior <: ContinuousMultivariateDistribution end

function Base.length(d::ArbitraryPrior) 
    return length(d.x)
end

Distributions.rand(rng::AbstractRNG, d::ArbitraryPrior) = d.sampler(rng, d)
function Distributions._rand!(rng::AbstractRNG, d::ArbitraryPrior, x::AbstractVector) 
    x = d.sampler(rng, d)
end

Distributions._logpdf(d::ArbitraryPrior, x::AbstractArray) = d.logpdf(x)
Bijectors.bijector(d::ArbitraryPrior) = Bijectors.Identity{1}()

########################################### LJ #######################################################
struct LJ_Prior_Distribution <: ArbitraryPrior
    y_train :: Vector{Potentials.Configuration}
    x       :: Vector{Float64}
    t       :: TransformVariables.AbstractTransform
    sampler :: Function 
    logpdf  :: Function
end

Bijectors.bijector(d::LJ_Prior_Distribution) = Bijectors.Logit{1}(0.5, 1.5)

########################################### SNAP  #######################################################
struct SNAP_Prior_Distribution <: ArbitraryPrior
    A :: Array
    b :: Vector
    x :: Vector
    t :: TransformVariables.AbstractTransform
    sampler :: Function 
    logpdf :: Function 
end
function SNAP_Prior_Distribution(A::Array, b::Vector, v::NamedTuple)
    t = namedtp_to_vec(v)
    x = inverse(t, v)
    sampler(rng::AbstractRNG, d::ArbitraryPrior) = randn(length(d))
    function logpdf(θ::Vector)
        n = length(θ)
        loglikelihood( MvNormal( zeros(n), I(n) ), θ )
    end
    SNAP_Prior_Distribution(A, b, x, t, sampler, logpdf)
end

#####################################################################################################







