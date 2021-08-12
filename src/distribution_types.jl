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
        new(potential, prior, likelihood, x, t, zeros(0))
    end
end

function (potential_dist::PotentialDistribution)(θ)
    x = inverse(potential_dist.trans, θ)
    potential_dist.distribution(x)
end

#########################################################################################################
########################################### Prior ####################################################### 
#########################################################################################################

########################################### LJ #######################################################
struct LJ_Prior_Distribution <: ContinuousMultivariateDistribution
    y_train :: Vector{Potentials.Configuration}
    x       :: Vector{Float64}
    t       :: TransformVariables.AbstractTransform
    sampler :: Function 
    logpdf  :: Function
end
function Base.length(d::LJ_Prior_Distribution)
    return length(d.x)
end

Distributions.rand(rng::AbstractRNG, d::LJ_Prior_Distribution) = d.sampler(rng, d)
Distributions.logpdf(d::LJ_Prior_Distribution, x::Vector) = d.logpdf(x)
Bijectors.bijector(d::LJ_Prior_Distribution) = Bijectors.Logit{1}(0.5, 1.5)

########################################### SNAP  #######################################################
struct SNAP_Prior_Distribution <: ContinuousMultivariateDistribution
    A :: Array{Float64}
    b :: Vector{Float64}
    x :: Vector{Float64}
    t :: TransformVariables.AbstractTransform
    sampler :: Function 
    logpdf :: Function 
end
function SNAP_Prior_Distribution(A::Array{Float64}, b::Vector{Float64}, v::NamedTuple)
    t = namedtp_to_vec(v)
    x = inverse(t, v)
    sampler(rng::AbstractRNG, d::ContinuousMultivariateDistribution) = randn(length(d))
    function logpdf(θ::Vector)
        n = length(θ)
        loglikelihood( MvNormal( zeros(n), I(n) ), θ )
    end
    SNAP_Prior_Distribution(A, b, x, t, sampler, logpdf)
end         

function Base.length(d::SNAP_Prior_Distribution) 
    return length(d.x)
end

Distributions.rand(rng::AbstractRNG, d::SNAP_Prior_Distribution) = d.sampler(rng, d)
Distributions.logpdf(d::SNAP_Prior_Distribution, x::Vector) = d.logpdf(x)
Bijectors.bijector(d::SNAP_Prior_Distribution) = Bijectors.Identity{1}()





