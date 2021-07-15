using Base: AbstractFloat
########################################################################################################
## Details
# This file will create the type structures for the UQ inference problems.
# 
#########################################################################################################
abstract type Potential{T} end

mutable struct SNAP{T<:Base.AbstractFloat} <: Potential{T}
    A :: Matrix{T}
    b :: Vector{T}
    β :: Vector{T}
end

abstract type ArbitraryDistribution{T} end

mutable struct PotentialDistribution{T<:AbstractFloat} <: ArbitraryDistribution{T}
    potential::Potential{T}       # potential defined in Potential.jl
    x                             # Container of learnable parameters
    p                             # Container of nonlearnable parameters  
    distribution                  # -log likelihood   
end

function PotentialDistribution(potential :: Potential{T}, distribution) where T<: AbstractFloat 
    pd = PotentialDistribution(potential, p.x, p.q, distribution)
    return pd
end


#########################################################################################################
## SNAP 

# Nonlearnable Q
mutable struct SNAPDistribution{T<:AbstractFloat} <: ArbitraryDistribution{T}
    snap::SNAP{T}               # SNAP potential, assume has A, b, β
    x                        # Container of learnable parameters
    p                        # Container of nonlearnable parameters
    distribution             # -log likelihood
end

function SNAPDistribution(snap :: SNAP{T}, Q) where T<: AbstractFloat
    x = (β = snap.β, )
    p = (A = snap.A, b = snap.b, Q = Q)
    
    distribution(x, p) = 0.5*tr( transpose(p.A*x.β-p.b)*(p.Q\(p.A*x.β-p.b)) + logdet(p.Q))
    pd = SNAPDistribution{T}(snap, x, p, distribution)
    return pd
end

# Learnable Q 
mutable struct SNAPQDistribution{T<:AbstractFloat} <: ArbitraryDistribution{T}
    snap::SNAP{T}               # SNAP potential, assume has A, b, β
    x                           # Container of learnable parameters
    p                           # Container of nonlearnable parameters
    distribution                # SNAP likelihood(p)
end

function SNAPQDistribution(snap :: SNAP{T}, Q) where T<: AbstractFloat
    x = (β = snap.β, Q = Q)
    p = (A = snap.A, b = snap.b)
    distribution(x, p) = 0.5*tr( transpose(p.A*x.β-p.b)*(x.Q\(p.A*x.β-p.b)) + logdet(x.Q))
    pd = SNAPQDistribution{T}(snap, x, p, distribution)
    return pd
end





