# [WIP] PotentialUQ.jl
Perform UQ for learned potentials

## Roadmap

This is planned to be implemented alongside Potential.jl (soon to exist) and PotentialLearning.jl.

- **Potential.jl** should provide access to energies, forces, and stresses (along with derivatives of each of those quantities with respect to the potential parameters).
- **PotentialLearning.jl** should implement fitting strategies for each of the potentials in Potential.jl. 
- **PotentialUQ.jl** will extend PotentialLearning.jl by providing a distribution layer. This distribution layer will keep track of the current negative log likelihood, trainable parameters, and non-trainable parameters. 
    - **PotentialUQ.jl** will also allow users to add priors to the distribution (to be implemented). 

## Basic Usage
See "/test" for current test examples.

SNAP Example:
Define A, b, β, and Q for b ∼ N(Aβ, Q) 
```julia
    A = ...         (n x m Matrix)
    b = ...         (n x 1 Vector)
    beta = ...      (m x 1 Vector)
    Q = ...         (n x n Covariance)
```

Set up corresponding structures
```julia
sampler(rng::Random.AbstractRNG, d::ContinuousMultivariateDistribution) = ...
snap_prior = PotentialUQ.SNAP_Prior_Distribution(...)
snap_likelihood(x::Vector) = ...
dsnap = PotentialUQ.PotentialDistribution(snap, named_tuple, snap_prior, snap_likelihood)
```

Produces samples using **Turing.jl**
```julia
samp = PotentialUQ.Sample(dsnap)
```

MAP estimate is placed in data structure
```julia
dsnap.x.β
```


