using LinearAlgebra: svd, svdvals, cond, pinv, I, diagm
using Statistics
using Distributions
using Random
using JLD

include("../../../Potentials.jl/src/Potentials.jl")
include("solver_utils.jl")

# Parameters
kb              = 8.617e-5 # eV/K
Tc              = 120.0    # K
μ               = 39.948   # u
ϵ               = Tc*kb    # 0.0103404 eV 
σ               = 3.4      # Angstroms Å

# Use reduced units 
ϵ               = 1.0
σ               = 1.0
μ               = 1.0 
Tc_kb           = 1.0 
Tc              = 1.0

β_samples = load("examples/Argon7/samples_snap.jld")["samples"] 
twojmax = 6
rcutfac = 3.4
lj = Potentials.LennardJones(ϵ, σ)
snap = Potentials.SNAP(rcutfac, twojmax, 1)
snap.β = load("examples/Argon7/samples_snap.jld")["map"]
show(stdout, "text/plain", snap)
println(" ")
c = Potentials.Configuration("examples/Argon7/DATA_training/DATA2", atom_names = [:Ar], radii = [0.5], weights = [1.0])
# c = init_position(c, 2)
# c = init_velocity(c, 2, 0.25*Tc)
Nt = 50000
dT = 100
for b_index = 20:100
    println("β index ", b_index)
    mkdir("examples/Argon7/DATA_md/SNAP/sample_$b_index/")
    snap.β = β_samples[b_index*9+1]
    for seed = 1:25
        println("seed ", seed)
        path = "examples/Argon7/DATA_md/SNAP/sample_$b_index/seed_$seed/"
        mkdir(path)
        @time rdf, pe, msd = Potentials.run_md_snap(c, snap, Nt, path; dim = 2, Temp = 0.70, dT = dT, seed = seed)
    end
end

# T = dT:dT:Nt
# num_configs = length(T)
# r = Vector{Potentials.Configuration}(undef, num_configs)
# for (i,t) in enumerate(T)
#     file = "DATA.$t"
#     c = Potentials.Configuration("examples/Argon7/DATA_md/SNAP/MAP/seed_1/"*file; atom_names = [:Ar], 
#                     radii = [0.5], weights = [1.0])
#     r[i] = c
# end

