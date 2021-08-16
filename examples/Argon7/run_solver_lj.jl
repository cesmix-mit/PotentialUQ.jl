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

lj = Potentials.LennardJones(ϵ, σ)
c = Potentials.Configuration("examples/Argon7/DATA_training/DATA_start", atom_names = [:Ar], radii = [3.5], weights = [1.0])
# c = init_position(c, 2)
# c = init_velocity(c, 2, 0.25*Tc)
for Temp = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    println("Temp $Temp")
    c = Potentials.Configuration("examples/Argon7/DATA_training/DATA_start", atom_names = [:Ar], radii = [3.5], weights = [1.0])
    Nt = 1000000
    dT = 1000
    dt = 0.01
    path = "examples/Argon7/DATA_training/TEMP_$(Int(100*Temp))/"
    # mkdir(path)
    for seed = 1:100
        if seed % 10 == 0
            println("seed $seed")
        end
        path = "examples/Argon7/DATA_training/TEMP_$(Int(100*Temp))/"
        path = path * "seed_$seed/"
        # mkdir(path)
        @time rdf, pe, msd = Potentials.run_md(c, lj, Nt, path; dt = dt, dim = 2, Temp = Temp, dT = dT, seed = seed)
    end
end

# Nt = 1000000
# dT = 1000
# dt = 0.01
# Temp = 0.30
# seed = 2
# T = dT:dT:Nt
# num_configs = length(T)
# r = Vector{PotentialUQ.Potentials.Configuration}(undef, num_configs)
# for (i,t) in enumerate(T)
#     file = "DATA.$t"
#     c_temp = PotentialUQ.Potentials.Configuration("examples/Argon7/DATA_training/TEMP_30/seed_$seed/"*file; atom_names = [:Ar], 
#                     radii = [3.5], weights = [1.0])
#     r[i] = c_temp
# end

# for seed = 2:100
#     path = "examples/Argon7/DATA_md/LJ/seed_$seed/"
#     mkdir(path)
#     Nt = 50000
#     dT = 100
#     println("Seed $seed")
#     @time rdf, pe, msd = Potentials.run_md_lj(c, Nt, path; dim = 2, Temp = 0.75, dT = dT, seed = seed)
# end








