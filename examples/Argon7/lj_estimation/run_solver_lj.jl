using LinearAlgebra: svd, svdvals, cond, pinv, I, diagm
using Statistics
using Distributions
using Random
using JLD

include("../../../../Potentials.jl/src/Potentials.jl")

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
Temp = 0.25
samples = load("examples/Argon7/lj_estimation/samples_lj_TEMP_25.jld")
map = samples["map"]
lj = Potentials.LennardJones(map.ϵ, map.σ)
c = Potentials.Configuration("examples/Argon7/DATA_training/DATA_start", atom_names = [:Ar], radii = [3.5], weights = [1.0])

    
Nt = 1000000
dT = 1000
dt = 0.01
path = "examples/Argon7/lj_estimation/TEMP_$(Int(100*Temp))/"
# mkdir(path)
# mkdir(path*"/MAP/")
for seed = 1:100
    if seed % 10 == 0
        println("seed $seed")
    end
    path = "examples/Argon7/lj_estimation/TEMP_$(Int(100*Temp))/MAP/"
    path = path * "seed_$seed/"
    mkdir(path)
    @time rdf, pe, msd = Potentials.run_md(c, lj, Nt, path; dt = dt, dim = 2, Temp = Temp, dT = dT, seed = 60)
end

# T = dT:dT:Nt
# num_configs = length(T)
# r = Vector{Potentials.Configuration}(undef, num_configs)
# for (i,t) in enumerate(T)
#     file = "DATA.$t"
#     c_temp = Potentials.Configuration("examples/Argon7/temp/"*file; atom_names = [:Ar], 
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








