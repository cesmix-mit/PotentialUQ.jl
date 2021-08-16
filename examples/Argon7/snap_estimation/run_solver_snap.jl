using LinearAlgebra: svd, svdvals, cond, pinv, I, diagm
using Statistics
using Distributions
using Random
using JLD

include("../../../../Potentials.jl/src/Potentials.jl")
# include("solver_utils.jl")

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



c = Potentials.Configuration("examples/Argon7/DATA_training/DATA_start", atom_names = [:Ar], radii = [3.4], weights = [1.0])
lj = Potentials.LennardJones(ϵ, σ)
twojmax = 6
rcutfac = 0.75/3.4
snap = Potentials.SNAP(rcutfac, twojmax, 1)
β_samples = load("examples/Argon7/snap_estimation/TEMP_$(Int(100*Temp))/samples_snap.jld")["samples"] 
snap.β = load("examples/Argon7/snap_estimation/TEMP_$(Int(100*Temp))/samples_snap.jld")["mle"]


Nt = 50000
dT = 100
dt = 0.01
# mkdir("examples/Argon7/snap_estimation/TEMP_$(Int(100*Temp))/MAP/")
seed = 1
println("seed ", seed)
path = "examples/Argon7/snap_estimation/TEMP_$(Int(100*Temp))/MAP/seed_$seed/"
# mkdir(path)
@time r, rdf, pe, msd = Potentials.run_md(c, snap, Nt, path; dt = dt, dim = 2, Temp = 0.25, dT = dT, seed = seed)

# T = dT:dT:Nt
# num_configs = length(T)
# r = Vector{Potentials.Configuration}(undef, num_configs)
# for (i,t) in enumerate(T)
#     file = "DATA.$t"
#     c_temp = Potentials.Configuration("examples/Argon7/snap_estimation/TEMP_$(Int(100*Temp))/MAP/seed_1/"*file; atom_names = [:Ar], 
#                     radii = [3.4], weights = [1.0])
#     r[i] = c_temp
# end

