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
c = Potentials.Configuration("examples/Argon7/2/DATA", atom_names = [:Ar], rcutoff = [2.25*σ], neighbor_weight = [1.0])
c = init_position(c, 2)
c = init_velocity(c, 2, 0.25*Tc)
Nt = 500000
@time ct, t = solve(lj, c, 0.005, Nt, 300.0*Tc .* ones(Nt); save_dt = 100);

plot(t, hcat(get_positions(ct)...)')
plot(t, Potentials.potential_energy(ct, lj))
animate_atoms(ct, t; dT = 10)
dT = Int(length(t) / 100)
num = 1
for j = 1:dT:length(t)
    c_save = ct[j]
    file = "examples/Argon7/DATA/DATA$num"
    Potentials.save_as_lammps_data(c_save; file = file)
    num += 1
end