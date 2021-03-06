import PotentialUQ
using LinearAlgebra: svd, svdvals, cond, pinv, I, diagm, norm
using Statistics
using Distributions
using TransformVariables
using Random
using Turing

num_configs = 40
r = Vector{PotentialUQ.Potentials.Configuration}(undef, num_configs)
file_path = "../examples/GaN/DATA"
for j = 1:num_configs
    c = PotentialUQ.Potentials.load_lammps(file_path * "/" * string(j) * "/DATA"; atom_names = [:Ga, :N], 
                    radii = [0.5, 0.5], weights = [1.0, 0.5], boundary_type = ["p", "p", "p"], units = "metal")
    r[j] = c
end

num_configs_test = 20
rtest = Vector{PotentialUQ.Potentials.Configuration}(undef, num_configs_test)
file_path = "../examples/GaN/DATA"
for j = 1:num_configs_test
    c = PotentialUQ.Potentials.load_lammps(file_path * "/" * string(40+j) * "/DATA"; atom_names = [:Ga, :N], 
            radii = [0.5, 0.5], weights = [1.0, 0.5], boundary_type = ["p", "p", "p"], units = "metal")
    rtest[j] = c
end

# Set up true potential
ϵ_Ga_Ga = 0.643
σ_Ga_Ga = 2.390
ϵ_N_N   = 1.474
σ_N_N   = 1.981
A_Ga_N  = 608.54
ρ_Ga_N  = 0.435
q_Ga    = 3.0
q_N     = -3.0
ϵ0      = 55.26349406

lj_Ga_Ga = PotentialUQ.Potentials.LennardJones(ϵ_Ga_Ga,σ_Ga_Ga)
lj_N_N   = PotentialUQ.Potentials.LennardJones(ϵ_N_N, σ_N_N)
bm_Ga_N  = PotentialUQ.Potentials.BornMayer(A_Ga_N,ρ_Ga_N)
c_Ga_N   = PotentialUQ.Potentials.Coulomb(q_Ga, q_N, ϵ0)
gan = PotentialUQ.Potentials.GaN(lj_Ga_Ga, lj_N_N, bm_Ga_N, c_Ga_N)
gantest = PotentialUQ.Potentials.GaN(lj_Ga_Ga, lj_N_N, bm_Ga_N, c_Ga_N)

println(gan)
pe = PotentialUQ.Potentials.potential_energy(r, gan)
ptest = PotentialUQ.Potentials.potential_energy(rtest, gantest)
println("GaN Energy ", pe)
fo = PotentialUQ.Potentials.force(r, gan)
fotest = PotentialUQ.Potentials.force(rtest, gantest)
println("GaN Forces ")
show(stdout, "text/plain", fo[1][1:3])
println(" ")
v = PotentialUQ.Potentials.virial(r, gan)
vtest = PotentialUQ.Potentials.virial(rtest, gantest)

println("GaN virial ", v)
v_tensor = PotentialUQ.Potentials.virial_stress(r, gan)
println("GaN Virial Tensor ", v_tensor)
println(" ")
# Form right hand side
b = vcat(pe, reduce(vcat, reduce(vcat, fo)), reduce(vcat, v_tensor))
# b = [pe; vec(v_tensor)]
println("Right hand side shape: ", size(b))
show(stdout, "text/plain", b[1:10])
println(" ")

# Form SNAP A matrix
twojmax = 6
rcutfac = 4.0
snap = PotentialUQ.Potentials.SNAP(rcutfac, twojmax, r[1])
A = PotentialUQ.Potentials.get_snap(r, snap)
println("Shape of A: ", size(A))
show(stdout, "text/plain", A[1:20, 1:9])
println(" ")

# Fit
snap.β = A \ b 
println("Fitted β = ")
show(stdout, "text/plain", snap.β)
println(" ")
# Test 
pe_snap = PotentialUQ.Potentials.potential_energy(rtest, snap)
println("SNAP potential_energy = ", pe_snap)
println("Relative Error = ", mean(abs.(ptest - pe_snap)./abs.(ptest)))
f_snap = PotentialUQ.Potentials.force(rtest, snap)
show(stdout, "text/plain", vcat(fotest...)[1:5])
println(" ")
show(stdout, "text/plain", vcat(f_snap...)[1:5])
println(" ")
ff = vcat( (fotest - f_snap)... ) 
fff = vcat( fotest... )
println("Force Relative Error = ", mean( norm.(ff)./norm.(fff)  ) )
v_snap = PotentialUQ.Potentials.virial(rtest, snap)
println("SNAP Virial = ", v_snap)
println("Virial Error = ", mean(abs.(vtest - v_snap)./abs.(vtest)))

# Regularize 
println("Regularizing")
AA = A' * A 
println(" ")
snap.β = pinv(AA, 1e-6) * (A'*b)
println("Regularized Fitted β = ")
show(stdout, "text/plain", snap.β)
println(" ")
pe_snap = PotentialUQ.Potentials.potential_energy(rtest, snap)
println("Regularized SNAP potential_energy = ", pe_snap)
println("Regularized Relative Error = ", mean(abs.(ptest - pe_snap)./abs.(ptest)))
f_snap = PotentialUQ.Potentials.force(rtest, snap)
ff = vcat( (fotest - f_snap)... ) 
fff = vcat( fotest... )
println("Force Relative Error = ", mean( norm.(ff)./norm.(fff)  ) )
v_snap = PotentialUQ.Potentials.virial(rtest, snap)
println("Regularized SNAP Virial = ", v_snap)
println("Regularized Virial Error = ", mean(abs.(vtest - v_snap)./abs.(vtest)))


############### UQ ######################
# Unknowns
n, m = size(A)

β = pinv(A' * A, 1e-6)*(A'*b)
nt = (β = β, )
# println("v = ", nt)
t = PotentialUQ.namedtp_to_vec(nt)
x = TransformVariables.inverse(t, nt)

# # Prior
F = svd(A)
Σ = diagm(F.S)
scale_A = vcat(200.0 .+ 0.0 .* pe, 1.0 .+ 0.0 .*reduce(vcat, reduce(vcat,fo)), 1.0 .+ 0.0 .*reduce(vcat, v_tensor))
Q =  F.U' * ( diagm(scale_A) ) * (F.U)
Q = 1e-2*I(m) + 0.5*Q + 0.5*Q'
println("Q = ")
show(stdout, "text/plain", Q)
println(" ")
println(" y = ")
y =  F.U' * b
show(stdout, "text/plain", y)
println(" ")
sampler(rng::Random.AbstractRNG, d::ContinuousMultivariateDistribution) = β + 1e-1*randn(length(d))
logpdf(x::Vector) = loglikelihood(MvNormal(β, 1e-1*I(m)), x)
snap_prior = PotentialUQ.SNAP_Prior_Distribution(A, b, x, t, sampler, logpdf)
println("logpdf = ", Distributions.logpdf(snap_prior, x))
snap_likelihood(x::Vector) = loglikelihood( MvNormal( Σ * F.Vt*x, Q ),  y )
println("Likelihood = ", snap_likelihood(x))
dsnap = PotentialUQ.PotentialDistribution(snap, nt, snap_prior, snap_likelihood)

@time samp = PotentialUQ.Sample(dsnap; num_samples = 1000, num_adapts = 500, verbose = true)

snap.β = dsnap.x.β
println("MAP Fitted β = ")
show(stdout, "text/plain", snap.β)
println(" ")
pe_snap = PotentialUQ.Potentials.potential_energy(rtest, snap)
println("MAP SNAP potential_energy = ", pe_snap)
println("MAP Relative Error = ", mean(abs.(ptest - pe_snap)./abs.(ptest)))
f_snap = PotentialUQ.Potentials.force(rtest, snap)
ff = vcat( (fotest - f_snap)... ) 
fff = vcat( fotest... )
println("Force Relative Error = ", mean( norm.(ff)./norm.(fff)  ) )
v_snap = PotentialUQ.Potentials.virial(rtest, snap)
println("MAP SNAP Virial = ", v_snap)
println("MAP Virial Error = ", mean(abs.(vtest - v_snap)./abs.(vtest)))

# using Plots 

# plot(pinv(AA, 1e-6) * (A'*b))
# plot!(vec(m), ribbon = vec(s), fillalpha = 0.5)
# savefig("snap_map_samples.png")