using Statistics
using PyCall
pygui(:tk)
import PyPlot; const plt = PyPlot
import InteratomicPotentials as Potentials

n=50
########################################################################################
########################################################################################

n = 50
pe_true = zeros(5000, n)
pe_snap = zeros(5000, n)
pe_lj = zeros(5000, n)
for j = 1:n
    t, pe = Potentials.get_pe("examples/Argon13/DATA_training/true_ensemble/seed_$j/tmp.pe", 0.005)
    pe_true[:, j] = pe

    t, pe = Potentials.get_pe("examples/Argon13/fit_lj/samples/sample_1/seed_$j/tmp.pe", 0.005)
    pe_lj[:, j] = pe    

   t, pe = Potentials.get_pe("examples/Argon13/fit_snap/samples/sample_1/seed_$j/tmp.pe", 0.005)
   pe_snap[:, j] = pe     
    
end

fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(1, 2, 1)

ax.hist(vec(pe_true), bins = 100, density = true, stacked = true, color = "red", alpha = 0.75, label = "True LJ")
ax.hist(vec(pe_lj), bins = 100, density = true, stacked = true, color = "blue", alpha = 0.5, label = "Fitted LJ")
ax.hist(vec(pe_snap), bins = 100, density = true, stacked = true, color = "black", alpha = 0.25, label = "Fitted SNAP")
ax.legend()
ax.set_ylabel("Probability")
ax.set_xlabel("Energy (LJ Units)")
ax.set_title("Distribution of Energy over seeds")
plt.grid("on")


# pe_snap = zeros(5000, 4*12)
# pe_lj = zeros(5000, 4*12)
# count = 1
# for j = 1:4
#     for i = 1:12
#         t, pe = Potentials.get_pe("examples/Argon13/fit_lj/samples/sample_$i/seed_$j/tmp.pe", 0.005)
#         pe_lj[:, count] = pe  

#         t, pe = Potentials.get_pe("examples/Argon13/fit_snap/samples/sample_$i/seed_$j/tmp.pe", 0.005)
#         pe_snap[:, count] = pe
#         count += 1
#     end
# end

# ax = fig.add_subplot(1, 3, 2)
# # ax.hist(vec(pe_true), bins = 100, density = true, stacked = true, color = "red", alpha = 0.25, label = "True LJ")
# ax.hist(vec(pe_lj), bins = 100, density = true, stacked = true, color = "blue", alpha = 0.5, label = "Fitted LJ")
# ax.hist(vec(pe_snap), bins = 100, density = true, stacked = true, color = "black", alpha = 0.25, label = "Fitted SNAP")
# ax.set_xlabel("Energy (LJ Units)")
# ax.set_title("Distribution of Energy over samples")
# plt.grid("on")

pe_snap = zeros(5000, 40*12)
pe_lj = zeros(5000, 40*12)
count = 1
for j = 1:50
    # println(j)
    for i = 1:12
        # println(i)
        try Potentials.get_rdf("examples/Argon13/fit_lj/samples/sample_$i/seed_$j/tmp.rdf")
            # println("SNAP")
            t, pe = Potentials.get_pe("examples/Argon13/fit_snap/samples/sample_$i/seed_$j/tmp.pe", 0.005)
            pe_snap[:, count] = pe

            # println("LJ")
            t, pe = Potentials.get_pe("examples/Argon13/fit_lj/samples/sample_$i/seed_$j/tmp.pe", 0.005)
            pe_lj[:, count] = pe
            count += 1
        catch
            continue
        end
    
        if count > 40*12
            break
        end
    end

end

ax = fig.add_subplot(1, 2, 2)
ax.hist(vec(pe_true), bins = 100, density = true, stacked = true, color = "red", alpha = 0.75, label = "True LJ")
ax.hist(vec(pe_lj), bins = 100, density = true, stacked = true, color = "blue", alpha = 0.5, label = "Fitted LJ")
ax.hist(vec(pe_snap), bins = 100, density = true, stacked = true, color = "black", alpha = 0.25, label = "Fitted SNAP")
ax.set_xlabel("Energy (LJ Units)")
ax.set_title("Variability of Energy over seeds and samples")
plt.grid("on")

plt.gcf()

plt.savefig("variability_of_pe.png", dpi = 600, bbox_inches = "tight")

########################################################################################
########################################################################################
lj = Potentials.LennardJones(1.0, 1.0)

c = Potentials.load_lammps("examples/Argon13/DATA_start", atom_names = [:Ar], radii = [3.1], weights = [1.0], boundary_type = ["f", "f", "f"])
Nt = 500000
dT = 1000
dt = 0.01
T = dT:dT:Nt

n = 50
pe_true = zeros(500, n)
pe_snap = zeros(500, n)
pe_lj = zeros(500, n)
for j = 1:n
    r = Potentials.get_positions(c, "examples/Argon13/DATA_training/true_ensemble/seed_$j/", T)
    pe = Potentials.potential_energy(r, lj)
    pe_true[:, j] = pe

    r = Potentials.get_positions(c, "examples/Argon13/fit_lj/samples/sample_1/seed_$j/", T)
    pe = Potentials.potential_energy(r, lj)
    pe_lj[:, j] = pe    

   r = Potentials.get_positions(c, "examples/Argon13/fit_snap/samples/sample_1/seed_$j/", T)
   pe = Potentials.potential_energy(r, lj)
   pe_snap[:, j] = pe     
    
end

fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(1, 2, 1)

ax.hist(vec(pe_true), bins = 100, density = true, stacked = true, color = "red", alpha = 0.75, label = "True LJ")
ax.hist(vec(pe_lj), bins = 100, density = true, stacked = true, color = "blue", alpha = 0.5, label = "Fitted LJ")
ax.hist(vec(pe_snap), bins = 100, density = true, stacked = true, color = "black", alpha = 0.25, label = "Fitted SNAP")
ax.legend()
ax.set_ylabel("Probability")
ax.set_xlabel("Energy (LJ Units)")
ax.set_title("Distribution of Forces over seeds")
plt.grid("on")


# pe_snap = zeros(5000, 4*12)
# pe_lj = zeros(5000, 4*12)
# count = 1
# for j = 1:4
#     for i = 1:12
#         t, pe = Potentials.get_pe("examples/Argon13/fit_lj/samples/sample_$i/seed_$j/tmp.pe", 0.005)
#         pe_lj[:, count] = pe  

#         t, pe = Potentials.get_pe("examples/Argon13/fit_snap/samples/sample_$i/seed_$j/tmp.pe", 0.005)
#         pe_snap[:, count] = pe
#         count += 1
#     end
# end

# ax = fig.add_subplot(1, 3, 2)
# # ax.hist(vec(pe_true), bins = 100, density = true, stacked = true, color = "red", alpha = 0.25, label = "True LJ")
# ax.hist(vec(pe_lj), bins = 100, density = true, stacked = true, color = "blue", alpha = 0.5, label = "Fitted LJ")
# ax.hist(vec(pe_snap), bins = 100, density = true, stacked = true, color = "black", alpha = 0.25, label = "Fitted SNAP")
# ax.set_xlabel("Energy (LJ Units)")
# ax.set_title("Distribution of Energy over samples")
# plt.grid("on")

pe_snap = zeros(500, 20*14)
pe_lj = zeros(500, 20*14)
count = 1
for j = 1:25
    println(j)
    for i = 1:14
        # println(i)
        try Potentials.get_rdf("examples/Argon13/fit_lj/samples/sample_$i/seed_$j/tmp.rdf")
            # println("SNAP")
            r = Potentials.get_positions(c, "examples/Argon13/fit_lj/samples/sample_$i/seed_$j/", T)
            pe = Potentials.potential_energy(r, lj)
            pe_lj[:, count] = pe 

            # println("LJ")
            r = Potentials.get_positions(c, "examples/Argon13/fit_snap/samples/sample_$i/seed_$j/", T)
            pe = Potentials.potential_energy(r, lj)
            pe_snap[:, count] = pe
            count += 1
        catch
            continue
        end
    
        if count > 20*14
            break
        end
    end

end

ax = fig.add_subplot(1, 2, 2)
# ax.hist(vec(pe_true), bins = 100, density = true, stacked = true, color = "red", alpha = 0.25, label = "True LJ")
ax.hist(vec(pe_true), bins = 100, density = true, stacked = true, color = "red", alpha = 0.75)
ax.hist(vec(pe_lj), bins = 100, density = true, stacked = true, color = "blue", alpha = 0.5, label = "Fitted LJ")
ax.hist(vec(pe_snap), bins = 100, density = true, stacked = true, color = "black", alpha = 0.25, label = "Fitted SNAP")
ax.set_xlabel("Energy (LJ Units)")
ax.set_title("Variability of Energy over seeds and samples")
plt.grid("on")

plt.gcf()

plt.savefig("variability_of_lj_pe.png", dpi = 600, bbox_inches = "tight")

##################################################################################################################################
##################################################################################################################################
