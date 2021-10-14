using Statistics
using PyCall
pygui(:tk)
import PyPlot; const plt = PyPlot
import InteratomicPotentials as Potentials

n=50
########################################################################################
########################################################################################
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 2, 1)
rdf_true = zeros(100, n)
rdf_snap = zeros(100, n)
rdf_lj = zeros(100, n)
for j = 1:n
    # println(j)
    # println("True")
    bins, rdf = Potentials.get_rdf("examples/Argon13/DATA_training/true_ensemble/seed_$j/tmp.rdf")
    rdf_true[:, j] = rdf
    if j == 1
        ax.plot(bins, rdf, color="red", linewidth = 1.0, linestyle="--", label= "True LJ")
    else
        ax.plot(bins, rdf, color="red", linewidth = 0.1, linestyle="--")
    end

    # println("LJ")
    bins, rdf = Potentials.get_rdf("examples/Argon13/fit_lj/samples/sample_1/seed_$j/tmp.rdf")
    rdf_lj[:, j] = rdf
    if j == 1
        ax.plot(bins, rdf, color="blue", linewidth = 1.0, linestyle="--", label= "Fitted LJ")
    else
        ax.plot(bins, rdf, color="blue", linewidth = 0.1, linestyle="--")
    end
    

    # println("SNAP")
    bins, rdf = Potentials.get_rdf("examples/Argon13/fit_snap/samples/sample_1/seed_$j/tmp.rdf")
    if j == 1
        ax.plot(bins, rdf, color="black", linewidth = 1.0, linestyle="--", label= "Fitted SNAP")
    else
        ax.plot(bins, rdf, color="black", linewidth = 0.1, linestyle="--")
    end
    rdf_snap[:, j] = rdf
    
end
max_var_true = round(maximum(std(rdf_true, dims = 2)), digits = 2)
max_var_lj = round(maximum(std(rdf_lj, dims = 2)), digits = 2)
max_var_snap = round(maximum(std(rdf_snap, dims = 2)), digits = 2)
ax.text(1.5, 20.0, "Maximum STD: \n True LJ:         $max_var_true \n Fitted LJ:       $max_var_lj \n Fitted SNAP: $max_var_snap")
ax.set_title("Variability of RDF over random seeds")
ax.set_ylabel("RDF")
ax.set_xlabel("r (LJ Units)")
ax.legend()
ax.set_xlim([0.75, 2.5])
plt.grid("on")



# ax = fig.add_subplot(1, 3, 2)
# for j = 1:n
#     bins, rdf = Potentials.get_rdf("examples/Argon13/DATA_training/true_ensemble/seed_$j/tmp.rdf")
#     ax.plot(bins, rdf, color="red", linewidth = 0.1, linestyle="--")
# end

# rdf_snap = zeros(100, 48)
# rdf_lj = zeros(100, 48)
# count = 1
# for j = 1:4
#     for i = 1:12
#     bins, rdf = Potentials.get_rdf("examples/Argon13/fit_snap/samples/sample_$i/seed_$j/tmp.rdf")
#     rdf_snap[:, count] = rdf
#     ax.plot(bins, rdf, color = "black", linewidth = 0.1, linestyle = "--")

#     bins, rdf = Potentials.get_rdf("examples/Argon13/fit_lj/samples/sample_$i/seed_$j/tmp.rdf")
#     rdf_lj[:, count] = rdf
#     ax.plot(bins, rdf, color="blue", linewidth = 0.1, linestyle="--")
#     plt.title("Variability of RDF over random samples")
#     count += 1
#     end
# end
# max_var_lj = round(maximum(std(rdf_lj, dims = 2)), digits = 2)
# max_var_snap = round(maximum(std(rdf_snap, dims = 2)), digits = 2)
# ax.text(1.5, 20.0, "Maximum STD: \n \n Fitted LJ:       $max_var_lj \n Fitted SNAP: $max_var_snap")
# ax.legend()
# ax.set_xlabel("r (LJ Units)")
# ax.set_xlim([0.75, 2.5])
# plt.grid("on")


ax = fig.add_subplot(1, 2,2)
for j = 1:n
    bins, rdf = Potentials.get_rdf("examples/Argon13/DATA_training/true_ensemble/seed_$j/tmp.rdf")
    ax.plot(bins, rdf, color="red", linewidth = 0.1, linestyle="--")
end

rdf_snap = zeros(100, 40*12)
rdf_lj = zeros(100, 40*12)
count = 1
for j = 1:50
    # println(j)
    for i = 1:12
        # println(i)
        try Potentials.get_rdf("examples/Argon13/fit_lj/samples/sample_$i/seed_$j/tmp.rdf")
            # println("SNAP")
            bins, rdf = Potentials.get_rdf("examples/Argon13/fit_snap/samples/sample_$i/seed_$j/tmp.rdf")
            rdf_snap[:, count] = rdf
            ax.plot(bins, rdf, color = "black", linewidth = 0.1, linestyle = "--")

            # println("LJ")
            bins, rdf = Potentials.get_rdf("examples/Argon13/fit_lj/samples/sample_$i/seed_$j/tmp.rdf")
            rdf_lj[:, count] = rdf
            ax.plot(bins, rdf, color="blue", linewidth = 0.1, linestyle="--")
            plt.title("Variability of RDF over seeds and samples")
            count += 1
        catch
            continue
        end
    
        if count > 40*12
            break
        end
    end

end
max_var_lj = round(maximum(std(rdf_lj, dims = 2)), digits = 2)
max_var_snap = round(maximum(std(rdf_snap, dims = 2)), digits = 2)
ax.text(1.5, 20.0, "Maximum STD: \n \n Fitted LJ:       $max_var_lj \n Fitted SNAP: $max_var_snap")
ax.legend()
ax.set_xlabel("r (LJ Units)")
ax.set_xlim([0.75, 2.5])
plt.grid("on")

plt.gcf()
plt.savefig("variation_of_rdf_two.png", dpi=600, bbox_inches = "tight")


