using Statistics
using PyCall
pygui(:tk)
import PyPlot; const plt = PyPlot
import InteratomicPotentials as Potentials

n=50
#############################################################################################################################################
#############################################################################################################################################
using JLD
lj_samples = load("examples/Argon13/fit_lj/samples_lj_TEMP_65.jld")["samples"]
lj_samples = hcat(lj_samples...)
snap_samples = load("examples/Argon13/fit_snap/samples_snap_TEMP_65_date_9_30_2021.jld")["samples"]
snap_samples = hcat(snap_samples...)

## Histogram 2d
fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(1,1,1)
counts, xedges, yedges, im = ax.hist2d(lj_samples[1, :], lj_samples[2, :], bins = 100, range = [[0.9, 1.1],[0.9, 1.1]], density = true, cmap="Blues")
ax.set_xlabel("σ")
ax.set_yllabel("ϵ")
ax.set_title("Distribution of Fitted LJ Parameters")
plt.gcf()

plt.savefig("distribution_of_lj_parameters_hexbin.png", dpi = 600, bbox_inches = "tight")

## Scatter with Contours
fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(1,1,1)
ax.hexbin(lj_samples[1, :], lj_samples[2, :], gridsize = 20, cmap="Blues")
ax.set_xlabel("σ")
ax.set_ylabel("ϵ")
ax.set_title("Distribution of LJ Parameters")
plt.gcf()


fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(1,1,1)
ax.boxplot(snap_samples[1:30, :]', notch = true)

ax.set_xlabel("Parameter")
ax.set_ylabel("SNAP Parameter Values")
ax.set_title("Distribution of Fitted SNAP Parameters")
plt.gcf()
plt.savefig("boxplot_snap_parameters.png", dpi = 600, bbox_inches = "tight")

fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(1,1,1)
ax.plot(1:31, std(snap_samples, dims = 2), linestyle = " ", marker = "*")
ax.set_ylabel("Standard Deviation of SNAP Parameters")
ax.set_xlabel("SNAP Parameters")
ax.set_yscale("log")

plt.gcf()
plt.savefig("standard_deviation_snap_parameters.png", dpi = 600, bbox_inches = "tight")
