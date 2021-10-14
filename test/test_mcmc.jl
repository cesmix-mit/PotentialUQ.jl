## Test MCMC routines
using PotentialUQ
using LinearAlgebra
using Statistics
x0 = randn(2)
num_samples = 5

target_mean = 2.0*ones(2)
target_cov = [1.0 0.2; 0.2 1.0]
target_cov_inverse = inv(target_cov)
log_likelihood(x) = 0.5*(x - target_mean)' * target_cov_inverse * (x - target_mean)
proposal_distribution(x) = x + randn(length(x))

current_state = state(x0, log_likelihood(x0))
chain_diagnostics(num_samples)
chain(x0, num_samples)
@time mcmc_step(current_state, log_likelihood, proposal_distribution)
@time burnin_chain, final_chain = mcmc(x0, log_likelihood, proposal_distribution, 1000, 500)

# println("Burnin Chain")
# show(stdout, "text/plain", burnin_chain.x)
# show(stdout, "text/plain", burnin_chain.diagnostics.log_likelihood)
# println(" ")

println("Sampling Chain:")
show(stdout, "text/plain", final_chain.x)
println(" ")
show(stdout, "text/plain", final_chain.diagnostics.log_likelihood)
println(" ")

println("Mean of chain: ")
show(stdout, "text/plain", mean(burnin_chain.x)); println(" "); 
show(stdout, "text/plain", mean(final_chain.x)); println(" ")

println("Covariance of chain: ")
show(stdout, "text/plain", cov(final_chain.x))
println(" ")
println("Acceptance rate: $(burnin_chain.diagnostics.acceptance_ratio[end])  $(final_chain.diagnostics.acceptance_ratio[end])")
