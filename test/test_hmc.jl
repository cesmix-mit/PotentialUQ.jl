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
grad_log_likelihood(x) = target_cov_inverse * (x - target_mean)

current_state = state(x0, log_likelihood(x0), grad_log_likelihood(x0))
chain_diagnostics(num_samples)
chain(x0, num_samples)
@time hmc_step(current_state, log_likelihood, grad_log_likelihood, randn(size(x0)), 1e-2, 20)
@time burnin_chain, final_chain = hmc(x0, log_likelihood, grad_log_likelihood, 1000, 500)

# println("Burnin Chain")
# show(stdout, "text/plain", burnin_chain.x)
# show(stdout, "text/plain", burnin_chain.diagnostics.log_likelihood)
# println(" ")

println("Sampling Chain:")
show(stdout, "text/plain", final_chain.x[end-10:end])
println(" ")
show(stdout, "text/plain", final_chain.diagnostics.log_likelihood[end-10:end])
println(" ")

println("Mean of chain: ")
show(stdout, "text/plain", mean(burnin_chain.x)); println(" "); 
show(stdout, "text/plain", mean(final_chain.x)); println(" ")

println("Covariance of chain: ")
show(stdout, "text/plain", cov(final_chain.x))
println(" ")
println("Acceptance rate: $(burnin_chain.diagnostics.acceptance_ratio[end])  $(final_chain.diagnostics.acceptance_ratio[end])")
