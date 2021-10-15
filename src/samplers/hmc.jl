function hmc_step(current_state::state, log_likelihood::Function, grad_log_likelihood::Function, current_p::Vector{<:Real}, ϵ::Float64, L::Int)

    current_log_likelihood = current_state.log_likelihood[1] + 0.5 * sum(current_p.^2)

    # Evolve p half-step 
    current_p += 0.5 * ϵ * current_state.grad_log_likelihood

    # Leap-Frog Integrate
    x1 = current_state.x
    for i = 1:L-1
        x1 += ϵ * current_p 
        current_p += ϵ * grad_log_likelihood(x1)
    end
    x1 += ϵ * current_p

    current_p += 0.5 * ϵ * grad_log_likelihood(x1)
    current_p = -1 * current_p

    # Perform comparison
    new_log_likelihood = log_likelihood(x1)
    proposed_log_likelihood = new_log_likelihood + 0.5 * sum(current_p.^2)
    alpha = current_log_likelihood - proposed_log_likelihood
    u = log(rand())
    if u <= alpha 
        current_state.x[:] .= x1 
        current_state.log_likelihood[1] = new_log_likelihood
        return 1
    else
        return 0
    end

end

function hmc(x0 :: Vector{<:Real}, log_likelihood :: Function, grad_likelihood_function :: Function, num_burnin :: Int, num_samples :: Int)
    
    # Initialize
    ϵ = 1e-2
    L = 20

    # Initialize 
    current_state = state(x0, log_likelihood(x0), grad_likelihood_function(x0))

    # Burnin
    burnin_chain = chain(x0, num_burnin)
    burnin_chain.diagnostics.log_likelihood[1] = log_likelihood(x0)
    for iteration = 1:num_burnin-1
        p = randn(size(x0))
        flag = hmc_step(current_state, log_likelihood, grad_likelihood_function, p, ϵ, L)
        burnin_chain.x[iteration+1] = copy(current_state.x) 
        burnin_chain.diagnostics.log_likelihood[iteration+1] = current_state.log_likelihood[1]
        burnin_chain.diagnostics.acceptance_ratio[iteration+1] = ( burnin_chain.diagnostics.acceptance_ratio[iteration] * (iteration - 1) + flag ) / iteration
    end

    if burnin_chain.diagnostics.acceptance_ratio[end] < 0.25
        ϵ *= 0.5
    elseif burnin_chain.diagnostics.acceptance_ratio[end] > 0.9
        ϵ *= 2
    end

    # Sampling
    sampling_chain = chain(current_state.x, num_samples)
    sampling_chain.diagnostics.log_likelihood[1] = log_likelihood(x0)

    for iteration = 1:num_samples-1
        p = randn(size(x0))
        flag = hmc_step(current_state, log_likelihood, grad_likelihood_function, p, ϵ, L)
        sampling_chain.x[iteration+1] = copy(current_state.x) 
        sampling_chain.diagnostics.log_likelihood[iteration+1] = current_state.log_likelihood[1]
        sampling_chain.diagnostics.acceptance_ratio[iteration+1] = ( sampling_chain.diagnostics.acceptance_ratio[iteration] * (iteration - 1) + flag ) / iteration
    end
    
    return burnin_chain, sampling_chain
end