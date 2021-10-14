struct chain
    x :: Vector{Vector{<:Real}}
    diagnostics :: chain_diagnostics
end

function chain(x0::Vector{<:Real}, num_samples :: Int)
    x = Vector{Vector}(undef, num_samples); x[1] = x0
    diagnostics = chain_diagnostics(num_samples)
    return chain(x, diagnostics)
end

function mcmc_step(current_state::state, log_likelihood :: Function, proposal_distribution :: Function)
    x1 = proposal_distribution(current_state.x)
    new_likelihood = log_likelihood(x1)
    alpha = -new_likelihood + current_state.log_likelihood[1] 
    u = log(rand())
    if u <= alpha
        current_state.x[:] .= x1
        current_state.log_likelihood[1] = new_likelihood
        return 1
    else
        return 0
    end
end

function mcmc(x0 :: Vector{<:Real}, log_likelihood :: Function, proposal_distribution :: Function, num_burnin :: Int, num_samples :: Int)
    
    # Initialize 
    current_state = state(x0, log_likelihood(x0))

    # Burnin
    burnin_chain = chain(x0, num_burnin)
    burnin_chain.diagnostics.log_likelihood[1] = log_likelihood(x0)
    for iteration = 1:num_burnin-1
        flag = mcmc_step(current_state, log_likelihood, proposal_distribution)
        burnin_chain.x[iteration+1] = copy(current_state.x) 
        burnin_chain.diagnostics.log_likelihood[iteration+1] = current_state.log_likelihood[1]
        burnin_chain.diagnostics.acceptance_ratio[iteration+1] = ( burnin_chain.diagnostics.acceptance_ratio[iteration] * (iteration - 1) + flag ) / iteration
    end

    # Sampling
    sampling_chain = chain(current_state.x, num_samples)
    sampling_chain.diagnostics.log_likelihood[1] = log_likelihood(x0)

    for iteration = 1:num_samples-1
        flag = mcmc_step(current_state, log_likelihood, proposal_distribution)
        sampling_chain.x[iteration+1] = copy(current_state.x) 
        sampling_chain.diagnostics.log_likelihood[iteration+1] = current_state.log_likelihood[1]
        sampling_chain.diagnostics.acceptance_ratio[iteration+1] = ( sampling_chain.diagnostics.acceptance_ratio[iteration] * (iteration - 1) + flag ) / iteration
    end
    
    return burnin_chain, sampling_chain
end