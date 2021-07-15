########################################################################################################
## Details
# This file implements the sampled-based estimation proceedure for UQ inference problems.
#       We make heavy use of Turing.jl and Distributions.jl, many of the functions below are wrappers
#########################################################################################################
function sample(pdist :: SNAPDistribution{T}; num_samples = 600, return_type = "NamedTuple", verbose = true) where T<: AbstractFloat
    # Setup optimization utilities
    trans_x, x0 = namedtp_to_vec(pdist.x)
    n = length(x0)
    trans_p, params = namedtp_to_vec(pdist.p)
    f = distribution_wrapper(pdist, trans_x, trans_p)
    
    Turing.@model function regression(params)
        x ~ MvNormal(zeros(n), I(n))
        Turing.@addlogprob! -1.0*f(x, params)
    end
    
    model = regression(params)
    chain = Turing.sample(model, Turing.NUTS(0.65), num_samples)

    chain_mat = zeros(num_samples, n)
    for i = 1:n
        chain_mat[:, i] = chain[:, i, 1]
    end

    mean_chain = mean(chain_mat, dims = 1)
    pdist.x = transform(trans_x, vec(mean_chain))

    if return_type == "NamedTuple"
        posterior_samples = [ transform(trans_x, chain_mat[i, :]) for i = 1:num_samples]
        if verbose println("Saving   posterior samples as NamedTuple") end
        return posterior_samples
    elseif return_type == "Matrix"
        if verbose println("Returning samples as matrix, along with transform to NamedTuple") end
        return chain_mat, trans_x
    else
        if verbose println("Returning chains along with transform to NamedTuple") end
        return chain, trans_x
    end
end
