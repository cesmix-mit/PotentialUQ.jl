########################################################################################################
## Details
# This file implements the sampled-based estimation proceedure for UQ inference problems.
#       We make heavy use of AdvancedHMC.jl and Distributions.jl, many of the functions below are wrappers
#########################################################################################################
function Sample(pdist :: ArbitraryDistribution; num_chains = 1, estimate_map = false, return_chain = false, num_samples = 100, num_adapts = 300, verbose = false) 
    

    if estimate_map == true
        MAP(pdist; verbose = verbose)
    end

    # Define a Hamiltonian system
    x0 = inverse(pdist.t, pdist.x)
    n = length(x0)
    @model function demo()
        x ~ pdist.prior
        Turing.@addlogprob! pdist.distribution(x)
    end    

    model = demo()

    # # Start sampling.
    if num_chains == 1
        chain = Turing.sample(model, NUTS(num_adapts, 0.5; max_depth = 15), num_samples, init_theta = x0)
        samples = copy(group(chain, "x").value.data[:,:,1])
    else
        chain = Turing.sample(model, NUTS(num_adapts, 0.5, 15), MCMCThreads(),  num_samples, num_chains, init_theta = x0.*(1+0.1*randn()) )
        samples = copy( reshape(group(chain, "x").value.data, num_samples*num_chains, n) )
    end
    samples = [samples[i, :] for i = 1:size(samples, 1)]
    if verbose 
        show(stdout, "text/plain", chain) 
        println(" ")
    end
    
    # Extract Samples
    pdist.samples = transform.(pdist.t, samples) 
    values = pdist.distribution.(samples)
    map = samples[argmax( values )]
    pdist.x = transform(pdist.t, map)
    if return_chain
        return chain, samples
    else
        return samples
    end

end
