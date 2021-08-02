########################################################################################################
## Details
# This file implements the sampled-based estimation proceedure for UQ inference problems.
#       We make heavy use of AdvancedHMC.jl and Distributions.jl, many of the functions below are wrappers
#########################################################################################################
function Sample(pdist :: ArbitraryDistribution; num_samples = 100, num_adapts = 300, verbose = false) 
    
    # Define a Hamiltonian system
    x0 = inverse(pdist.t, pdist.x)
    n = length(x0)
    @model function demo()
        x ~ pdist.prior
        Turing.@addlogprob! pdist.distribution(x)
    end    


# # Start sampling.
    chain = Turing.sample(demo(), NUTS(num_adapts, 0.7), num_samples)
    samples = copy(group(chain, "x").value.data[:,:,1])
    samples = [samples[i, :] for i = 1:size(samples, 1)]
    if verbose 
        show(stdout, "text/plain", chain) 
        println(" ")
    end
    

    pdist.samples = transform.(pdist.t, samples)
    pdist.x = transform(pdist.t, mean(samples))
    return samples

end
