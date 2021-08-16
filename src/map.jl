########################################################################################################
## Details
# This file implements the MAP estimation proceedure for UQ inference problems.
#       Note, we convert the named tuple p.x into a vector for use in GalacticOptim.
#       For the conversion, we use TransformVariables.
#########################################################################################################

function MAP(pdist :: ArbitraryDistribution; solver = LBFGS(), verbose = false) 

    # Define a Hamiltonian system
    x0 = inverse(pdist.t, pdist.x)
    n = length(x0)
    @model function demo()
        x ~ pdist.prior
        Turing.@addlogprob! pdist.distribution(x)
    end    

    model = demo()
    map_estimate = optimize(model, Turing.MAP(), x0, solver)
    pdist.x = transform(pdist.t, map_estimate.values.array)
    if verbose
        println("Map Estimate ")
        show(stdout, "text/plain", map_estimate)
    end

    return pdist

end