########################################################################################################
## Details
# This file implements the MAP estimation proceedure for UQ inference problems.
#       Note, we convert the named tuple p.x into a vector for use in GalacticOptim.
#       For the conversion, we use TransformVariables.
#########################################################################################################

function MAP(pdist :: ArbitraryDistribution) 

    # Setup optimization utilities
    x0 = inverse(pdist.trans, pdist.x)
    f(x, params = nothing) = -pdist.distribution(x)
    # Define problem
    g = OptimizationFunction(f, GalacticOptim.AutoForwardDiff())
    prob = OptimizationProblem(g, x0)

    # Solve
    sol = solve(prob, BFGS())

    # return solution
    xf = sol.u 
    pdist.x = transform(pdist.trans, xf)
    Potentials.set_trainable_params!(pdist.snap, pdist.x)
    return pdist

end