########################################################################################################
## Details
# This file implements the MAP estimation proceedure for UQ inference problems.
#       Note, we convert the named tuple p.x into a vector for use in GalacticOptim.
#       For the conversion, we use TransformVariables.
#########################################################################################################

function MAP(p :: ArbitraryDistribution{T}) where T<: AbstractFloat

    # Setup optimization utilities
    trans, x0 = namedtp_to_vec(p.x)
    f = distribution_wrapper(p, trans)
    params = p.p 

    # Define problem
    g = OptimizationFunction(f, GalacticOptim.AutoForwardDiff())
    prob = OptimizationProblem(g, x0, p.p)

    # Solve
    sol = solve(prob, BFGS())

    # return solution
    xf = sol.u 
    p.x = transform(trans, xf)
    return p

end