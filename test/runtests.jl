using Test

@time begin
@testset "PotentialUQ.jl" begin
    include("test_mcmc.jl")
    include("test_hmc.jl")

    # include("snap_test_MAP.jl")
    # include("snap_test_sample.jl")
    # include("snap_GaN_sample.jl")
end
end
