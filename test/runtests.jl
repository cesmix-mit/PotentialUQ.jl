using Test

@time begin
@testset "PotentialUQ.jl" begin
    include("snap_test_MAP.jl")
    include("snap_test_sample.jl")
end
end
