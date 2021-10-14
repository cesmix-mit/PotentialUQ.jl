struct samples
    samples :: Vector{Vector}
end

function num_samples(s::samples)
    length(s.samples)
end