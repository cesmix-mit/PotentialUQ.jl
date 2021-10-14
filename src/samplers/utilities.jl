struct state
    x :: Vector{<:Real}
    log_likelihood :: Vector{<:Real}
    grad_log_likelihood :: Vector{<:Real}
end

function state(x::Vector{<:Real}, log_likelihood :: Real)
    return state(x, [log_likelihood], zeros(length(x)))
end

function state(x::Vector{<:Real}, log_likelihood :: Real, grad_log_likelihood :: Vector{<:Real})
    return state(x, [log_likelihood], grad_log_likelihood)
end

struct chain_diagnostics
    log_likelihood  :: Vector{<:Real}
    acceptance_ratio :: Vector{<:Real}
end 

function chain_diagnostics(num_samples :: Int)
    return chain_diagnostics(zeros(num_samples), zeros(num_samples))
end



