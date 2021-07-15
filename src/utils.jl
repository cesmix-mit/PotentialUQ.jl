function namedtp_to_vec( x :: NamedTuple ) 
    l = length(x)
    t = [ as( ( typeof(xx) <: Array) ? Array : typeof(xx), size(xx) ) for xx in x]
    trans = as(NamedTuple{keys(x)}(t))
    return trans, inverse(trans, x)
end

function distribution_wrapper(pdist :: ArbitraryDistribution, trans_x :: TransformVariables.TransformTuple)
    
    f(x, params) = pdist.distribution( transform(trans_x, x), params )

    return f
end

function distribution_wrapper(pdist :: ArbitraryDistribution, trans_x :: TransformVariables.TransformTuple, trans_params :: TransformVariables.TransformTuple)
    
    f(x, params) = pdist.distribution( transform(trans_x, x), transform(trans_params, params) )

    return f
end