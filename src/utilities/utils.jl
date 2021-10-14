function namedtp_to_vec( x :: NamedTuple ) 
    l = length(x)
    t = [ ( typeof(xx) <: AbstractFloat) ? asℝ : 
           as( (typeof(xx) <: AbstractArray) ? Array : typeof(xx), size(xx) ) for xx in x]
    trans = as(NamedTuple{keys(x)}(t))
    return trans
end
