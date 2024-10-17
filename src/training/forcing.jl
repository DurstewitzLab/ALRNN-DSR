using ChainRulesCore: NoTangent, @thunk
import ChainRulesCore

@inbounds """
    force(z, x)

Replace the first `N = dim(x)` dimensions of `z` with `x`.

Supplied with custom backward `ChainRulesCore.rrule`.
"""
function force(z::AbstractMatrix, x::AbstractMatrix)
    N = size(x, 1)
    return [x; z[N+1:end, :]]
end

@inbounds function ChainRulesCore.rrule(
    ::typeof(force),
    z::AbstractMatrix,
    x::AbstractMatrix,
)
    N = size(x, 1)
    function force_pullback(ΔΩ)
        ∂x = ΔΩ[1:N, :]
        # in-place here for speed
        ΔΩ[1:N, :] .= 0
        return (NoTangent(), ΔΩ, ∂x)
    end
    return force(z, x), force_pullback
end
