using Flux

using ..Models
using ..Utilities
using ..ObservationModels

abstract type AbstractTFRecur end
Flux.trainable(tfrec::AbstractTFRecur) = (tfrec.model, tfrec.O)

(tfrec::AbstractTFRecur)(X::AbstractArray{T, 3}) where {T} = forward(tfrec, X)

"""
    forward(tfrec, X)

Forward pass using teacher forcing. If the latent dimension of
the RNN is larger than the dimension the observations live in, 
partial teacher forcing of the first `N = size(X, 1)` neurons is
used. Initializing latent state `z₁` is taken care of by the observation model.
"""
function forward(tfrec::AbstractTFRecur, X::AbstractArray{T, 3}) where {T}
    N, _, T̃ = size(X)
    M = size(tfrec.z, 1)

    # number of forced states
    D = min(N, M)

    # precompute forcing signals
    Z⃰ = apply_inverse(tfrec.O, X)

    # initialize latent state
    tfrec.z = @views init_state(tfrec.O, X[:, :, 1])

    # process sequence X
    Z = @views [tfrec(Z⃰[1:D, :, t], t) for t = 2:T̃]

    # reshape to 3d array and return
    return reshape(reduce(hcat, Z), size(tfrec.z)..., :)
end

"""
Inspired by `Flux.Recur` struct, which by default has no way
of incorporating teacher forcing.

This is just a convenience wrapper around stateful models,
to be used during training.
"""
mutable struct TFRecur{M <: AbstractMatrix, 𝒪 <: ObservationModel} <: AbstractTFRecur
    # stateful model, e.g. ALRNN
    model::Any
    # observation model
    O::𝒪
    # state of the model
    z::M
    # forcing interval
    const τ::Int
end
Flux.@functor TFRecur

function (tfrec::TFRecur)(x::AbstractMatrix, t::Int)
    # determine if it is time to force the model
    z = tfrec.z

    # perform one step using the model, update model state
    z = tfrec.model(z)

    # force
    z̃ = (t - 1) % tfrec.τ == 0 ? force(z, x) : z
    tfrec.z = z̃
    return z
end

