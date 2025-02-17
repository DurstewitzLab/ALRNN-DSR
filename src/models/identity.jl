using Flux, LinearAlgebra, FillArrays
using ..Utilities

abstract type ObservationModel end
(O::ObservationModel)(z::AbstractArray) = forward(O, z)


"""
    Identity(z)

Applies the identity to the first `N` states and discards the remaining `M-N` states,
i.e. treats the first `N` neurons of the RNN as read-out neurons (asserts `M > N`). 
Uses an addition trained matrix `L` for a simple estimate of the `M-N` latent states.

Accepted inputs for `z` are
- a `M`-dimensional `Vector` (a single latent state vector)
- a `M × S`-dimensional `Matrix` (a batch of latent state vectors)
- a `M × S × T̃`-dimensional `Array` (a sequence of batched state vectors) 
"""
mutable struct Identity{M <: AbstractMatrix, V <: AbstractVector, MY <: Maybe{AbstractMatrix}} <: ObservationModel
    const B::M
    const b::V
    use_bias::Bool
    L::MY
end
Flux.trainable(O::Identity) = (O.L,)
Flux.@functor Identity

Identity(N::Int, M::Int) =
    Identity([I(N) zeros(Float32, N, M - N)], zeros(Float32, N), false, initialize_L(M, N))

init_state(O::Identity, x::AbstractVecOrMat) =
    if O.L === nothing
        return x
    else
        return [x; O.L * x]
    end

Utilities.num_params(O::Identity) = isnothing(O.L) ? 0 : length(O.L)

@inbounds forward(O::Identity, z::AbstractVector; return_view::Bool = false) =
    return_view ? @view(z[1:size(O.B, 1)]) : z[1:size(O.B, 1)]
@inbounds forward(O::Identity, z::AbstractMatrix; return_view::Bool = false) =
    return_view ? @view(z[1:size(O.B, 1), :]) : z[1:size(O.B, 1), :]
@inbounds forward(O::Identity, z::AbstractArray{T, 3}; return_view::Bool = false,) where {T} = 
    return_view ? @view(z[1:size(O.B, 1), :, :]) : z[1:size(O.B, 1), :, :]

apply_inverse(O::Identity, x::AbstractVector) =
    [x; Zeros{eltype(x)}(size(O.B, 2) - size(O.B, 1))]
apply_inverse(O::Identity, x::AbstractMatrix) =
    [x; Zeros{eltype(x)}(size(O.B, 2) - size(O.B, 1), size(x, 2))]
apply_inverse(O::Identity, x::AbstractArray{T, 3}) where {T} =
    [x; Zeros{eltype(x)}(size(O.B, 2) - size(O.B, 1), size(x, 2), size(x, 3))]

inverse(O::Identity) = O.B'

initialize_L(M::Int, N::Int) =
    if M == N
        L = nothing
    else
        L = Flux.glorot_uniform(M - N, N)
    end