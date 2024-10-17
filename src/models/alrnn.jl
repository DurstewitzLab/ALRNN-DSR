using Flux: @functor
using Statistics: mean

using ..ObservationModels: ObservationModel, init_state

# abstract type
abstract type AbstractALRNN end
(m::AbstractALRNN)(z::AbstractVecOrMat) = step(m, z)
jacobian(m::AbstractALRNN, z::AbstractVector) = Flux.jacobian(z -> m(z), z)[1]
jacobian(m::AbstractALRNN, z::AbstractMatrix) = jacobian.([m], eachcol(z))


"""
Almost linear RNN with specified number P of RELUs applied on latent states
"""
mutable struct ALRNN{V <: AbstractVector, M <: AbstractMatrix} <: AbstractALRNN
    A::V
    W::M
    h::V
    P::Int64
end
@functor ALRNN (A, W, h)

# initialization/constructor
function ALRNN(M::Int, P::Int)
    A, W, h = initialize_A_W_h(M)
    return ALRNN(A, W, h, P)
end

"""
    step(model, z)

Evolve `z` in time for one step according to the model `m` (equation).
`z` is either a `M` dimensional column vector or a `M x S` matrix, where `S` is
the batch dimension.
"""
step(m::ALRNN, z::AbstractVecOrMat) = m.A .* z .+ m.W * vcat(z[1:end-m.P,:], relu.(z)[end-(m.P-1):end,:]) .+ m.h
jacobian(m::ALRNN, z::AbstractVector) = Diagonal(m.A) + m.W * Diagonal(vcat(ones(size(m.W)[1]-m.P),z[end-(m.P-1):end] .> 0))




@inbounds """
    generate(model, z₁, T)

Generate a trajectory of length `T` using ALRNN model `m` given initial condition `z₁`.
"""
function generate(m::AbstractALRNN, z₁::AbstractVector, T::Int)
    # trajectory placeholder
    Z = similar(z₁, T, length(z₁))
    # initial condition for model
    @views Z[1, :] .= z₁
    # evolve initial condition in time
    @views for t = 2:T
        Z[t, :] .= m(Z[t-1, :])
    end
    return Z
end

"""
    generate(model, observation_model, x₁, T)

Generate a trajectory of length `T` using ALRNN model `m` given initial condition `x₁`.
"""
function generate(m::AbstractALRNN, obs::ObservationModel, x₁::AbstractVector, T::Int)
    z₁ = init_state(obs, x₁)
    ts = generate(m, z₁, T)
    return permutedims(obs(ts'), (2, 1))
end

