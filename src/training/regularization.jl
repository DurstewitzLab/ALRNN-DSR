using LinearAlgebra
using Zygote

using ..Models
using ..ObservationModels


"""
    AR_convergence_loss(m::AbstractALRNN, λ::Float32, p::Real = 2)

Pushes entries of `A + diag(W)` to stay below 1, avoiding divergent dynamics.
"""
function AR_convergence_loss(m::AbstractALRNN, λ::Float32, p::Real = 2, ϵ::Float32 = 0.5f-1)
    @assert ϵ > 0.0f0
    loss = norm(relu.((m.A .+ diag(m.W)) .- (1 - ϵ)), p)
    return λ * loss
end

"""
    regularize(model, λ; penalty)

Weight regularization. Defaults to L2 penalization.
"""
regularize(O::ObservationModel, λ::Float32; penalty = l2_penalty) =
    λ * sum(penalty, Flux.params(O))

regularize(O::Identity, λ::Float32; penalty = l2_penalty) =
    λ * penalty(O.L)

function regularize(m::AbstractALRNN, λ::Float32; penalty = l1_penalty)
    A_reg = penalty(1 .- m.A)
    W_reg = penalty(m.W)
    h_reg = penalty(m.h)
    return λ * (A_reg + W_reg + h_reg)
end

regularize(m, args...; kwargs...) =
    throw("Regularization for model type $(typeof(m)) not implemented yet!")

l2_penalty(θ) = isnothing(θ) ? 0 : sum(abs2, θ)
l1_penalty(θ) = isnothing(θ) ? 0 : sum(abs, θ)
