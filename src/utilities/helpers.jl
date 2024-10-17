using Flux
using Graphs
using SimpleWeightedGraphs
using LinearAlgebra
using Statistics
using StatsBase
using NaNStatistics
using BSON: load

num_params(m) = sum(length, Flux.params(m))

load_model(path::String;mod=@__MODULE__) = load(path, mod)[:model]

function evaluate_Dstsp(X, X_gen, bins_or_scaling)
    compute = bins_or_scaling > zero(bins_or_scaling)
    return compute ? state_space_distance(X, X_gen, bins_or_scaling) : missing
end

function evaluate_PSE(X, X_gen, smoothing)
    compute = smoothing > zero(smoothing)
    return compute ? power_spectrum_error(X, X_gen, smoothing) : missing
end

function evaluate_PE(m, O, X, n)
    compute = n > zero(n)
    return compute ? prediction_error(m, O, X, n) : missing
end

uniform(a, b) = rand(eltype(a)) * (b - a) + a
uniform(size, a, b) = rand(eltype(a), size) .* (b - a) .+ a

randn_like(X::AbstractArray{T, N}) where {T, N} = randn(T, size(X)...)
# @code_warntype still returns any for this..
add_gaussian_noise!(X::AbstractArray{T, N}, noise_level::T) where {T, N} =
    X .+= noise_level .* randn_like(X)
add_gaussian_noise(X::AbstractArray{T, N}, noise_level::T) where {T, N} =
    X .+ noise_level .* randn_like(X)
