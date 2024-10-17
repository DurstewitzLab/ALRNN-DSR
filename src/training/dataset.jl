using Flux: batch
using NPZ

abstract type AbstractDataset end
"""
    Dataset(args; kwargs)

Standard dataset storing a continuous time series of
size T × N, where N is the data dimension.
"""
struct Dataset{M <: AbstractMatrix} <: AbstractDataset
    X::M
    name::String
end

function Dataset(path::String, name::String; device = cpu, dtype = Float32)
    X = npzread(path) .|> dtype |> device
    @assert ndims(X) == 2 "Data must be 2-dimensional but is $(ndims(X))-dimensional."
    return Dataset(X, name)
end

Dataset(path::String; device = cpu, dtype = Float32) =
    Dataset(path, ""; device = device, dtype = dtype)



@inbounds """
    sample_sequence(dataset, sequence_length)

Sample a sequence of length `T̃` from a time series X.
"""
function sample_sequence(D::Dataset, T̃::Int)
    T = size(D.X, 1)
    i = rand(1:T-T̃-1)
    return D.X[i:i+T̃, :]
end

@inbounds """
    sample_batch(dataset, seq_len, batch_size)

Sample a batch of sequences of batch size `S` from time series X
(with replacement!).
"""
function sample_batch(D::Dataset, T̃::Int, S::Int)
    N = size(D.X, 2)
    Xs = similar(D.X, N, S, T̃ + 1)
    Threads.@threads for i = 1:S
        Xs[:, i, :] .= sample_sequence(D, T̃)'
    end
    return Xs
end
