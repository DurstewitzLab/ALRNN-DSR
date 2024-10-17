using Flux
using ..Models
using ..ObservationModels

@inbounds function prediction_error(model, O::ObservationModel, X::AbstractMatrix, n::Int)
    T = size(X, 1)
    T̃ = T - n

    # batched computations from here on
    # time dim -> batch dim
    z = @views init_state(O, X[1:T̃, :]')
    for t = 1:n
        z = model(z)
    end

    # compute MSE
    mse = @views Flux.mse(O(z)', X[n+1:end, :])
    return mse
end
