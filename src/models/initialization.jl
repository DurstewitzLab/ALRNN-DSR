function normalized_positive_definite(M::Int)
    R = randn(Float32, M, M)
    K = R'R ./ M + I
    λ = maximum(abs.(eigvals(K)))
    return K ./ λ
end

function uniform_init(shape::Tuple; eltype::Type{T} = Float32) where {T <: AbstractFloat}
    @assert length(shape) < 3
    din = Float32(shape[end])
    r = 1 / √din
    return uniform(shape, -r, r)
end

function gaussian_init(M::Int, N::Int)
    return Float32.(randn(Float32, M, N) .* 0.01)
end

function initialize_A_W_h(M::Int)
    A = diag(normalized_positive_definite(M))
    W = gaussian_init(M,M)
    h = zeros(Float32, M)
    return A, W, h
end

function initialize_L(M::Int, N::Int)
    if M == N
        L = nothing
    else
        L = uniform_init((M - N, N))
    end
end