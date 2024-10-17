using FFTW: rfft
using ImageFiltering: Kernel, imfilter!
using StatsBase: standardize

"""
    power_spectrum_error(X, X̃)

Compute the power spectrum error between ground truth `X` and
simulated trajectory `X̃`.
"""
function power_spectrum_error(X::AbstractMatrix, X̃::AbstractMatrix, σ::Real = 1)
    @assert all(size(X) .== size(X̃))

    X_ps = normalized_and_smoothed_power_spectrum(X, σ)
    X̃_ps = normalized_and_smoothed_power_spectrum(X̃, σ)
    hd = hellinger_distance(X_ps, X̃_ps)

    return mean(hd), hd
end

"""
    power_spectrum_correlation(X, X̃)

[LEGACY] Compute the power spectrum correlation between ground truth `X` and
simulated trajectory `X̃`. 
"""
function power_spectrum_correlation(X::AbstractMatrix, X̃::AbstractMatrix, σ::Real = 1.0)
    @assert all(size(X) .== size(X̃))
    X_ps = normalized_and_smoothed_power_spectrum(X, σ)
    X̃_ps = normalized_and_smoothed_power_spectrum(X̃, σ)

    r = cor.(eachcol(X_ps), eachcol(X̃_ps))
    return mean(r), r
end

function normalized_and_smoothed_power_spectrum(
    X::AbstractMatrix,
    σ::Real;
    normalize::Bool = true,
)
    # standardize
    X_ = standardize(ZScoreTransform, X, dims = 1)

    # compute PS
    PS = abs2.(rfft(X_, 1))

    if σ > zero(σ)
        # smooth PS
        smooth_dims!(PS, σ)

        # remove negative outliers (artifact?)
        PS[PS.<zero(eltype(PS))] .= zero(eltype(PS))
    end

    # normalize PS
    normalize ? PS ./= sum(PS, dims = 1) : nothing

    return PS
end

"""
    smooth_dims!(X::AbstractMatrix, σ::Real)

Smooth every observation (row) of matrix `X` using a gaussian kernel 
with standard deviation `σ`. Filtering parameters adjusted such that
method aligns with default behaviour of python's `scipy.gaussian_filter1d`,
specifically padding mode and kernel length.
"""
function smooth_dims!(X::AbstractMatrix, σ::Real)
    # set kernel radius to be equal to python implementation
    # which is 4 * standard deviation instead of default of 2
    # used by ImageFiltering.Kernel.gaussian()
    l = 2 * 4 * σ
    # kernel length has to be odd
    l = l % 2 == 0 ? l + 1 : l
    κ = Kernel.gaussian((σ,), (Int(l),))

    # reflect padding as used by scipy.gaussian_filter1d
    @inbounds @views for i in axes(X, 2)
        imfilter!(X[:, i], X[:, i], κ, "reflect")
    end
end

#hellinger_distance(X::AbstractVecOrMat, X̃::AbstractVecOrMat) =
#    sqrt.(1.0f0 .- sum(sqrt.(max.(0, X .* X̃)), dims = 1))

hellinger_distance(X::AbstractVecOrMat, X̃::AbstractVecOrMat) =
    eltype(X̃)(1 / √2) .* sqrt.(sum((sqrt.(X) .- sqrt.(X̃)) .^ 2, dims = 1))
