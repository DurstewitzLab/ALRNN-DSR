module Models

using Flux, LinearAlgebra

using ..Utilities

export AbstractALRNN,
    ALRNN,
    generate,
    jacobian,
    uniform_init,
    gaussian_init

include("initialization.jl")
include("alrnn.jl")

end