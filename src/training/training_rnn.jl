using Flux
using BSON: @save
using DataStructures
using Plots: savefig
using JLD

using ..Measures
using ..Models
using ..ObservationModels
using ..Utilities

"""
    loss(tfrec, X̃, S̃)

Performs a forward pass using the teacher forced recursion wrapper `tfrec` and
computes and return the loss w.r.t. data `X̃`.
"""
function loss(tfrec::AbstractTFRecur, X̃::AbstractArray{T, 3}) where {T}
    Z = tfrec(X̃)
    X̂ = tfrec.O(Z)
    return @views Flux.mse(X̂, X̃[:, :, 2:end])
end


function train_!(
    m::AbstractALRNN,
    O::ObservationModel,
    d::Dataset,
    opt::Flux.Optimise.Optimiser,
    args::AbstractDict,
    save_path::String,
)
    # data shape
    T, N = size(d.X)

    #metrics
    metrics = []

    # hypers (type hinting reduces startup time drastically)
    E = args["epochs"]::Int
    M = args["latent_dim"]::Int
    Sₑ = args["batches_per_epoch"]::Int
    S = args["batch_size"]::Int
    τ = args["teacher_forcing_interval"]::Int
    σ_noise = args["gaussian_noise_level"]::Float32
    T̃ = args["sequence_length"]::Int
    σ²_scaling = args["D_stsp_scaling"]::Float32
    bins = args["D_stsp_bins"]::Int
    σ_smoothing = args["PSE_smoothing"]::Float32
    PE_n = args["PE_n"]::Int
    isi = args["image_saving_interval"]::Int
    ssi = args["scalar_saving_interval"]::Int
    exp = args["experiment"]::String
    name = args["name"]::String
    run = args["run"]::Int
    λₒ = args["obs_model_regularization"]::Float32
    λₗ = args["lat_model_regularization"]::Float32

    prog = Progress(joinpath(exp, name), run, 20, E, 0.8)
    stop_flag = false

    # decide on D_stsp scaling
    scal, stsp_name = decide_on_measure(σ²_scaling, bins, N)

    # initialize stateful model wrapper
    tfrec = TFRecur(m, O, similar(d.X, M, S), τ)

    # model parameters
    θ = Flux.params(tfrec)

    for e = 1:E
        # process a couple of batches
        t₁ = time_ns()
        for sₑ = 1:Sₑ
            # sample a batch
            X̃ = sample_batch(d, T̃, S)

            # add noise noise if noise level > 0
            σ_noise > zero(σ_noise) ? add_gaussian_noise!(X̃, σ_noise) : nothing

            # forward and backward pass
            grads = Flux.gradient(θ) do
                Lₜᵣ = loss(tfrec, X̃)
                Lᵣ = regularization_loss(tfrec, λₗ, λₒ)
                return Lₜᵣ + Lᵣ
            end

            # optimiser step
            Flux.Optimise.update!(opt, θ, grads)

            # check for NaNs in parameters (exploding gradients)
            stop_flag = check_for_NaNs(θ)
            if stop_flag
                break
            end
        end
        if stop_flag
            save_model(
                [tfrec.model, tfrec.O],
                joinpath(save_path, "checkpoints", "model_$e.bson"),
            )
            save(joinpath("Results", args["experiment"], args["name"], Utilities.format_run_ID(args["run"]), "metrics.jld"), "metrics", metrics)
            @warn "NaN(s) in parameters detected! \
                This is likely due to exploding gradients. Aborting training..."
            break
        end
        t₂ = time_ns()
        Δt = (t₂ - t₁) / 1e9
        update!(prog, Δt, e)

        # plot trajectory
        if e % ssi == 0
            # loss
            X̃ = sample_batch(d, T̃, S)
            Lₜᵣ = loss(tfrec, X̃)
            Lᵣ = regularization_loss(tfrec, λₗ, λₒ)

            # generated trajectory
            X_gen = @views generate(tfrec.model, tfrec.O, d.X[1, :], T)

            # move data to cpu for metrics and plotting
            X_cpu = d.X |> cpu
            X_gen_cpu = X_gen |> cpu

            # metrics
            D_stsp = state_space_distance(X_cpu, X_gen_cpu, scal)
            pse, _ = power_spectrum_error(X_cpu, X_gen_cpu, σ_smoothing)
            pe = prediction_error(tfrec.model, tfrec.O, d.X, PE_n)

            # progress printing
            scalars = gather_scalars(Lₜᵣ, Lᵣ, D_stsp, stsp_name, pse, pe, PE_n)
            push!(metrics, getindex.(Ref(scalars), ["∑L","Dₛₜₛₚ $stsp_name", "Dₕₑₗₗ","$PE_n-step PE"]))
            print_progress(prog, Δt, scalars)
            save_model(
                [tfrec.model, tfrec.O],
                joinpath(save_path, "checkpoints", "model_$e.bson"),
            )
            if e % isi == 0
                # plot
                plot_reconstruction(
                    X_gen_cpu,
                    X_cpu,
                    joinpath(save_path, "plots", "generated_$e.png"),
                )
            end
        end
    end

    #Save training metrics
    save(joinpath("Results", args["experiment"], args["name"], Utilities.format_run_ID(args["run"]), "metrics.jld"), "metrics", metrics)
    
end

function regularization_loss(
    tfrec::AbstractTFRecur,
    λₗ::Float32,
    λₒ::Float32,
)
    # latent model regularization
    Lᵣ = 0.0f0
    if λₗ > zero(λₗ)
        #Lᵣ += regularize(tfrec.model, λₗ)
        Lᵣ += AR_convergence_loss(tfrec.model, λₗ)
    end

    # observation model regularization
    Lᵣ += (λₒ > zero(λₒ)) ? regularize(tfrec.O, λₒ) : zero(λₒ)
    return Lᵣ
end

function gather_scalars(Lₜᵣ, Lᵣ, D_stsp, stsp_name, pse, pe, PE_n)
    scalars = OrderedDict("∑L" => Lₜᵣ + Lᵣ)
    if Lᵣ > 0.0f0
        scalars["Lₜᵣ"] = Lₜᵣ
        scalars["Lᵣ"] = Lᵣ
    end
    scalars["Dₛₜₛₚ $stsp_name"] = D_stsp
    scalars["Dₕₑₗₗ"] = pse
    scalars["$PE_n-step PE"] = pe
    return scalars
end