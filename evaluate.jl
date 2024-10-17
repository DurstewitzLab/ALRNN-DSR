using SymbolicDSR
using LinearAlgebra
using Plots
using JSON
using JLD
using NPZ
using Statistics
using LaTeXStrings
using NaNStatistics

"""
Performance evaluation:
Function evaluates performance of an experiment provided the arguments and number of runs done in this experiment
The l specifier filters out failed trainings
"""

function eval(args, runs, l=20)
    Dstsp = []
    DH = []

    for i in 1:runs
        metrics = load(joinpath(pwd(), "Results", args["experiment"], args["name"], Utilities.format_run_ID(i), "metrics.jld"))["metrics"]
        metrics = Matrix(mapreduce(permutedims,vcat, metrics)')
        
        if length(metrics[2,:])>=l # Filter out failed trainings
            push!(Dstsp, nanminimum(metrics[2,:])) # Get best performing model
            if isnan(metrics[3,nanargmin(metrics[2,:])]) # Check for divergence
                push!(DH, 1.0) 
            else
                push!(DH, metrics[3,nanargmin(metrics[2,:])])
            end
        else
            push!(Dstsp, NaN)
            push!(DH, NaN)
        end

    end
    return Dstsp, DH
end

"""
Subregion evaluation:
This function generates subregion specific information about the model. It requires the ALRNN model (model), the observation model (O) and the data (data).
For detailed statistics set detailed = true.

The function evaluates:
    LS_N: the number of used linear subregions in the model
    LS_bit_v: the symbolic sequence encoded as a bit vector
    LS_int: the symbolic sequence (symbols sorted from highest freq. visited subregions to lowest)
    LS_cum: Amount of datapoints in the respective subregions
    Z: the latent space trajectory
"""
function LS(model, O, data, detailed =false)

    z₁ = init_state(O, data[1,:])
    Z = generate(model, z₁, Int(1e5))

    #Count subregions
    LS_bit = Z .> 0
    LS_bit_v = [LS_bit[i,end-(model.P-1):end] for i in 1:size(LS_bit,1)]
    LS_N = length(unique(LS_bit_v))

    if detailed
        #Count their dominance
        LS_cum = []
        for i in 1:LS_N
            push!(LS_cum, sum(LS_bit_v .== [unique(LS_bit_v)[i]]))
        end

        #Count transitions
        LS_int_temp = zeros(Int(1e5))
        for i in 1:LS_N
            LS_int_temp[LS_bit_v .== [unique(LS_bit_v)[i]]] .= i
        end
        LS_int = copy(LS_int_temp)
        LS_cum_temp = copy(LS_cum)
        #Sort for highest dominance
        for i in 1:LS_N
            LS_int[LS_int_temp .== argmax(LS_cum_temp)] .= i
            LS_cum_temp[Int(argmax(LS_cum_temp))] = 0
        end

        return LS_N, LS_cum, Int.(LS_int), LS_bit_v, Z
    else
        return LS_N, nothing, nothing, LS_bit_v, Z
    end
end

"""
Fixed point and stability evaluation:
This function evaluates the location of the fixed points in each subregion (z_fp) and the respective stability of that subregion (stab)
"""
function calc_FP(model, relu,M,LS_bit)
    z_fp = []
    stab = []
    
    for i in 1:Int(2^relu) # Go through all possible subregions
        bit=bitstring(i)[end-(relu-1):end]
        bit = Float64.(parse.(Int,split(bit,"")))
        D = Diagonal(vcat(ones(M-relu),bit)) # subregion encoding
        if bit in LS_bit # Calculate FP and theri stability for all used subregions in the model
            push!(z_fp,(inv(Matrix(I, M, M) - Diagonal(model.A) -model.W * D) * model.h))
            push!(stab,maximum(norm.(eigvals(Diagonal(model.A) + model.W * D))))
        end
    end
    z_fp = mapreduce(permutedims,vcat,z_fp)
    
    return z_fp, stab
end



"""
Evaluate subregions and their dynamical properties
"""

# Load arguments
load_defaults() = load_json_f32(joinpath(pwd(), "settings", "defaults.json"))
args = load_defaults(); # Load settings, make sure to specify the correct model path
data = npzread(joinpath(pwd(),"example_data/lorenz63.npy")) # load data
id = 1 # run id

# Load performance metrics
metrics = load(joinpath(pwd(), "Results", args["experiment"], args["name"], Utilities.format_run_ID(id), "metrics.jld"))["metrics"]
metrics = Matrix(mapreduce(permutedims,vcat, metrics)')
plot(metrics[2,:],yaxis=L"D_{stsp}",xaxis="epoch")
plot(metrics[3,:],yaxis=L"D_{H}",xaxis="epoch")
best_id = argmin(metrics[2,:]) # Get best performing model

# Load model
model, O = load_model(joinpath("Results", args["experiment"], args["name"], Utilities.format_run_ID(id), "checkpoints", "model_"*string(args["scalar_saving_interval"]*best_id)*".bson"))
LS_N, LS_cum, LS_int, LS_bit, Z=LS(model,O,data,true) # Analyse linear subregions


# Plotting
lim=10000
plot(Z[1:lim,1],Z[1:lim,2],Z[1:lim,3],axis=([], false),lw=2,markerstrokewidth=0,markersize=6,label="generated trajectory",colorbar=false)
scatter(Z[1:lim,1],Z[1:lim,2],Z[1:lim,3],axis=([], false),lw=2,zcolor=LS_int,markerstrokewidth=0,markersize=6,label="",colorbar=false,color=:blues)

z_fp, stab = calc_FP(model,2,20,LS_bit)
scatter!(z_fp[:,1],z_fp[:,2],z_fp[:,3],markersize=8,msw=0,color="gray",label="learned FP")


"""
Evaluate model setting performance (e.g. Fig 3)
"""

# Load arguments
load_defaults() = load_json_f32(joinpath(pwd(), "settings", "defaults.json"))
args = load_defaults();
runs = 10 # number of runs per experimental setting
Dstsp, DH = eval(args, runs)


# Evaluate grid search over P
load_defaults() = load_json_f32(joinpath(pwd(), "settings", "defaults.json"))
args = load_defaults();
relus =  [0,1,2,3,4,5]
runs = 10

Dstsp = []
DH = []
for i in relus
    args["name"] = "ALRNN_M20-P_"*string(i)
    a, b = eval(args,runs)
    push!(Dstsp, a)
    push!(DH,b)
end
Dstsp = Float32.(mapreduce(permutedims,vcat,Dstsp))
DH = Float32.(mapreduce(permutedims,vcat,DH))

# Plotting
p1 = plot(relus,nanmean(Dstsp,dims=2),yerror=nansem(Dstsp,dims=2),
    label="",xlabel="",ylabel=L"$D_{stsp}$",
    lw=4,marker=:dot, markersize=7,msw=2,markerstrokecolor=palette(:grayC10)[2],color=palette(:grayC10)[2],
    tickfontsize=18, labelfontsize=20,legendfontsize=18,size=(500,550)#,yticks=([2,4,6,8])
)
p5 = plot(relus,nanmean(DH,dims=2),yerror=nansem(DH,dims=2),
    label="",xlabel="# PWL units "*L"(P)",ylabel=L"$D_{H}$",
    lw=4,marker=:dot, markersize=7,msw=2,markerstrokecolor=palette(:grayC10)[2],color=palette(:grayC10)[2],
    tickfontsize=18, labelfontsize=20,legendfontsize=18,size=(500,550),yticks=([0.1,0.5,0.9]),bottommargin=12Plots.mm
)
title = plot(title = L"\textbf{Lorenz\;63}", grid = false, showaxis = false, bottom_margin = -80Plots.px,titlefontsize=30)
plot(title,p1,p5,margin=5Plots.mm,layout = @layout([A{0.01h}; B; C]))




"""
Evaluate # subregions vs. # ReLUs (e.g. Fig 4 Left)
"""

load_defaults() = load_json_f32(joinpath(pwd(), "settings", "defaults.json"))
args = load_defaults();
relus =  [0,1,2,3,4,5] # investigated relus
runs = 10 # number of runs per experiment
data = npzread(joinpath(pwd(),"example_data/lorenz63.npy"))

N = []
for i in relus
    n = []
    for j in 1:runs
        args["name"] = "ALRNN_M20-P_"*string(i)
        metrics = load(joinpath(pwd(), "Results", args["experiment"], args["name"], Utilities.format_run_ID(j), "metrics.jld"))["metrics"]
        metrics = Matrix(mapreduce(permutedims,vcat, metrics)')
        best_id = argmin(metrics[2,:]) # Get best performing model
        model, O = load_model(joinpath("Results", args["experiment"], args["name"], Utilities.format_run_ID(j), "checkpoints", "model_"*string(args["scalar_saving_interval"]*best_id)*".bson"))
        LS_N, _, _, _, _ = LS(model, O, data) # Compute number of used subregions
        push!(n, LS_N)
    end
    push!(N,n)
end
N=Float32.(mapreduce(permutedims,vcat,N))

# Plotting
plot(relus,nanmean(N,dims=2),yerror=nanstd(N,dims=2),
    label="Lorenz-63",xlabel="# PWL units "*L"(P)",ylabel="Used linear subregions",
    lw=5,marker=:dot, markersize=10,msw=3,markerstrokecolor=palette(:viridis)[1],color=palette(:viridis)[1],
    yaxis=:log,tickfontsize=18, labelfontsize=20,legendfontsize=18,size=(1000,550),margin=10Plots.mm
)
p1=plot!(relus,2 .^relus,
    label="Total",xlabel="# PWL units "*L"(P)",ylabel="# subregions",
    lw=5,marker=:dot, markersize=10,msw=3,markerstrokecolor=palette(:tab10)[4],color=palette(:tab10)[4],
    yaxis=:log,tickfontsize=22, labelfontsize=24,legendfontsize=20,size=(1000,400),margin=9Plots.mm,legend=:topleft
)



"""
Datapoints covered through subregions (e.g. Fig 4 Right)
"""

load_defaults() = load_json_f32(joinpath(pwd(), "settings", "defaults.json"))
args = load_defaults();
data = npzread(joinpath(pwd(),"example_data/lorenz63.npy"))
id = 1 # run index

# Get model
metrics = load(joinpath(pwd(), "Results", args["experiment"], args["name"], Utilities.format_run_ID(id), "metrics.jld"))["metrics"]
metrics = Matrix(mapreduce(permutedims,vcat, metrics)')
best_id = argmin(metrics[2,:]) # Get best performing one

model, O = load_model(joinpath("Results", args["experiment"], args["name"], Utilities.format_run_ID(id), "checkpoints", "model_"*string(args["scalar_saving_interval"]*best_id)*".bson"))
LS_N, LS_cum, LS_int, LS_bit, Z=LS(model,O,data,true) # Calculate subregions information

plot(cumsum(reverse(sort(LS_cum)))./1e5 .*100,
    xlabel="Number of subregions",ylabel="Total datapoints covered (%)",label="Lorenz-63",
    lw=7, markersize=0,msw=3,color=palette(:viridis)[1],
)


