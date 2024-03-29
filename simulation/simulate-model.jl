using ArgParse
using DataFrames
using Distances
using Mamba
using SpatialModels
using YAML

function parse_commandline()
    s = ArgParseSettings("Fit a model to artificially-generated data.")
    @add_arg_table s begin
        "output"
            help = "File to which simulation results are written."
        "--iters", "-i"
            arg_type = Int
            default = 500
            help = "Number of MCMC iterations."
        "--burnin", "-b"
            arg_type = Int
            default = 0
            help = "Number of initial MCMC iterations to remove as burn-in."
        "--thin", "-t"
            arg_type = Int
            default = 1
            help = "Retain one of every 'thin' iterations."
        "--chains", "-c"
            arg_type = Int
            default = 1
            help = "Number of MCMC chains to run."
        "--data"
            help = "YAML defining the settings for artificial data generation."
        "--monitor"
            help = "YAML defining the nodes to monitor through MCMC."
        "--inits"
            help = "YAML defining initial values of stochastic nodes."
        "--hyper"
            help = "YAML defining hyperparameters of stochastic node priors."
        "--covariate"
            help = "A subtype of SpatialCovariateMissingness."
    end
    return parse_args(s)
end

function generate_data(;
        T::Int=10, nm::Int=50, m_active::Float64=0.5, p::Int=1, misspecify::Bool=false,
        βy::Vector{Float64}=zeros(p), βz::Vector{Float64}=zeros(p), βw::Vector{Float64}=zeros(p),
        αy::Float64=0.0, αw::Float64=0.0, σ2y::Float64=1.0, σ2z::Float64=1.0, σ2w::Float64=0.5,
        τ2y::Float64=1.0, τ2z::Float64=1.0, ϕy::Float64=1.0, ϕz::Float64=1.0, ϕw::Float64=1.0)

    @assert 0.0 <= m_active <= 1.0

    S = expandgrid(0.0:0.05:1.0)
    Sk = expandgrid(0.0625:0.125:0.9375)
    ux, uy = [Uniform(minmax...) for minmax in extrema(S, 2)]
    Sm = transpose(hcat(rand(ux, nm), rand(uy, nm)))
    nearest_cell = dimmin(pairwise(Euclidean(), S, Sm), 1)

    n = size(S, 2)
    x = ones(n, 1)
    xm = ones(nm, 1)
    p -= 1
    while p > 0
        v = randn(n)
        x = hcat(x, v)
        xm = hcat(xm, v[nearest_cell])
        p -= 1
    end
    X = array3d_from_mats([x for _ in 1:T])
    Xm = array3d_from_mats([xm for _ in 1:T])
    p = size(X, 2)

    # generate spatial covariate
    Xβz = matrix_from_rows([X[:, :, t] * βz for t in 1:T])
    D = pairwise(metric, S)
    ηz = transpose(rand(MvNormal(σ2z .* exponential(D, ϕz)), T))
    z_mean = Xβz + ηz
    z = matrix_from_rows([rand(MvNormal(z_mean[t, :], sqrt(τ2z))) for t in 1:T])

    # generate response variable conditional on spatial covariate
    zαy = z .* αy
    Xβy = matrix_from_rows([X[:, :, t] * βy for t in 1:T])
    ηy = transpose(rand(MvNormal(σ2y .* exponential(D, ϕy)), T))
    y_mean = zαy + Xβy + ηy
    y = matrix_from_rows([rand(MvNormal(y_mean[t, :], sqrt(τ2y))) for t in 1:T])

    # generate missingness factor conditional on spatial covariate
    z_meanαw = z_mean .* αw
    Xβw = matrix_from_rows([X[:, :, t] * βw for t in 1:T])
    τ2w = 1 - σ2w
    ηw = transpose(rand(MvNormal(σ2w .* exponential(D, ϕw)), T))
    if misspecify
        Φ = cdf(Normal(), ηz ./ sqrt(σ2z))
        a = sqrt(σ2z) ./ sqrt(Φ .* (1 .- Φ))
        b = Xβz .- sqrt(σ2z) .* sqrt(Φ) ./ sqrt(1 .- Φ)
        w_mean = αw .* (a .* (ηz .> 0) + b) + Xβw + ηw
    else
        w_mean = z_meanαw + Xβw + ηw
    end
    w = matrix_from_rows([rand(MvNormal(w_mean[t, :], sqrt(τ2w))) for t in 1:T])

    # make spatial covariate missing based on missingness factor
    z_all = copy(z)
    for i in eachindex(z, w)
        if w[i] > 0.0
            z[i] = NaN
        end
    end

    # make only a subset of monitors active at each timepoint
    y_all = copy(y)
    for i in eachindex(y)
        if rand() > m_active
            y[i] = NaN
        end
    end
    ym = y[:, nearest_cell]

    # define and return dicts of data (for mcmc) and truth (for validation)
    truth = Dict{Symbol, Any}(
        :y => y_all,
        :z => z_all,
        :αy => αy,
        :βy => βy,
        :σ2y => σ2y,
        :τ2y => τ2y,
        :ϕy => ϕy,
        :βz => βz,
        :σ2z => σ2z,
        :τ2z => τ2z,
        :ϕz => ϕz,
        :αw => αw,
        :βw => βw,
        :σ2w => σ2w,
        :ϕw => ϕw
    )

    data = Dict{Symbol, Any}(
        :ym => ym,
        :Xm => Xm,
        :Sm => Sm,
        :y => y,
        :z => z,
        :X => X,
        :S => S,
        :Sk => Sk,
        :D => D,
        :T => T,
        :n => n,
        :nm => nm,
        :p => p
    )
    data[:nk] = size(Sk, 2)
    data[:Ck] = pairwise(Euclidean(), S, Sk)
    data[:Dk] = pairwise(Euclidean(), Sk)
    data[:Dm] = pairwise(Euclidean(), Sm)
    data[:y_missing] = isnan.(y)
    data[:z_missing] = isnan.(z)

    return data, truth
end

args = parse_commandline()

@assert 0 < args["iters"]   "Iters must be positive"
@assert 0 <= args["burnin"] "Burn-in must be non-negative"
@assert 0 < args["thin"]    "Thin must be positive"
@assert 0 < args["chains"]  "Chains must be positive"

missingness = ["NotAvailable", "MissingAtRandom", "MissingNotAtRandom"]
@assert args["covariate"] in missingness
covariate = eval(Symbol(args["covariate"]))

monitor_conf = load_config(abspath(args["monitor"]))
hyper_conf = load_config(abspath(args["hyper"]))
data_conf = load_config(abspath(args["data"]))
inits_conf = load_config(abspath(args["inits"]))

model = get_model(covariate, monitor_conf, hyper_conf)
data, truth = generate_data(; data_conf...)
inits = get_inits(covariate, inits_conf, data)
inits = [inits for _ in 1:args["chains"]]

mcmc_kwargs = Dict(Symbol(key) => args[key] for key in ["burnin", "thin", "chains"])
sim = mcmc(model, data, inits, args["iters"]; mcmc_kwargs...)

# summarize simulation results in DataFrame
results = DataFrame(MambaName = sim.names)
nodes = Symbol[]
values = Float64[]
for name in results[:MambaName]
    for (node, value) in truth
        if startswith(name, String(node))
            push!(nodes, node)
            if '[' in name
                index = strip(name, [collect(String(node))..., '[', ']'])
                index = parse.(split(index, ','))
                push!(values, value[index...])
            else
                push!(values, value)
            end
        end
    end
end
results[:MambaNode] = nodes
results[:Value] = values

post_summary = summarystats(sim)
post_quantiles = quantile(sim)
results[:Mean] = post_summary.value[:, 1]
for (i, q) in enumerate(post_quantiles.colnames)
    results[Symbol(q)] = post_quantiles.value[:, i]
end

print(results)
writetable(abspath(args["output"]), results)
