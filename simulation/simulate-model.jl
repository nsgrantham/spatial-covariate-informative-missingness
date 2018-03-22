using ArgParse
using Mamba
using YAML

include(joinpath(@__DIR__, "..", "utils.jl"))

function parse_commandline()
    s = ArgParseSettings("Fit a model to artificially-generated data.")
    @add_arg_table s begin
        "output"
            help = "File to which simulation results are written."
        "--iters", "-i"
            arg_type = Int
            default = 1000
            help = "Number of MCMC iterations."
        "--burnin", "-b"
            arg_type = Int
            default = 500
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
        "--model"
            help = "Julia file defining Mamba model and MCMC sampling scheme."
    end
    return parse_args(s)
end

function parse_config(conf)
    greek = Dict(
        "alpha" => "α",
        "beta" => "β",
        "eta" => "η",
        "sigma2" => "σ2",
        "tau2" => "τ2",
        "phi" => "ϕ"
    )
    parsed_conf = Dict{Symbol, Any}()
    for key in keys(conf)
        if key in ["y", "z", "w"]
            var_level = conf[key]
            for (param, value) in var_level
                parsed_conf[Symbol(greek[param] * key)] = value
            end
        else
            parsed_conf[Symbol(key)] = conf[key]
        end
    end
    return parsed_conf
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
    metric = Euclidean()
    nearest_cell = dimmin(pairwise(metric, S, Sm), 1)

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
        :y_all => y_all,
        :z_all => z_all
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
    data[:Ck] = pairwise(metric, S, Sk)
    data[:Dk] = pairwise(metric, Sk)
    data[:Dm] = pairwise(metric, Sm)
    data[:y_missing] = isnan.(y)
    data[:z_missing] = isnan.(z)

    return data, truth
end

args = parse_commandline()

@assert 0 < args["iters"]
@assert 0 < args["burnin"]
@assert 0 < args["thin"]
@assert 0 < args["chains"]

@assert isfile(args["model"])
include(abspath(args["model"]))  # get_model, get_inits

@assert isfile(args["monitor"])
monitor_conf = YAML.load(open(abspath(args["monitor"])))
monitor_dict = parse_config(monitor_conf)

@assert isfile(args["hyper"])
hyper_conf = YAML.load(open(abspath(args["hyper"])))
hyper_dict = parse_config(hyper_conf)

@assert isfile(args["data"])
data_conf = YAML.load(open(abspath(args["data"])))
data_dict = parse_config(data_conf)

@assert isfile(args["inits"])
inits_conf = YAML.load(open(abspath(args["inits"])))
inits_dict = parse_config(inits_conf)

model = get_model(monitor_dict, hyper_dict)
data, truth = generate_data(; data_dict...)
inits = get_inits(inits_dict, data)
inits = [inits for _ in 1:args["chains"]]

mcmc_dict = Dict(Symbol(key) => args[key] for key in ["burnin", "thin", "chains"])
sim = mcmc(model, data, inits, args["iters"]; mcmc_dict...)
describe(sim)
