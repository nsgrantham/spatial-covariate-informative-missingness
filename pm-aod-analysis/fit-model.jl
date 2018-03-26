using ArgParse
using DataFrames
using Distances: Metric, pairwise, get_pairwise_dims
using Mamba
using SpatialModels
using YAML

import Distances: pairwise!

function parse_commandline()
    s = ArgParseSettings("Fit a model to artificially-generated data.")
    @add_arg_table s begin
        "output"
            help = "File to which analysis results are written."
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
            help = "Directory containing PM-AOD data files."
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

struct GreatCircle <: Metric end

function greatcircle(s1, s2)
    lat1, lon1 = s1
    lat2, lon2 = s2
    ϕ1 = deg2rad(90 - lat1)
    ϕ2 = deg2rad(90 - lat2)
    θ1 = deg2rad(lon1)
    θ2 = deg2rad(lon2)
    cos_ = sin(ϕ1) * sin(ϕ2) * cos(θ1 - θ2) + cos(ϕ1) * cos(ϕ2)
    arc = acos(clamp(cos_, -1, 1))
    arc_len = 6373  # km
    return arc_len * arc
end

function pairwise!(r::AbstractMatrix, dist::GreatCircle, a::AbstractMatrix, b::AbstractMatrix)
    m, na, nb = get_pairwise_dims(r, a, b)
    @assert m == 2
    @inbounds for j in 1:nb
        for i in 1:na
            r[i, j] = max(greatcircle(a[:, i], b[:, j]), 0)
        end
    end
    r
end

function pairwise!(r::AbstractMatrix, dist::GreatCircle, a::AbstractMatrix)
    m, n = get_pairwise_dims(r, a)
    @assert m == 2
    @inbounds for j in 1:n
        for i in j+1:n
            r[i, j] = max(greatcircle(a[:, i], a[:, j]), 0)
        end
        r[j, j] = 0
        for i in 1:j-1
            r[i, j] = r[j, i]
        end
    end
    r
end

function load_data(dirname)

    drop_cols = [:date, :monID, :gridID]
    function selectcols(df)
        keep_cols = filter(col -> !(col in drop_cols), names(df))
        df[:, keep_cols]
    end

    Sm = transpose(convert(Matrix{Float64}, selectcols(readtable(joinpath(dirname, "Sm.csv")))))
    S = transpose(convert(Matrix{Float64}, selectcols(readtable(joinpath(dirname, "S.csv")))))

    # Mamba requires NaNs over NAs to represent missing values
    function na2nan!(df)
        for col in eachcol(df)
            name, vals = col
            df[isna.(vals), name] = NaN 
        end
    end

    function bydate(df)
        keep_cols = filter(col -> !(col in drop_cols), names(df))
        xs = Matrix{Float64}[]
        for subdf in groupby(df, :date)
            subdf = subdf[:, keep_cols]
            na2nan!(subdf)
            x = convert(Matrix{Float64}, subdf)
            push!(xs, x)
        end
        xs
    end

    ym = readtable(joinpath(dirname, "ym.csv"))
    ym = matrix_from_rows(bydate(ym))

    zm = readtable(joinpath(dirname, "zm.csv"))
    zm = matrix_from_rows(bydate(zm))
    
    Xm = readtable(joinpath(dirname, "Xm.csv"))
    Xm = array3d_from_mats(bydate(Xm))

    y = readtable(joinpath(dirname, "y.csv"))
    y = matrix_from_rows(bydate(y))

    z = readtable(joinpath(dirname, "z.csv"))
    z = matrix_from_rows(bydate(z))

    X = readtable(joinpath(dirname, "X.csv"))
    X = array3d_from_mats(bydate(X))

    mapping = readtable(joinpath(dirname, "mapping.csv"))
    cell_of_mon = Dict{Int, Int}(zip(mapping[:monID], mapping[:cellID]))

    data = Dict{Symbol, Any}(
        :ym => ym,
        :Xm => Xm,
        :Sm => Sm,
        :zm => zm,
        :y => y,
        :z => z,
        :X => X,
        :S => S,
        :cell_of_mon => cell_of_mon
    )
    data[:T], data[:n] = size(y)
    data[:nm] = size(Sm, 2)
    data[:p] = size(X, 2)
    data[:D] = pairwise(GreatCircle(), S)
    data[:Dm] = pairwise(GreatCircle(), Sm)
    data[:y_missing] = isnan.(y)
    data[:z_missing] = isnan.(z)

    return data
end

args = parse_commandline()

@assert 0 < args["iters"]   "Iters must be positive"
@assert 0 <= args["burnin"] "Burn-in must be non-negative"
@assert 0 < args["thin"]    "Thin must be positive"
@assert 0 < args["chains"]  "Chains must be positive"

missingness = Dict(
    "NotAvailable" => NotAvailable, 
    "MissingAtRandom" => MissingAtRandom, 
    "MissingNotAtRandom" => MissingNotAtRandom
)
@assert args["covariate"] in collect(keys(missingness))
covariate = missingness[args["covariate"]]

monitor_conf = load_config(abspath(args["monitor"]))
hyper_conf = load_config(abspath(args["hyper"]))
inits_conf = load_config(abspath(args["inits"]))

@assert isdir(args["data"])
data = load_data(args["data"])
data[:Sk] = overlaypoints(data[:S], metric=GreatCircle(), ticks=12, margin=0., maxdist=10.)
data[:nk] = size(data[:Sk], 2)
data[:Dk] = pairwise(GreatCircle(), data[:Sk])
data[:Ck] = pairwise(GreatCircle(), data[:S], data[:Sk])

model = get_model(covariate, monitor_conf, hyper_conf)
inits = get_inits(covariate, inits_conf, data)
inits = [inits for _ in 1:args["chains"]]

mcmc_kwargs = Dict(Symbol(key) => args[key] for key in ["burnin", "thin", "chains"])
sim = mcmc(model, data, inits, args["iters"]; mcmc_kwargs...)

# summarize simulation results in DataFrame
results = DataFrame(MambaName = sim.names)
nodes = Symbol[]
for name in results[:MambaName]
    for node in keys(monitor_conf)
        if startswith(name, String(node))
            push!(nodes, node)
        end
    end
end
results[:MambaNode] = nodes

post_summary = summarystats(sim)
post_quantiles = quantile(sim)
results[:Mean] = post_summary.value[:, 1]
for (i, q) in enumerate(post_quantiles.colnames)
    results[Symbol(q)] = post_quantiles.value[:, i]
end

print(results)
writetable(abspath(args["output"]), results)
