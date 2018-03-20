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
