# Spatial regression model that does _not_ include an
# informatively-missing covariate

using Mamba: Model, Stochastic, Logical, Sampler, setsamplers!
using StatsBase: Weights
using Distributions

include(joinpath(@__DIR__, "..", "utils.jl"))

function get_model(monitor::Dict{Symbol, Any}, hyper::Dict{Symbol, Any})

    model = Model(
        y = Stochastic(2,
            (y_mean, τ2y, T) -> MultivariateDistribution[
                MvNormal(y_mean[t, :], sqrt(τ2y)) for t in 1:T
            ],
            monitor[:y_pred]
        ),

        y_mean = Logical(2,
            (Xβy, Hηy) -> Xβy + Hηy,
            false
        ),

        Xβy = Logical(2,
            (X, βy, T) -> matrix_from_rows([X[:, :, t] * βy for t in 1:T]),
            false
        ),

        Hηy = Logical(2,
            (Hy, ηy) -> ηy * transpose(Hy),
            false
        ),

        τ2y = Stochastic(
            () -> InverseGamma(hyper[:τ2y]["a"], hyper[:τ2y]["b"]),
            monitor[:τ2y]
        ),

        y_sss = Logical(
            (ηy, Ry_inv, T) -> sum([(transpose(ηy[t, :]) * Ry_inv * ηy[t, :])[1] for t in 1:T]),
            false
        ),

        ηy = Stochastic(2,
            (σ2y, Ry, T) -> MultivariateDistribution[
                MvNormal(σ2y .* Ry) for t in 1:T
            ],
            monitor[:ηy]
        ),

        y_sse = Logical(
            (y, y_mean) -> sum(abs2, y - y_mean),
            false
        ),

        βy = Stochastic(1,
            (p) -> MvNormal(p, sqrt(hyper[:βy]["var"])),
            monitor[:βy]
        ),

        σ2y = Stochastic(
            () -> InverseGamma(hyper[:σ2y]["a"], hyper[:σ2y]["b"]),
            monitor[:σ2y]
        ),

        ϕy = Logical(
            (ϕy_support, ϕy_index) -> ϕy_support[round(Int, ϕy_index)],
            monitor[:ϕy]
        ),

        ϕy_index = Stochastic(
            (ϕy_support) -> Categorical(length(ϕy_support)),
            false
        ),

        ϕy_support = Logical(1,
            () -> collect(hyper[:ϕy]["start"]:hyper[:ϕy]["by"]:hyper[:ϕy]["end"]),
            false
        ),

        HytHy = Logical(2,
            (HytHy_array, ϕy_index) -> HytHy_array[:, :, round(Int, ϕy_index)],
            false
        ),

        Hy = Logical(2,
            (Hy_array, ϕy_index) -> Hy_array[:, :, round(Int, ϕy_index)],
            false
        ),

        Ry_inv = Logical(2,
            (Ry_inv_array, ϕy_index) -> Ry_inv_array[:, :, round(Int, ϕy_index)],
            false
        ),

        Ry = Logical(2,
            (Ry_array, ϕy_index) -> Ry_array[:, :, round(Int, ϕy_index)],
            false
        ),


        # The following terms are only computed once and are used repeatedly
        # in the other nodes in the model and their posterior samplers.

        HytHy_array = Logical(3,
            (Hy_array) -> array3d_from_mats(
                [(Hy = Hy_array[:, :, i]; transpose(Hy) * Hy) for i in 1:size(Hy_array, 3)]
            ),
            false
        ),

        Hy_array = Logical(3,
            (Ck, ϕy_support, Ry_inv_array) -> array3d_from_mats(
                [exponential(Ck, ϕy_support[i]) * Ry_inv_array[:, :, i] for i in eachindex(ϕy_support)]
            ),
            false
        ),

        Ry_inv_array = Logical(3,
            (Ry_array) -> array3d_from_mats(
                [inv(Ry_array[:, :, i]) for i in 1:size(Ry_array, 3)]
            ),
            false
        ),

        logdetRy_array = Logical(1,
            (Ry_array, ϕy_support) -> Float64[
                logdet(Ry_array[:, :, i]) for i in 1:size(Ry_array, 3)
            ],
            false
        ),

        Ry_array = Logical(3,
            (ϕy_support, Dk) -> array3d_from_mats(
                [exponential(Dk, ϕ) for ϕ in ϕy_support]
            ),
            false
        ),

        sumXtX = Logical(2,
            (X, T) -> sum(
                [(x = X[:, :, t]; transpose(x) * x) for t in 1:T]
            ),
            false
        ),

        # It may seem silly to create new nodes in our DAG simply to
        # compute the following values, but there is a reason for it.
        # The eval function supplied to Mamba.Sampler requires that its
        # arguments are known to the network in some fashion. By defining
        # the following nodes, we can pass their value and the value of
        # their arguments to our custom posterior sampling functions.
        # E.g., Removing N0 here would not allow the :τ2y Sampler below
        # to use N0 or even n0 in its update of hyperparameter u because
        # no other node in the network directly requires n0.

        y_missing_total = Logical(
            (y_missing) -> sum(y_missing),
            false
        ),

        n_total = Logical(
            (n, T) -> n * T,
            false
        ),

        nk_total = Logical(
            (nk, T) -> nk * T,
            false
        ),
    )

    samplers = [
        Sampler(:y, (y, y_mean, y_missing, τ2y) ->
            begin
                for i in eachindex(y, y_mean, y_missing)
                    if y_missing[i]
                        y[i] = rand(Normal(y_mean[i], sqrt(τ2y)))
                    end
                end
                y
            end
        ),

        Sampler(:τ2y, (y_sse, τ2y, n_total) ->
            begin
                a = n_total / 2 + shape(τ2y.distr)
                b = y_sse / 2 + scale(τ2y.distr)
                rand(InverseGamma(a, b))
            end
        ),

        Sampler(:βy, (y, Hηy, βy, X, sumXtX, τ2y, T) ->
            begin
                Σ = inv(sumXtX ./ τ2y + invcov(βy.distr))
                A = y - Hηy
                B = Vector{Float64}[transpose(X[:, :, t]) * A[t, :] for t in 1:T]
                μ = Σ * sum(B) ./ τ2y
                rand(MvNormal(μ, Hermitian(Σ)))
            end
        ),

        Sampler(:ηy, (y, Xβy, ηy, Hy, HytHy, Ry_inv, σ2y, τ2y, T) ->
            begin
                A = y - Xβy
                Σ = inv(HytHy ./ τ2y + Ry_inv ./ σ2y)
                for t in 1:T
                    μ = Σ * transpose(Hy) * (A[t, :] ./ τ2y)
                    ηy[t, :] = rand(MvNormal(μ, Hermitian(Σ)))
                end
                ηy
            end
        ),

        Sampler(:σ2y, (σ2y, y_sss, nk_total) ->
            begin
                a = nk_total / 2 + shape(σ2y.distr)
                b = y_sss / 2 + scale(σ2y.distr)
                rand(InverseGamma(a, b))
            end
        ),

        Sampler(:ϕy_index, (y, Xβy, ηy, σ2y, τ2y, logdetRy_array, Ry_inv_array, Hy_array, T, ϕy_support) ->
            begin
                logpost = zeros(ϕy_support)
                A = y - Xβy
                ϕy_index_array = 1:length(ϕy_support)
                for i in ϕy_index_array
                    logpost[i] = -sum(abs2, A - ηy * transpose(Hy_array[:, :, i])) / (2 * τ2y)
                    logpost[i] += -(T / 2) * logdetRy_array[i]
                    for t in 1:T
                        logpost[i] += -(transpose(ηy[t, :]) * Ry_inv_array[:, :, i] * ηy[t, :])[1] / (2 * σ2y)
                    end
                end
                logpost .-= maximum(logpost)
                sample(ϕy_index_array, Weights(exp.(logpost)))
            end
        )
    ]

    setsamplers!(model, samplers)
    return model
end

function get_inits(params::Dict{Symbol, Any}, data::Dict{Symbol, Any})
    y_init = copy(data[:y])
    y_mean = nanmean(y_init)
    for i in eachindex(y_init)
        if isnan(y_init[i])
            y_init[i] = y_mean
        end
    end

    βy_init = params[:βy] == "zeros" ? zeros(data[:p]) : params[:βy]

    inits = Dict{Symbol, Any}(
        :y => y_init,
        :ηy => zeros(data[:T], data[:nk]),
        :βy => βy_init,
        :αy => params[:αy],
        :σ2y => params[:σ2y],
        :τ2y => params[:τ2y],
        :ϕy_index => params[:ϕy]["index"]
    )
    return inits
end
