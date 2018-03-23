# Spatial regression models that includes a spatially-missing covariate
# under the assumption that its values are missing-not-at-random

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
            (Xβy, zαy, Hηy) -> Xβy + zαy + Hηy,
            false
        ),

        zαy = Logical(2,
            (z, αy) -> αy .* z,
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

        αy = Stochastic(
            () -> Normal(0, sqrt(hyper[:αy]["var"])),
            monitor[:αy]
        ),

        βy = Stochastic(1,
            (p) -> MvNormal(p, sqrt(hyper[:βy]["var"])),
            monitor[:βy]
        ),

        σ2y = Stochastic(
            () -> InverseGamma(hyper[:σ2y]["a"], hyper[:σ2y]["b"]),
            monitor[:σ2y]
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

        z = Stochastic(2,
            (z_mean, τ2z, T) -> MultivariateDistribution[
                MvNormal(z_mean[t, :], sqrt(τ2z)) for t in 1:T
            ],
            monitor[:z_pred]
        ),

        τ2z = Stochastic(
            () -> InverseGamma(hyper[:τ2z]["a"], hyper[:τ2z]["b"]),
            monitor[:τ2z]
        ),

        z_mean = Logical(2,
            (Xβz, Hηz) -> Xβz + Hηz,
            false
        ),

        z_sse = Logical(
            (z, z_mean) -> sum(abs2, z - z_mean),
            false
        ),

        Hηz = Logical(2,
            (Hz, ηz) -> ηz * transpose(Hz),
            false
        ),

        z_sss = Logical(
            (ηz, Rz_inv, T) -> sum([(transpose(ηz[t, :]) * Rz_inv * ηz[t, :])[1] for t in 1:T]),
            false
        ),

        ηz = Stochastic(2,
            (σ2z, Rz, T) -> MultivariateDistribution[
                MvNormal(σ2z .* Rz) for t in 1:T
            ],
            monitor[:ηz]
        ),

        Xβz = Logical(2,
            (X, βz, T) -> matrix_from_rows([X[:, :, t] * βz for t in 1:T]),
            false
        ),

        βz = Stochastic(1,
            (p) -> MvNormal(p, sqrt(hyper[:βz]["var"])),
            monitor[:βz]
        ),

        σ2z = Stochastic(
            () -> InverseGamma(hyper[:σ2z]["a"], hyper[:σ2z]["b"]),
            monitor[:σ2z]
        ),

        ϕz = Logical(
            (ϕz_support, ϕz_index) -> ϕz_support[round(Int, ϕz_index)],
            monitor[:ϕz]
        ),

        ϕz_index = Stochastic(
            (ϕz_support) -> Categorical(length(ϕz_support)),
            false
        ),

        ϕz_support = Logical(1,
            () -> collect(hyper[:ϕz]["start"]:hyper[:ϕz]["by"]:hyper[:ϕz]["end"]),
            false
        ),

        HztHz = Logical(2,
            (HztHz_array, ϕz_index) -> HztHz_array[:, :, round(Int, ϕz_index)],
            false
        ),

        Hz = Logical(2,
            (Hz_array, ϕz_index) -> Hz_array[:, :, round(Int, ϕz_index)],
            false
        ),

        Rz_inv = Logical(2,
            (Rz_inv_array, ϕz_index) -> Rz_inv_array[:, :, round(Int, ϕz_index)],
            false
        ),

        Rz = Logical(2,
            (Rz_array, ϕz_index) -> Rz_array[:, :, round(Int, ϕz_index)],
            false
        ),

        # The following terms are only computed once

        HytHy_array = Logical(3,
            (Hy_array) -> array3d_from_mats(
                [(Hy = Hy_array[:, :, i]; transpose(Hy) * Hy) for i in 1:size(Hy_array, 3)]
            ),
            false
        ),

        Hy_array = Logical(3,
            (Ck, ϕy_support, Ry_inv_array) -> array3d_from_mats(
            [exponential(Ck, ϕy_support[i]) * Ry_inv_array[:, :, i] for i in 1:length(ϕy_support)]
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
            (Ry_array) -> Float64[
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

        HztHz_array = Logical(3,
            (Hz_array) -> array3d_from_mats(
                [(Hz = Hz_array[:, :, i]; transpose(Hz) * Hz) for i in 1:size(Hz_array, 3)]
            ),
            false
        ),

        Hz_array = Logical(3,
            (Ck, ϕz_support, Rz_inv_array) -> array3d_from_mats(
                [exponential(Ck, ϕz_support[i]) * Rz_inv_array[:, :, i] for i in 1:size(Rz_inv_array, 3)]
            ),
            false
        ),

        Rz_inv_array = Logical(3,
            (Rz_array) -> array3d_from_mats(
                [inv(Rz_array[:, :, i]) for i in 1:size(Rz_array, 3)]
            ),
            false
        ),

        logdetRz_array = Logical(1,
            (Rz_array) -> Float64[logdet(Rz_array[:, :, i]) for i in 1:size(Rz_array, 3)],
            false
        ),

        Rz_array = Logical(3,
            (ϕz_support, Dk) -> array3d_from_mats(
                [exponential(Dk, ϕ) for ϕ in ϕz_support]
            ),
            false
        ),

        w = Stochastic(2,
            (w_mean, τ2w, T) -> MultivariateDistribution[
                MvNormal(w_mean[t, :], sqrt(τ2w)) for t in 1:T
            ],
            monitor[:w_pred] 
        ),

        τ2w = Logical(
            (σ2w) -> 1 - σ2w,
            false
        ),

        w_mean = Logical(2,
            (Xβzαw, Hηzαw, Xβw, Hηw) -> Xβzαw + Hηzαw + Xβw + Hηw,
            false
        ),

        Xβw = Logical(2,
            (X, βw, T) -> matrix_from_rows([X[:, :, t] * βw for t in 1:T]),
            false
        ),

        βw = Stochastic(1,
            (p) -> MvNormal(p, sqrt(hyper[:βw]["var"])),
            monitor[:βw]
        ),

        Hηzαw = Logical(2,
            (αw, Hηz) -> αw .* Hηz,
            false
        ),

        Xβzαw = Logical(2,
            (αw, Xβz) -> αw .* Xβz,
            false
        ),

        αw = Stochastic(
            () -> Normal(0.0, sqrt(hyper[:αw]["var"])),
            monitor[:αw]
        ),

        Hηw = Logical(2,
            (Hw, ηw) -> ηw * transpose(Hw),
            false
        ),

        w_sse = Logical(
            (w, w_mean) -> sum(abs2, w - w_mean),
            false
        ),

        w_sss = Logical(
            (ηw, Rw_inv, T) -> sum(
                [(transpose(ηw[t, :]) * Rw_inv * ηw[t, :])[1] for t in 1:T]
            ),
            false
        ),

        ηw = Stochastic(2,
            (σ2w, Rw, T) -> MultivariateDistribution[
                MvNormal(σ2w .* Rw) for t in 1:T
            ],
            monitor[:ηw]
        ),

        σ2w = Stochastic(
            () -> Beta(hyper[:σ2w]["a"], hyper[:σ2w]["b"]),
            monitor[:σ2w]
        ),

        ϕw = Logical(
            (ϕw_support, ϕw_index) -> ϕw_support[round(Int, ϕw_index)],
            monitor[:ϕw]
        ),

        ϕw_index = Stochastic(
            (ϕw_support) -> Categorical(length(ϕw_support)),
            false
        ),

        ϕw_support = Logical(1,
            () -> collect(hyper[:ϕw]["start"]:hyper[:ϕw]["by"]:hyper[:ϕw]["end"]),
            false
        ),

        HwtHw = Logical(2,
            (HwtHw_array, ϕw_index) -> HwtHw_array[:, :, round(Int, ϕw_index)],
            false
        ),

        Hw = Logical(2,
            (Hw_array, ϕw_index) -> Hw_array[:, :, round(Int, ϕw_index)],
            false
        ),

        Rw_inv = Logical(2,
            (Rw_inv_array, ϕw_index) -> Rw_inv_array[:, :, round(Int, ϕw_index)],
            false
        ),

        Rw = Logical(2,
            (Rw_array, ϕw_index) -> Rw_array[:, :, round(Int, ϕw_index)],
            false
        ),

        HwtHw_array = Logical(3,
            (Hw_array) -> array3d_from_mats(
                [(Hw = Hw_array[:, :, i]; transpose(Hw) * Hw) for i in 1:size(Hw_array, 3)]
            ),
            false
        ),

        Hw_array = Logical(3,
            (Ck, ϕw_support, Rw_inv_array) -> array3d_from_mats(
                [exponential(Ck, ϕw_support[i]) * Rw_inv_array[:, :, i] for i in 1:length(ϕw_support)]
            ),
            false
        ),

        Rw_inv_array = Logical(3,
            (Rw_array) -> array3d_from_mats(
                [inv(Rw_array[:, :, i]) for i in 1:size(Rw_array, 3)]
            ),
            false
        ),

        logdetRw_array = Logical(1,
            (Rw_array) -> Float64[
                logdet(Rw_array[:, :, i]) for i in 1:size(Rw_array, 3)
            ],
            false
        ),

        Rw_array = Logical(3,
            (ϕw_support, Dk) -> array3d_from_mats(
                [exponential(Dk, ϕ) for ϕ in ϕw_support]
            ),
            false
        ),

        sumXtX = Logical(2,
            (X, T) -> sum([(x = X[:, :, t]; transpose(x) * x) for t in 1:T]),
            false
        ),

        y_missing_total = Logical(
            (y_missing) -> sum(y_missing),
            false
        ),

        z_missing_total = Logical(
            (z_missing) -> sum(z_missing),
            false
        ),

        n_total = Logical(
            (n, T) -> n * T,
            false
        ),

        nk_total = Logical(
            (nk, T) -> nk * T,
            false
        )
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

        Sampler(:αy, (y, z, τ2y, Xβy, Hηy, αy) ->
            begin
                σ2 = 1 / (sum(abs2, z) / τ2y + 1 / var(αy.distr))
                μ = σ2 * sum(z .* (y - Xβy - Hηy)) / τ2y
                rand(Normal(μ, sqrt(σ2)))
            end
        ),

        Sampler(:βy, (y, zαy, Hηy, βy, X, sumXtX, τ2y, T) ->
            begin
                Σ = inv(sumXtX ./ τ2y + invcov(βy.distr))
                A = y - zαy - Hηy
                μ = Σ * sum([transpose(X[:, :, t]) * A[t, :] for t in 1:T]) ./ τ2y
                rand(MvNormal(μ, Hermitian(Σ)))
            end
        ),

        Sampler(:ηy, (y, Xβy, zαy, ηy, Hy, HytHy, Ry_inv, σ2y, τ2y, T) ->
            begin
                A = y - Xβy - zαy
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

        Sampler(:ϕy_index, (ηy, σ2y, τ2y, y, Xβy, zαy, logdetRy_array, Ry_inv_array, Hy_array, ϕy_support, T) ->
            begin
                logpost = zeros(ϕy_support)
                A = y - Xβy - zαy
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
        ),        
        
        Sampler(:z, (z_mean, z, τ2z, αy, τ2y, y, Xβy, Hηy, z_missing) ->
            begin
                σ2 = 1 / (1 / τ2z + (αy^2) / τ2y)
                μ = σ2 .* (z_mean ./ τ2z + (y - Xβy - Hηy) .* (αy / τ2y))
                for i in eachindex(z, μ, z_missing)
                    if z_missing[i]
                        z[i] = rand(Normal(μ[i], sqrt(σ2)))
                    end
                end
                z
            end
        ),

        Sampler(:τ2z, (z_sse, τ2z, n_total) ->
            begin
                a = n_total / 2 + shape(τ2z.distr)
                b = z_sse / 2 + scale(τ2z.distr)
                rand(InverseGamma(a, b))
            end
        ),

        Sampler(:βz, (z, Hηz, βz, αw, τ2w, X, sumXtX, τ2z, w, Xβw, Hηw, Hηzαw, T) ->
            begin
                Σ = inv(sumXtX .* (1 / τ2z + (αw^2) / τ2w) + invcov(βz.distr))
                A = z - Hηz
                B = w - Xβw - Hηw - Hηzαw
                μ = Σ * sum([transpose(X[:, :, t]) * A[t, :] for t in 1:T]) ./ τ2z
                μ += Σ * sum([transpose(X[:, :, t]) * B[t, :] for t in 1:T]) .* (αw / τ2w)
                rand(MvNormal(μ, Hermitian(Σ)))
            end
        ),

        Sampler(:ηz, (z, Xβz, ηz, αw, τ2w, HztHz, Hz, Rz_inv, σ2z, τ2z, T, w, Xβw, Hηw, Xβzαw) ->
            begin
                A = z - Xβz
                B = w - Xβw - Hηw - Xβzαw
                Σ = inv(HztHz .* (1 / τ2z + (αw^2) / τ2w) + Rz_inv ./ σ2z)
                for t in 1:T
                    μ = Σ * (transpose(Hz) * A[t, :] ./ τ2z)
                    μ += Σ * (transpose(Hz) * B[t, :] .* (αw / τ2w))
                    ηz[t, :] = rand(MvNormal(μ, Hermitian(Σ)))
                end
                ηz
            end
        ),

        Sampler(:σ2z, (σ2z, z_sss, T, nk_total) ->
            begin
                a = nk_total / 2 + shape(σ2z.distr)
                b = z_sss / 2 + scale(σ2z.distr)
                rand(InverseGamma(a, b))
            end
        ),

        Sampler(:ϕz_index, (ηz, σ2z, τ2z, z, Xβz, w, Xβw, Hηw, Xβzαw, αw, τ2w, logdetRz_array, Rz_inv_array, Hz_array, T, ϕz_support) ->
            begin
                logpost = zeros(ϕz_support)
                A = z - Xβz
                B = w - Xβw - Hηw - Xβzαw
                ϕz_index_array = 1:length(ϕz_support)
                for i in ϕz_index_array
                    Hηz = ηz * transpose(Hz_array[:, :, i])
                    logpost[i] = -sum(abs2, A - Hηz) / (2 * τ2z)
                    logpost[i] += -sum(abs2, B - αw .* Hηz) / (2 * τ2w)
                    logpost[i] += -(T / 2) * logdetRz_array[i]
                    for t in 1:T
                        logpost[i] += -(transpose(ηz[t, :]) * Rz_inv_array[:, :, i] * ηz[t, :])[1] / (2 * σ2z)
                    end
                end
                logpost .-= maximum(logpost)
                sample(ϕz_index_array, Weights(exp.(logpost)))
            end
        ),

        Sampler(:w, (w, w_mean, τ2w, z_missing) ->
            begin
                for i in eachindex(w, w_mean, z_missing)
                    if z_missing[i]
                        w[i] = rand(TruncatedNormal(w_mean[i], sqrt(τ2w), 0, Inf))
                    else
                        w[i] = rand(TruncatedNormal(w_mean[i], sqrt(τ2w), -Inf, 0))
                    end
                end
                w
            end
        ),

        Sampler(:βw, (βw, X, sumXtX, w, Xβzαw, Hηzαw, Hηw, τ2w, T) ->
            begin
                Σ = inv(sumXtX ./ τ2w + invcov(βw.distr))
                A = w - Xβzαw - Hηzαw - Hηw
                μ = Σ * sum([transpose(X[:, :, t]) * A[t, :] for t in 1:T]) / τ2w
                rand(MvNormal(μ, Hermitian(Σ)))
            end
        ),

        Sampler(:ηw, (ηw, Hw, HwtHw, Rw_inv, σ2w, w, Xβw, Hηzαw, Xβzαw, τ2w, T) ->
            begin
                A = w - Xβw - Hηzαw - Xβzαw
                Σ = inv(HwtHw ./ τ2w + Rw_inv ./ σ2w)
                for t in 1:T
                    μ = Σ * ((transpose(Hw) * A[t, :]) ./ τ2w)
                    ηw[t, :] = rand(MvNormal(μ, Hermitian(Σ)))
                end
                ηw
            end
        ),

        Sampler(:αw, (αw, z_mean, τ2w, w, Xβw, Hηw, T) ->
            begin
                σ2 = 1 / (sum(abs2, z_mean) / τ2w + 1 / var(αw.distr))
                μ = σ2 * sum(z_mean .* (w - Xβw - Hηw)) / τ2w
                rand(Normal(μ, sqrt(σ2)))
            end
        ),

        AMWG(:σ2w, 0.05, adapt=:burnin),

        Sampler(:ϕw_index, (ηw, σ2w, τ2w, w, Xβw, Xβzαw, Hηzαw, logdetRw_array, Rw_inv_array, Hw_array, T, ϕw_support) ->
            begin
                logpost = zeros(ϕw_support)
                A = w - Xβw - Xβzαw - Hηzαw
                ϕw_support_index = 1:length(ϕw_support)
                for i in ϕw_support_index
                    logpost[i] = -sum(abs2, A - ηw * transpose(Hw_array[:, :, i])) / (2 * τ2w)
                    logpost[i] += -(T / 2) * logdetRw_array[i]
                    for t in 1:T
                        logpost[i] += -(transpose(ηw[t, :]) * Rw_inv_array[:, :, i] * ηw[t, :])[1] / (2  * σ2w)
                    end
                end
                logpost .-= maximum(logpost)
                sample(ϕw_support_index, Weights(exp.(logpost)))
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
    z_init = copy(data[:z])
    z_mean = nanmean(z_init)
    for i in eachindex(z_init)
        if isnan(z_init[i])
            z_init[i] = z_mean
        end
    end
    w_init = zeros(data[:T], data[:n])
    z_missing = data[:z_missing]
    for i in eachindex(w_init, z_missing)
        w_init[i] = z_missing[i] ? 0.5 : -0.5
    end
    
    βy_init = params[:βy] == "zeros" ? zeros(data[:p]) : params[:βy]
    βz_init = params[:βz] == "zeros" ? zeros(data[:p]) : params[:βz]
    βw_init = params[:βw] == "zeros" ? zeros(data[:p]) : params[:βw]

    inits = Dict{Symbol, Any}(
        :y => y_init,
        :ηy => zeros(data[:T], data[:nk]),
        :βy => βy_init,
        :αy => params[:αy],
        :σ2y => params[:σ2y],
        :τ2y => params[:τ2y],
        :ϕy_index => params[:ϕy]["index"],
        :z => z_init,
        :ηz => zeros(data[:T], data[:nk]),
        :βz => βz_init,
        :σ2z => params[:σ2z],
        :τ2z => params[:τ2z],
        :ϕz_index => params[:ϕz]["index"],
        :w => w_init,
        :ηw => zeros(data[:T], data[:nk]),
        :βw => βw_init,
        :αw => params[:αw],
        :σ2w => params[:σ2w],
        :ϕw_index => params[:ϕw]["index"]
    )
    return inits
end

