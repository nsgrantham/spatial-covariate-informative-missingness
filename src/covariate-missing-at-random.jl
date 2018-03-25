# Spatial regression model that includes a spatially-missing covariate
# under the assumption that its values are missing-at-random

function get_model(::Type{MissingAtRandom}, monitor::Dict{Symbol, Any}, hyper::Dict{Symbol, Any})
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

        Sampler(:z, (z_mean, z, τ2z, αy, τ2y, y, Xβy, Hηy, ϕz, T, z_missing) ->
            begin
                Σ = 1 / (1 / τ2z + (αy^2) / τ2y)
                μ = Σ .* (z_mean ./ τ2z + (y - Xβy - Hηy) .* (αy / τ2y))
                for i in eachindex(z, μ, z_missing)
                    if z_missing[i]
                        z[i] = rand(Normal(μ[i], sqrt(Σ)))
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

        Sampler(:βz, (z, Hηz, βz, X, sumXtX, τ2z, T) ->
            begin
                Σ = inv(sumXtX ./ τ2z + invcov(βz.distr))
                A = z - Hηz
                μ = Σ * sum([transpose(X[:, :, t]) * A[t, :] for t in 1:T]) ./ τ2z
                rand(MvNormal(μ, Hermitian(Σ)))
            end
        ),

        Sampler(:ηz, (z, Xβz, ηz, Hz, HztHz, Rz_inv, σ2z, τ2z, T) ->
            begin
                A = z - Xβz
                Σ = inv(HztHz ./ τ2z + Rz_inv ./ σ2z)
                for t in 1:T
                    μ = Σ * (transpose(Hz) * A[t, :] ./ τ2z)
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

        Sampler(:ϕz_index, (ηz, σ2z, τ2z, z, Xβz, logdetRz_array, Rz_inv_array, Hz_array, ϕz_support, T) ->
            begin
                logpost = zeros(ϕz_support)
                A = z - Xβz
                ϕz_index_array = 1:length(ϕz_support)
                for i in ϕz_index_array
                    Hηz = ηz * transpose(Hz_array[:, :, i])
                    logpost[i] = -sum(abs2, A - Hηz) / (2 * τ2z)
                    logpost[i] += -(T / 2) * logdetRz_array[i]
                    for t in 1:T
                        logpost[i] += -(transpose(ηz[t, :]) * Rz_inv_array[:, :, i] * ηz[t, :])[1] / (2 * σ2z)
                    end
                end
                logpost .-= maximum(logpost)
                sample(ϕz_index_array, Weights(exp.(logpost)))
            end
        )
    ]
    
    setsamplers!(model, samplers)
    return model
end

function get_inits(::Type{MissingAtRandom}, params::Dict{Symbol, Any}, data::Dict{Symbol, Any})
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

    βy_init = params[:βy] == "zeros" ? zeros(data[:p]) : params[:βy]
    βz_init = params[:βz] == "zeros" ? zeros(data[:p]) : params[:βz]

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
        :ϕz_index => params[:ϕz]["index"]
    )
    return inits
end
