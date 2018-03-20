using Iterators: product
using Distances: Metric, Euclidean, pairwise

function expandgrid(x::Range, y::Range)
    matrix_from_cols([collect(xy) for xy in product(x, y)])
end

expandgrid(x::Range) = expandgrid(x, x)

function exponential(D, ϕ)
    R = zeros(D)
    for i in eachindex(D)
        R[i] = exp(-D[i] / ϕ)
    end
    R
end

function matrix_from_rows{T}(xs::Vector{Vector{T}})
    mat = Matrix{T}(length(xs), length(first(xs)))
    for (i, row) in enumerate(xs)
        @inbounds mat[i, :] = row
    end
    mat
end
matrix_from_rows{T}(Xs::Vector{Matrix{T}}) = matrix_from_rows([vec(X) for X in Xs])

function matrix_from_cols{T}(xs::Vector{Vector{T}})
    mat = Matrix{T}(length(first(xs)), length(xs))
    for (i, col) in enumerate(xs)
        @inbounds mat[:, i] = col
    end
    mat
end
matrix_from_cols{T}(Xs::Vector{Matrix{T}}) = matrix_from_cols([vec(X) for X in Xs])

function array3d_from_mats{T}(Xs::Vector{Matrix{T}})
    n, m = size(first(Xs))
    arr = Array{T, 3}(n, m, length(Xs))
    for (i, mat) in enumerate(Xs)
        @inbounds arr[:, :, i] = mat
    end
    arr
end

function nansum{T}(A::Array{T})
    s = zero(T)
    for a in A
        if !isnan(a)
            s += a
        end
    end
    s
end

function nanmean{T}(A::Array{T})
    s = 0.0
    n = 0
    for a in A
        if !isnan(a)
            s += a
            n += 1
        end
    end
    s / n
end

function dimmin(A::Matrix, dim::Int)
    inds = vec(findmin(A, dim)[2])
    [ind2sub(size(A), ind)[dim] for ind in inds]
end

function writemc(filename::String, mc::ModelChains; names=mc.names)
    sim = mc[:, names, :]
    vals = sim.value
    vals = vcat([vals[:, :, i] for i in 1:size(vals, 3)]...)
    df = DataFrame([names, [vals[i, :] for i in 1:size(vals, 1)]...],
                   [:name; Symbol["_$i" for i in 1:size(vals, 1)]...])
    writetable(filename, df)
end

function overlaypoints(S::Matrix{Float64}; metric::Metric=Euclidean,
                       ticks::Int64=10, margin::Float64=5e-2, maxdist::Real=1e-1)
    x = linspace(minimum(S[1, :]) + margin, maximum(S[1, :]) - margin, ticks)
    y = linspace(minimum(S[2, :]) + margin, maximum(S[2, :]) - margin, ticks)
    Sk = expandgrid(x, y)
    within_maxdist = vec(any(pairwise(metric, S, Sk) .< maxdist, 1))
    any(within_maxdist) || error("No new points fall within max allowable distance from points in S")
    Sk[:, within_maxdist]
end


function generate_data(;
        nm::Int=100,
        T::Int=10,
        prop_obs_per_timepoint::Float64=0.5,
        metric::Metric=Euclidean(),
        S::Matrix{Float64}=expandgrid(0.0:0.05:1.0),
        Sm::Matrix{Float64}=
            begin
                d = Uniform(minimum(S), maximum(S))
                transpose(hcat(rand(d, nm), rand(d, nm)))
            end,
        Sk::Matrix{Float64}=expandgrid(0.0625:0.125:0.9375),
        Xm::Array{Float64,3}=array3d_from_mats([ones(nm, 1) for t in 1:T]),
        X::Array{Float64,3}=array3d_from_mats([ones(size(S, 2), 1) for t in 1:T]),
        βy::Vector{Float64}=zeros(size(X, 2)),
        βz::Vector{Float64}=zeros(size(X, 2)),
        βw::Vector{Float64}=zeros(size(X, 2)),
        αy::Float64=0.0,
        αw::Float64=0.0,
        σ2y::Float64=1.0,
        σ2z::Float64=1.0,
        σ2w::Float64=0.5,
        τ2y::Float64=1.0,
        τ2z::Float64=1.0,
        ϕy::Float64=1.0,
        ϕz::Float64=1.0,
        ϕw::Float64=1.0,
        misspecify::Bool=false)
    n = size(X, 1)
    p = size(X, 2)
    @assert 0.0 <= prop_obs_per_timepoint <= 1.0
    @assert size(Xm, 3) == T
    @assert size(X, 3) == T
    @assert nm == size(Sm, 2)
    @assert n == size(S, 2)
    @assert size(Xm, 2) == p
    @assert length(βy) == p
    @assert length(βz) == p
    @assert length(βw) == p

    # generate spatial covariate
    D = pairwise(metric, S)
    nearest_cell = dimmin(pairwise(metric, S, Sm), 1)
    Xβz = matrix_from_rows([X[:, :, t] * βz for t in 1:T])
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
        if rand() > prop_obs_per_timepoint
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
