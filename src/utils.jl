
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

function matrix_from_rows(xs::Vector{Vector{T}}) where T
    mat = Matrix{T}(length(xs), length(first(xs)))
    for (i, row) in enumerate(xs)
        @inbounds mat[i, :] = row
    end
    mat
end
matrix_from_rows(Xs::Vector{Matrix{T}}) where T = matrix_from_rows([vec(X) for X in Xs])

function matrix_from_cols(xs::Vector{Vector{T}}) where T
    mat = Matrix{T}(length(first(xs)), length(xs))
    for (i, col) in enumerate(xs)
        @inbounds mat[:, i] = col
    end
    mat
end
matrix_from_cols(Xs::Vector{Matrix{T}}) where T = matrix_from_cols([vec(X) for X in Xs])

function array3d_from_mats(Xs::Vector{Matrix{T}}) where T
    n, m = size(first(Xs))
    arr = Array{T, 3}(n, m, length(Xs))
    for (i, mat) in enumerate(Xs)
        @inbounds arr[:, :, i] = mat
    end
    arr
end

function nansum(A::Array{T}) where T <: Real
    s = zero(T)
    for a in A
        if !isnan(a)
            s += a
        end
    end
    s
end

function nanmean(A::Array{T}) where T <: Real
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

function overlaypoints(S::Matrix{Float64}; metric::Metric=Euclidean,
                       ticks::Int64=10, margin::Float64=5e-2, maxdist::Real=1e-1)
    x = linspace(minimum(S[1, :]) + margin, maximum(S[1, :]) - margin, ticks)
    y = linspace(minimum(S[2, :]) + margin, maximum(S[2, :]) - margin, ticks)
    Sk = expandgrid(x, y)
    within_maxdist = vec(any(pairwise(metric, S, Sk) .< maxdist, 1))
    any(within_maxdist) || error("No new points fall within max allowable distance from points in S")
    Sk[:, within_maxdist]
end
