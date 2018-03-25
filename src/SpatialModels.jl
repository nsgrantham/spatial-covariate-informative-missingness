module SpatialModels

using Mamba
using Distributions

using StatsBase: Weights
using Iterators: product
using Distances: Metric, pairwise

export
    NotAvailable,
    MissingAtRandom,
    MissingNotAtRandom,
    get_model,
    get_inits,
    expandgrid,
    exponential,
    matrix_from_rows,
    matrix_from_cols,
    array3d_from_mats,
    nansum,
    nanmean,
    dimmin,
    overlaypoints

abstract type SpatialCovariateMissingness end
struct NotAvailable <: SpatialCovariateMissingness end
struct MissingAtRandom <: SpatialCovariateMissingness end
struct MissingNotAtRandom <: SpatialCovariateMissingness end

include("covariate-not-available.jl")
include("covariate-missing-at-random.jl")
include("covariate-missing-not-at-random.jl")
include("utils.jl")

end  # module
