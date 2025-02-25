"""
    function linear_llsq(x::Vector{<:Real}, y::Vector{<:Real})

Calculate the linear least squares fit of y = α x + β.
"""
function linear_llsq(x::Vector{<:Real}, y::Vector{<:Real})
    N = length(x)
    sx, sy, sx2 = sum(x), sum(y), sum(x.^2)
    sxy = sum(x .* y)

    norm = (sx^2 - N * sx2)
    α = (sx * sy - N * sxy) / norm
    β = (10).^((sx * sxy - sx2 * sy) / norm)
    return α, β
end


"""
    bootstrap_powerlaw(
        x::Vector{<:Real},
        y::Vector{<:Real};
        Nboots::Integer=1000,
        integration_error::Real=1e-12
    )

Calculate the bootstrap mean and standard deviation on the function f(x) = β x^α.
"""
function bootstrap_powerlaw(
    x::Vector{<:Real},
    y::Vector{<:Real},
    Nboots::Integer=1000,
    integration_error::Real=1e-12,
    minpoints::Integer=6
)
    # Initial checks
    @assert length(x) == length(y) "`x` and `y` must have equal length."
    x, y = cut_below_integration_error(x, y, integration_error)

    if length(x) < minpoints
        return Dict("alpha" => [NaN, NaN],
                    "beta" => [NaN, NaN])
    end


    x .= log10.(x)
    y .= log10.(y)
    N = length(x)
    # Preallocate the scratch arrays
    res = zeros(Nboots, 2)
    mask = zeros(Int64, N)

    for i in 1:Nboots
        mask .= rand(1:N, N)
        res[i, :] .= linear_llsq(x[mask], y[mask])
    end

    out = Dict("alpha" => [mean(res[:, 1]), std(res[:, 1])],
               "beta" => [mean(res[:, 2]), std(res[:, 2])])

    # Exponenitate the intercept
    out["beta"][1] = 10^out["beta"][1]
    # And propagate the uncertainty
    out["beta"][2] = log(10) * out["beta"][1] * out["beta"][2]
    return out
end


"""
    cut_below_integration_error(
        x::Vector{<:Real},
        y::Vector{<:Real},
        integration_error::Real=1e-12;
        return_mask::Bool=false,
        verbose::Bool=true
    )

Remove elements from x and y satisfying |y| < integration_error and return the
(new) x and and y.
"""
function cut_below_integration_error(
    x::Vector{<:Real},
    y::Vector{<:Real},
    integration_error::Real=1e-12;
    mask_only::Bool=false,
    verbose::Bool=true
)
    mask = abs.(y) .> integration_error
    N = sum(.~mask)
    if verbose && N > 0
        @info "$N element(s) x=$(x[.~mask]) y=$(y[.~mask]) below integration error $integration_error. Removing."
    end
    if mask_only
        return mask
    end
        return x[mask], y[mask]
end


"""
    fit_Δts(ϵs::Vector{<:Real}, Xgshe::Array{<:Real, 3}, geometry::Geometry)

Fit a power law to the difference between the GSHE time of arrivals as a function of ϵ for
a specific geodesic.
"""
function fit_Δts(ϵs::Vector{<:Real}, Xgshe::Array{<:Real, 3}, geometry::Geometry)
    @unpack Nboots, integration_error, minpoints = geometry.postproc_options
    Δt = abs.(Xgshe[1, :, 3] - Xgshe[2, :, 3])
    return bootstrap_powerlaw(ϵs, Δt, Nboots, integration_error, minpoints)
end


"""
    fit_Δts(ϵs::Vector{<:Real}, Xgshe::Array{<:Real, 4}, geometry::Geometry)

Fit a power law to the difference between the GSHE time of arrivals as a function of ϵ for
geodesics. Returns a vector of fits.
"""
function fit_Δts(ϵs::Vector{<:Real}, Xgshe::Array{<:Real, 4}, geometry::Geometry)
    Ngeos = size(Xgshe, 1)

    X = Any[]
    for n in 1:Ngeos
        push!(X, fit_Δts(ϵs, Xgshe[n, ..], geometry))
    end

    return X
end


"""
    fit_Δts(
        ϵs::Vector{<:Real},
        Xgshe::Matrix{<:Real, 2},
        Xgeo::Vector{<:Real},
        geometry::Geometry
    )

Fit a power law to the difference between the GSHE of a given polarisation and a geodesic time
of arrival of a specific geodesic.
"""
function fit_Δts(
    ϵs::Vector{<:Real},
    Xgshe::Matrix{<:Real},
    Xgeo::Vector{<:Real},
    geometry::Geometry
)
    @unpack Nboots, integration_error, minpoints = geometry.postproc_options
    Δt = abs.(Xgshe[:, 3] .- Xgeo[3])
    return bootstrap_powerlaw(ϵs, Δt, Nboots, integration_error, minpoints)
end


"""
    fit_Δts(
        ϵs::Vector{<:Real},
        Xgshe::Array{<:Real, 3},
        Xgeo::Vector{<:Real},
        geometry::Geometry
    )

Fit a power law to the difference between the GSHE and a geodesic time of arrival of a
specific geodesic. The vector elements correspond to the fits for the ± |s| polarisation
states.
"""
function fit_Δts(
    ϵs::Vector{<:Real},
    Xgshe::Array{<:Real, 3},
    Xgeo::Vector{<:Real},
    geometry::Geometry
)
    return [fit_Δts(ϵs, Xgshe[s, ..], Xgeo, geometry) for s in 1:2]
end


"""
    function fit_Δts(
        ϵs::Vector{<:Real},
        Xgshe::Array{<:Real, 4},
        Xgeo::Matrix{<:Real},
        geometry::Geometry
    )

Fit a power law to the difference between the GSHE and a geodesic time of arrival. Returns
a nested vector, the outer elements are for the corresponding geodesics and the inner
elements for the ± |s| polarisation states.
"""
function fit_Δts(
    ϵs::Vector{<:Real},
    Xgshe::Array{<:Real, 4},
    Xgeo::Matrix{<:Real},
    geometry::Geometry
)
    Ngeos = size(Xgeo)[1]
    @assert Ngeos == size(Xgshe)[1]

    X = Any[]
    for n in 1:Ngeos
        push!(X, fit_Δts(ϵs, Xgshe[n, ..], Xgeo[n, :], geometry))
    end
    return X
end
