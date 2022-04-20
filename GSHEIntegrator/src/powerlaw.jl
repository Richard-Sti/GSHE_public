"""
    average(x::Vector{<:Real})

Calculate the average of a 1-dimensional vector.
"""
function average(x::Vector{<:Real})
    return sum(x) / length(x)
end


"""
    std(x::Vector{<:Real})

Calculate the standard deviation of a 1-dimensional vector.
"""
function std(x::Vector{<:Real})
    mu = average(x)
    return sqrt(average((x .- mu).^2))
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
        res[i, :] .= llsq(x[mask], y[mask])
    end

    out = Dict("alpha" => [average(res[:, 1]), std(res[:, 1])],
               "beta" => [average(res[:, 2]), std(res[:, 2])])

    # Exponenitate the intercept
    out["beta"][1] = 10^out["beta"][1]
    # TODO: check the error propagation
    out["beta"][2] = out["beta"][2] * out["beta"][1] * log(10)

    return out
end


"""
    cut_below_integration_error(
        x::Vector{<:Real},
        y::Vector{<:Real},
        integration_error::Real=1e-12
    )

Remove elements from x and y satisfying |y| < integration_error and return the
(new) x and and y.
"""
function cut_below_integration_error(
    x::Vector{<:Real},
    y::Vector{<:Real},
    integration_error::Real=1e-12
)
    mask = abs.(y) .> integration_error
    N = sum(.~mask)
    if N > 0
        @info "$N element(s) x=$(x[.~mask]) y=$(y[.~mask]) below integration error $integration_error. Removing."
    end
    return x[mask], y[mask]
end


"""
    fit_Δts(ϵs::Vector{<:Real}, Xspinhall::Array{<:Real, 4})


Fit a power law to the difference between the circular polarisation GSHE time of arrival as
a function of ϵ for each geodesic. Returns a vector whose elements are the fits to each
original geodesic.
"""
function fit_Δts(ϵs::Vector{<:Real}, Xgshe::Array{<:Real, 4}, geometry::Geometry)
    @unpack Nboots, integration_error, minpoints = geometry.postproc_options
    Ngeos = size(Xgshe)[1]
    Δt = zeros(size(Xgshe)[3])

    X = Any[]
    for n in 1:Ngeos
        Δt .= abs.(Xgshe[n, 1, :, 3] - Xgshe[n, 2, :, 3])
        push!(X, bootstrap_powerlaw(ϵs, Δt, Nboots, integration_error, minpoints))
    end

    return X
end


"""
    fit_Δts(ϵs::Vector{<:Real}, Xgeo::Matrix{<:Real}, Xspinhall::Array{<:Real, 4})

Fit a power law to the difference between the circular GSHE time of arrival and a geodesic
time of arrival. Retuns a nested vector, the outer elements are for the corresponding
geodesics and the inner elements for the ± s polarisation states.
"""
function fit_Δts(
    ϵs::Vector{<:Real},
    Xgshe::Array{<:Real, 4},
    Xgeo::Matrix{<:Real},
    geometry::Geometry
)
    Ngeos = size(Xgeo)[1]
    @assert Ngeos == size(Xgshe)[1]
    @unpack Nboots, integration_error, minpoints = geometry.postproc_options

    X = Any[]
    Δt = zeros(size(Xgshe)[3])
    for n in 1:Ngeos
        Xs = Any[]
        for s in 1:2
            Δt .= abs.(Xgshe[n, s, :, 3] .- Xgeo[n, 3])
            push!(Xs, bootstrap_powerlaw(ϵs, Δt, Nboots, integration_error, minpoints))
        end
        push!(X, Xs)
    end
    return X
end
