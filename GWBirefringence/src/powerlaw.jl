"""
    average(x::Vector{<:Real})

Calculate the average of a vector `x`.
"""
function average(x::Vector{<:Real})
    return sum(x) / length(x)
end


"""
    std(x::Vector{<:Real})

Calculate the standard deviation of a vector `x`.
"""
function std(x::Vector{<:Real})
    mu = average(x)
    return average((x .- mu).^2)^0.5
end


"""
    bootstrap_powerlaw(
        x::Vector{<:Real},
        y::Vector{<:Real}; 
        Nboots::Int64=1000,
        integration_error::Real=1e-12
    )

Calculate the bootstrap mean and standard deviation on the function :math:`f(x) = α x^β`.
The output rows give, respectively, the mean and standard deviation of α and β.
"""
function bootstrap_powerlaw(
    x::Vector{<:Real},
    y::Vector{<:Real}; 
    Nboots::Int64=1000,
    integration_error::Real=1e-12
)
    # Initial checks
    @assert length(x) == length(y) "`x` and `y` must have equal length."
    x, y = cut_below_integration_error(x, y, integration_error)

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

    stats = zeros(2, 2)
    stats[1, :] .= average(res[:, 1]), std(res[:, 1])
    stats[2, :] .= average(res[:, 2]), std(res[:, 2])
    # Exponeniate the intercept
    stats[2, 1] = 10^stats[2, 1]
    # Propagate the error -- CHECK THIS
    stats[2, 2] = stats[2,2] * stats[2, 1] * log(10)
    return stats
end


"""
    cut_below_integration_error(
        x::Vector{<:Real},
        y::Vector{<:Real},
        integration_error::Real=1e-12
    )

Remove elements from `x` and `y` satisfying |`y`| < `integration_error` and return the
(new) `x` and and `y`.
"""
function cut_below_integration_error(
    x::Vector{<:Real},
    y::Vector{<:Real},
    integration_error::Real=1e-12
)
    mask = abs.(y) .> integration_error
    N = sum(.~mask)
    if N > 0
        @info "$N element(s) below integration error $integration_error. Removing."
    end
    return x[mask], y[mask]
end


"""
    fit_Δts(ϵs::Vector{<:Real}, Xspinhall::Array{<:Real, 4})

Fit the spin-Hall s=±2 time of arrival differences as a function of ϵ for each geodesic.
Array indices represent first the geodesic, mean value of α and β, and third their
standard deviatations, respectively.
"""
function fit_Δts(ϵs::Vector{<:Real}, Xspinhall::Array{<:Real, 4})
    Ngeo = size(Xspinhall)[3]
    Δt = zeros(size(Xspinhall)[1])

    X = zeros(Ngeo, 2, 2)
    for igeo in 1:Ngeo
        xplus = @view Xspinhall[:, 1, igeo, 3]
        xminus = @view Xspinhall[:, 2, igeo, 3]
        Δt .= abs.(xplus - xminus)
        X[igeo, :, :] .= bootstrap_powerlaw(ϵs, Δt)
    end
    return X
end


"""
    fit_Δts(ϵs::Vector{<:Real}, Xgeo::Matrix{<:Real}, Xspinhall::Array{<:Real, 4})

Fit the time of arrival difference between the spin Hall solution and the geodesic as a
function of ϵ for s = ± 2. Array indices represent the geodesic, polarisation, mean value
of α and β, and third their standard deviatations, respectively.
"""
function fit_Δts(ϵs::Vector{<:Real}, Xgeo::Matrix{<:Real}, Xspinhall::Array{<:Real, 4})
    Ngeo = size(Xspinhall)[3]
    Δt = zeros(size(Xspinhall)[1])

    X = zeros(Ngeo, 2, 2, 2)
    for igeo in 1:Ngeo, s in [2, -2]
        Δt .= abs.(Xspinhall[:, s === 2 ? 1 : 2, igeo, 3] .- Xgeo[igeo, 3])
        X[igeo, s === 2 ? 1 : 2, :, :] .= bootstrap_powerlaw(ϵs, Δt)
    end
    return X
end