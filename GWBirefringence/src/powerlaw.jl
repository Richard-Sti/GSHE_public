import MultivariateStats: llsq

"""
    average(x::Vector{GWFloat})

Calculate the average of a vector `x`.
"""
function average(x::Vector{GWFloat})
    return sum(x) / length(x)
end


"""
    std(x::Vector{GWFloat})

Calculate the standard deviation of a vector `x`.
"""
function std(x::Vector{GWFloat})
    mu = average(x)
    return average((x .- mu).^2)^0.5
end


"""
    bootstrap_log_llsq(
        x::Vector{GWFloat},
        y::Vector{GWFloat};
        Nboots::Int64=1000
    )

Calculate the bootstrap mean and standard deviation on the function
:math:`f(x) = α x^β`. The output rows give, respectively, the mean and standard
deviation of α and β.
"""
function bootstrap_powerlaw(
    x::Vector{GWFloat},
    y::Vector{GWFloat};
    Nboots::Int64=1000
)
    x = log10.(x)
    y = log10.(y)
    @assert length(x) == length(y) "`x` and `y` must have equal length."
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
    fit_Δts(ϵs::Vector{GWFloat}, Xspinhall::Array{GWFloat, 4})

Fit the spin-Hall s=±2 time of arrival differences as a function of ϵ. For each
geodesic. Array indices represent first the geodesic, second mean value of
α and β and third their standard deviatations.
"""
function fit_Δts(ϵs::Vector{GWFloat}, Xspinhall::Array{GWFloat, 4})
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