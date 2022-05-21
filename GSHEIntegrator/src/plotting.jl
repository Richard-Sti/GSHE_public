"""
    plot_initial_conditions!(
        fig::Plots.Plot,
        Xgshe::Array{<:Real, 4},
        Xgeo::Matrix{<:Real},
        n::Int64
    )

Plot the initial conditions ψ / π and ρ / π.
"""
function plot_initial_conditions!(
    fig::Plots.Plot,
    Xgshe::Array{<:Real, 4},
    Xgeo::Matrix{<:Real},
    n::Int64
)
    for s in 1:2
        Xgshe[n, s, :, 1:2]
        Plots.scatter!(fig, Xgshe[n, s, :, 2]./π, Xgshe[n, s, :, 1]./π, label="s=$s")
    end
    Plots.scatter!(fig, [Xgeo[n, 2]]/π, [Xgeo[n, 1]]/π, label="Geodesic")
end


"""
    plot_arrival_times!(
        fig::Plots.Plot,
        ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
        Xgshe::Array{<:Real, 4},
        Xgeo::Matrix{<:Real},
        geometry::Geometry
    )

Plot the difference between the GSHE and geodesic time of arrival.
"""
function plot_arrival_times!(
    fig::Plots.Plot,
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
    Xgshe::Array{<:Real, 4},
    Xgeo::Matrix{<:Real},
    geometry::Geometry
)
    Ngeos = size(Xgeo)[1]
    pars = fit_Δts(ϵs, Xgshe, Xgeo, geometry)

    for n in 1:Ngeos, s in 1:2
        α = round(pars[n][s]["alpha"][1]; digits=3)
        β = round(pars[n][s]["beta"][1]; digits=3)
        dt = abs.(Xgshe[n, s, :, 3] .- Xgeo[n, 3])
        m = dt .> 0
        label = L"n = %$n, s_i = %$s; y(\epsilon) = %$β~\epsilon^{%$α}"
        Plots.scatter!(fig, ϵs[m], dt[m], label=label)
    end
    return fig
end


"""
    plot_time_difference!(
        fig::Plots.Plot,
        ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
        Xspinhall::Array{<:Real, 4},
        Xgeo::Matrix{<:Real}

    )

Plot the s = ± 2 arrival time difference.
"""
function plot_time_difference!(
    fig::Plots.Plot,
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
    Xgshe::Array{<:Real, 4},
    geometry::Geometry
)
    Ngeos = size(Xgshe)[1]


    pars = fit_Δts(ϵs, Xgshe, geometry)

    for n in 1:Ngeos
        α = round(pars[n]["alpha"][1]; digits=3)
        β = round(pars[n]["beta"][1]; digits=3)
        dt = abs.(Xgshe[n, 1, :, 3] - Xgshe[n, 2, :, 3])
        m = dt .> 0
        label = L"n = %$n; y(\epsilon) = %$β~\epsilon^{%$α}"
        Plots.scatter!(fig, ϵs[m], dt[m], label=label)
    end
    return fig
end


"""
    cartesian_trajectory(
        init_direction::Vector{<:Real},
        geometry::Geometry,
        is_geodesic::Bool,
        ϵ::Real=0.0,
        s::Integer=0
    )

Integrates a trajectory and returns its Cartesian representation.
"""
function cartesian_trajectory(
    init_direction::Vector{<:Real},
    geometry::Geometry,
    is_geodesic::Bool,
    ϵ::Real=0.0,
    s::Integer=0
)
    if is_geodesic
        sol = solve_geodesic(init_direction, geometry, save_everystep=true)
    else
        sol = solve_gshe(init_direction, geometry, ϵ, s, save_everystep=true)
    end
    g(x::Vector{<:Real}) = spherical_to_cartesian(x, geometry.a)
    return mapslices(g, sol[2:4, :], dims=1)
end


"""
    plot_geodesics!(fig::Plots.Plot, Xgeo::Matrix{<:Real}, geometry::Geometry)

Plot the geodesic solutions.
"""
function plot_geodesics!(fig::Plots.Plot, Xgeo::Matrix{<:Real}, geometry::Geometry)
    Ngeos = size(Xgeo)[1]
    for n in 1:Ngeos
        X = cartesian_trajectory(Xgeo[n, 1:2], geometry, true)
        Plots.plot!(fig, X[1,:], X[2, :], X[3, :];label=nothing, lw=1, ls=:dash)
    end
    return fig
end


"""
    plot_gshe_trajectories(
        fig::Plots.Plot,
        Xgshe::Array{<:Real, 4},
        ϵs::Vector{<:Real},
        geometry::Geometry{<:Real}
    )

Plot the GSHE solutions.
"""
function plot_gshe_trajectories!(
    fig::Plots.Plot,
    Xgshe::Array{<:Real, 4},
    ϵs::Vector{<:Real},
    geometry::Geometry{<:Real}
)
    s = geometry.s
    Ngeos = size(Xgshe)[1]
    Nϵs = length(ϵs)

    cols = Plots.palette(:rainbow, Nϵs)
    for n in 1:Ngeos, sx in 1:2, i in 1:Nϵs
        if any(isnan.(Xgshe[n, sx, i, :]))
            continue
        end
        X = cartesian_trajectory(Xgshe[n, sx, i, 1:2], geometry, false,
                                 ϵs[i], sx == 1 ? s : -s)
        Plots.plot!(fig, X[1,:], X[2, :], X[3, :], label=nothing, color=cols[i],lw=0.5)
    end

    return fig

end


"""
    plot_blackhole!(
        fig::Plots.Plot,
        loc::Tuple{Float64, Float64, Float64}=(0.0, 0.0, 0.0),
        radius::Float64=0.0
    )

Plot the black hole. If radius = 0 use as a point.
"""
function plot_blackhole!(
    fig::Plots.Plot,
    loc::Tuple{Float64, Float64, Float64}=(0.0, 0.0, 0.0),
    radius::Float64=0.0
)
    if radius == 0.0
        Plots.scatter!(fig, loc, c="black", label="Black hole")
    else
        s = Meshes.Sphere(loc, radius)
        Plots.scatter!(fig, s, alpha=0.01, label="Black hole")
    end
    return fig
end


"""
    plot_start_end!(fig::Plots.Plot, geometry::Geometry)

Plot the source and observer.
"""
function plot_start_end!(fig::Plots.Plot, geometry::Geometry)
    Xsource = [geometry.source.r, geometry.source.θ, geometry.source.ϕ]
    spherical_to_cartesian!(Xsource, geometry.a)
    Plots.scatter!(fig, [Xsource[1]], [Xsource[2]], [Xsource[3]], color="red", label="Source")

    Xobs = [geometry.observer.r, geometry.observer.θ, geometry.observer.ϕ]
    spherical_to_cartesian!(Xobs, geometry.a)
    Plots.scatter!(fig, [Xobs[1]], [Xobs[2]], [Xobs[3]], color="blue", label="Observer")
    return fig
end