"""
    plot_initial_conditions(
        Xspinhall::Array{<:Real, 4},
        Xgeo::Matrix{<:Real},
        igeo::Int64;
        dpi::Int64=600
    )

Plot the initial conditions ψ and ρ.
"""
function plot_initial_conditions!(
    fig::Plots.Plot,
    Xspinhall::Array{<:Real, 4},
    Xgeo::Matrix{<:Real},
    igeo::Int64
)
    for s in [2, -2]
        x = @view Xspinhall[:, s == 2 ? 1 : 2, igeo, 1:2]
        Plots.scatter!(fig, x[:, 2]/π, x[:, 1]/π, label="s=$s")
    end
    Plots.scatter!(fig, [Xgeo[igeo, 2]]/π, [Xgeo[igeo, 1]]/π, label="Geodesic")
end


"""
    plot_arrival_times!(
        fig::Plots.Plot,
        ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
        Xspinhall::Array{<:Real, 4},
        Xgeo::Matrix{<:Real},
        igeo::Int64
    )

Plot the arrival time of the geodesic and the spin-Hall perturbations.
"""
function plot_arrival_times!(
    fig::Plots.Plot,
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
    Xspinhall::Array{<:Real, 4},
    Xgeo::Matrix{<:Real},
    igeo::Int64
)   

    pars = fit_Δts(ϵs, Xgeo, Xspinhall)

    xplus = @view Xspinhall[:, 1, igeo, 3]
    xminus = @view Xspinhall[:, 2, igeo, 3]

    t = Xgeo[igeo, 3]
    β, α = round.(pars[1, 1, :, 1]; digits=3)
    dt = abs.(xplus .- t)
    m = dt .> 0
    Plots.scatter!(fig, ϵs[m], dt[m], label=L"s=2; y(\epsilon) = %$α~\epsilon^{%$β}")
    β, α = round.(pars[1, 2, :, 1]; digits=3)
    dt = abs.(xminus .- t)
    m = dt .> 0
    Plots.scatter!(fig, ϵs[m], dt[m], label=L"s=-2; y(\epsilon) = %$α~\epsilon^{%$β}")
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
    Xspinhall::Array{<:Real, 4},
    Xgeo::Matrix{<:Real}
)
    Nsols = size(Xgeo)[1]
    for igeo in 1:Nsols
        pars = fit_Δts(ϵs, Xspinhall)[igeo, :, 1]
        β, α = round.(pars; digits=5)

        dt = abs.(Xspinhall[:, 1, igeo, 3] - Xspinhall[:, 2, igeo, 3])
        m = dt .> 0
        Plots.scatter!(fig, ϵs[m], dt[m], label=L"\mathrm{Geodesic}~%$igeo; y(\epsilon) = %$α~\epsilon^{%$β}")
    end
end


"""
    cartesian_trajectory(
        p0::Vector{<:Real},
        geometry::Geometry,
        is_geodesic::Bool
    )

Integrate a trajectory and convert to Cartesian coordinates.
"""
function cartesian_trajectory(
    p0::Vector{<:Real},
    geometry::Geometry,
    is_geodesic::Bool
)
    if is_geodesic
        f = setup_geodesic_solver(geometry)
    else
        f = setup_spinhall_solver_norot(geometry)
    end
    sol = f(p0, true)
    g(x::Vector{<:Real}) = spherical_to_cartesian(x, geometry.params.a)
    return mapslices(g, sol[2:4, :], dims=1)
end


"""
    plot_geodesics!(
        fig::Plots.Plot,
        Xgeo::Matrix{<:Real},
        geometry::Geometry
    )

Plot the geodesic solutions.
"""
function plot_geodesics!(
    fig::Plots.Plot,
    Xgeo::Matrix{<:Real},
    geometry::Geometry,
)
    Ngeo = size(Xgeo)[1]
    colors = Plots.palette(:lightrainbow,Ngeo)
    for i in 1:Ngeo
        X = cartesian_trajectory(Xgeo[i, 1:2], geometry, true)
        Plots.plot!(fig, X[1,:], X[2, :], X[3, :];label=nothing, color=colors[i])
    end
end


"""
    plot_spinhall_trajectories!(
        fig::Plots.Plot,
        Xspinhall::Array{<:Real, 4},
        ϵs::Vector{<:Real},
        geometry::Geometry{<:Real}
    )

Plot the spin-Hall solutions.
"""
function plot_spinhall_trajectories!(
    fig::Plots.Plot,
    Xspinhall::Array{<:Real, 4},
    ϵs::Vector{<:Real},
    geometry::Geometry{<:Real}
)
    
    Neps = length(ϵs)
    geometries = [vary_ϵ(ϵ, geometry) for ϵ in ϵs]

    colpos = Plots.palette(:rainbow, Neps)
    colneg = Plots.palette(:rainbow, Neps)
    for igeo in 1:size(Xspinhall)[3]
        for i in 1:Neps
            geometry = geometries[i]
            for s in [-2, 2]
                if s < 0
                    geometry.params.s *= -1
                    colours = colneg
                else
                    colours = colpos
                end
                X = cartesian_trajectory(Xspinhall[i, s == 2 ? 1 : 2, igeo, 1:2], geometry, false)
                Plots.plot!(fig, X[1,:], X[2, :], X[3, :], label=nothing, color=colours[i], lw=0.25)
                s < 0 ? (geometry.params.s *= -1) : nothing
            end
        end
    end
end  


"""
    plot_blackhole!(
        fig::Plots.Plot,
        loc::Tuple{Float64, Float64, Float64},
        radius::Float64
    )

Plot the black hole. If radius = 0 use as a point.
"""
function plot_blackhole!(
    fig::Plots.Plot,
    loc::Tuple{Float64, Float64, Float64},
    radius::Float64
)
    if radius == 0.0
        Plots.scatter!(fig, loc, c="black", label="Black hole")
    else
        s = Meshes.Sphere(loc, radius)
        Plots.scatter!(fig, s, alpha=0.01, label="Black hole")
    end
end


"""
    plot_start_end!(fig::Plots.Plot, geometry)

Plot the source and observer.
"""
function plot_start_end!(fig::Plots.Plot, geometry)
    Xsource = [geometry.source.r, geometry.source.θ, geometry.source.ϕ]
    spherical_to_cartesian!(Xsource, geometry.params.a)
    Plots.scatter!(fig, [Xsource[1]], [Xsource[2]], [Xsource[3]], color="red", label="Source")

    Xobs = [geometry.observer.r, geometry.observer.θ, geometry.observer.ϕ]
    spherical_to_cartesian!(Xobs, geometry.params.a)
    Plots.scatter!(fig, [Xobs[1]], [Xobs[2]], [Xobs[3]], color="blue", label="Observer")
end


# """
#     grid_func(func::Function, thetas::T=LinRange(0, pi, 50),
#               phis::T=LinRange(0, 2pi, 50)) where T<:Union{Vector, LinRange}
# 
# Evaluate `func` on a grid given by `thetas` and `phis` if possible attempts
# to use multiple threads.
# """
# function grid_func(func::Function, thetas::T=LinRange(0, pi, 50),
#                    phis::T=LinRange(0, 2pi, 50)) where T<:Union{Vector,
#                                                                 LinRange}
#     N = length(thetas) * length(phis)
#     grid = zeros(N, 2)
#     Z = zeros(N)
#     k = 1
#     # Create a meshgrid
#     for phi in phis
#         for theta in thetas
#             grid[k, 1] = theta
#             grid[k, 2] = phi
#             k += 1
#         end
#     end
#     # Calculate fmin in parallel
#     Threads.@threads for i in 1:N
#         Z[i] = func([grid[i, 1], grid[i, 2]])
#     end
#     
#     return grid, Z
# end