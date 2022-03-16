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
        Plots.scatter!(fig, x[:, 2], x[:, 1], label="s=$s")
    end
    Plots.scatter!(fig, [Xgeo[igeo, 2]], [Xgeo[igeo, 1]], label="Geodesic")
end


@doc """
    plot_arrival_times!(
        fig::Plots.Plot,
        ϵs::Union{Vector{<:Real}, LinRange{Float64}},
        Xspinhall::Array{<:Real, 4},
        Xgeo::Matrix{<:Real},
        igeo::Int64
    )

Plot the arrival time of the geodesic and the spin-Hall perturbations.
"""
function plot_arrival_times!(
    fig::Plots.Plot,
    ϵs::Union{Vector{<:Real}, LinRange{Float64}},
    Xspinhall::Array{<:Real, 4},
    Xgeo::Matrix{<:Real},
    igeo::Int64
)   
    xplus = @view Xspinhall[:, 1, igeo, 3]
    xminus = @view Xspinhall[:, 2, igeo, 3]

    t = Xgeo[igeo, 3]

    Plots.scatter!(fig, ϵs, abs.(xplus .- t), label=L"s=2")
    Plots.scatter!(fig, ϵs, abs.(xminus .- t), label=L"s=-2")
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
        xplus = @view Xspinhall[:, 1, igeo, 3]
        xminus = @view Xspinhall[:, 2, igeo, 3]

        Plots.scatter!(fig, ϵs, abs.(xplus - xminus), label="Geodesic $igeo")
    end
end


"""
    cartesian_trajectory(
        p0::Vector{<:Real},
        geometry::GWBirefringence.Geometry,
        is_geodesic::Bool
    )

Integrate a trajectory and convert to Cartesian coordinates.
"""
function cartesian_trajectory(
    p0::Vector{<:Real},
    geometry::GWBirefringence.Geometry,
    is_geodesic::Bool
)
    if is_geodesic
        f = GWBirefringence.setup_geodesic_solver(geometry)
    else
        f = GWBirefringence.setup_spinhall_solver_norot(geometry)
    end
    sol = f(p0, true)
    return mapslices(GWBirefringence.spherical_to_cartesian, sol[2:4, :], dims=1)
end


"""
    plot_geodesics!(
        fig::Plots.Plot,
        Xgeo::Matrix{<:Real},
        geometry::GWBirefringence.Geometry
    )

Plot the geodesic solutions.
"""
function plot_geodesics!(
    fig::Plots.Plot,
    Xgeo::Matrix{<:Real},
    geometry::GWBirefringence.Geometry
)
    Ngeo = size(Xgeo)[1]
    colors = Plots.palette(:lightrainbow,Ngeo)
    for i in 1:Ngeo
        X = cartesian_trajectory(Xgeo[i, 1:2], geometry, true)
        Plots.plot!(fig, X[1,:], X[2, :], X[3, :], label="Geodesic $i", ls=:dash, color=colors[i])
    end
end


"""
    plot_spinhall_trajectories!(
        fig::Plots.Plot,
        Xspinhall::Array{<:Real, 4},
        geometries::Vector{GWBirefringence.Geometry}
    )

Plot the spin-Hall solutions.
"""
function plot_spinhall_trajectories!(
    fig::Plots.Plot,
    Xspinhall::Array{<:Real, 4},
    geometries::Vector{<:Geometry{<:Real}}
)
    
    Neps = length(geometries)
    colpos = Plots.palette(:matter, Neps)
    colneg = Plots.palette(:haline, Neps)
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
        Plots.scatter!(fig, loc, label=nothing, c="black")
    else
        s = Meshes.Sphere(loc, radius)
        Plots.scatter!(fig, s, alpha=0.0025, label=nothing)
    end
end


"""
    plot_start_end!(fig::Plots.Plot, geometry)

Plot the source and observer.
"""
function plot_start_end!(fig::Plots.Plot, geometry)
    Xsource = [geometry.source.r, geometry.source.θ, geometry.source.ϕ]
    GWBirefringence.spherical_to_cartesian!(Xsource)
    Plots.scatter!(fig, [Xsource[1]], [Xsource[2]], [Xsource[3]], color="red", label=nothing)

    Xobs = [geometry.observer.r, geometry.observer.θ, geometry.observer.ϕ]
    GWBirefringence.spherical_to_cartesian!(Xobs)
    Plots.scatter!(fig, [Xobs[1]], [Xobs[2]], [Xobs[3]], color="blue", label=nothing)
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