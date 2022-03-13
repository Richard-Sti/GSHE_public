using LaTeXStrings
import Plots


function plot_initial_conditions(
    Xspinhall::Array{Float64, 4},
    Xgeo::Matrix{Float64},
    igeo::Int64
)
    fig = Plots.plot(dpi=600, xlabel=L"\rho", ylabel=L"\psi")
    for s in [2, -2]
        x = @view Xspinhall[:, s == 2 ? 1 : 2, igeo, 1:2]
        Plots.scatter!(fig, x[:, 2], x[:, 1], label="s=$s")
    end
    Plots.scatter!(fig, [Xgeo[igeo, 2]], [Xgeo[igeo, 1]], label="Geodesic")
    return fig
end


function plot_arrival_times(
    ϵs::Union{Vector{Float64}, LinRange{Float64}},
    Xspinhall::Array{Float64, 4},
    Xgeo::Matrix{Float64},
    igeo::Int64
)   
    fig = Plots.plot(xaxis=:log, xlabel=L"\epsilon", ylabel=L"t_{\rm obs}",
                     legend=:topleft)
    xplus = @view Xspinhall[:, 1, igeo, 3]
    xminus = @view Xspinhall[:, 2, igeo, 3]

    Plots.scatter!(fig, ϵs, xplus, label=L"s=2")
    Plots.scatter!(fig, ϵs, xminus, label=L"s=-2")
    Plots.hline!(fig, [Xgeo[igeo, 3]], label=L"{\rm Geodesic}")
    return fig
end

function plot_time_difference(
    ϵs::Union{Vector{Float64}, LinRange{Float64}},
    Xspinhall::Array{Float64, 4},
    Xgeo::Matrix{Float64}
)
    Nsols = size(Xgeo)[1]
    fig = Plots.plot(
        xaxis=:log, yaxis=:log, legend=:topleft, dpi=600,
        xlabel=L"\epsilon",
        ylabel=L"|t_{\rm obs}(s=2) - t_{\rm obs}(s=-2)|")
    for igeo in 1:Nsols
        xplus = @view Xspinhall[:, 1, igeo, 3]
        xminus = @view Xspinhall[:, 2, igeo, 3]

        Plots.scatter!(fig, ϵs, abs.(xplus - xminus), label="Geodesic $igeo")
    end
    return fig
end


function cartesian_trajectory(
    p0::Vector{Float64},
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


function plot_geodesics!(
    fig::Plots.Plot,
    Xgeo::Matrix{Float64},
    geometry::GWBirefringence.Geometry
)
    Ngeo = size(Xgeo)[1]
    colors = Plots.palette(:lightrainbow,Ngeo)
    for i in 1:Ngeo
        X = cartesian_trajectory(Xgeo[i, 1:2], geometry, true)
        Plots.plot!(fig, X[1,:], X[2, :], X[3, :], label="Geodesic $i", ls=:dash, color=colors[i])
    end
end


function plot_blackhole!(fig::Plots.Plot, loc::Tuple{Float64, Float64, Float64}, radius::Float64)
    if radius == 0.0
        Plots.scatter!(fig, loc, label=nothing, c="black")
    else
        s = Meshes.Sphere(loc, radius)
        Plots.scatter!(fig, s, alpha=0.0025, label=nothing)
    end
end


function plot_start_end!(fig::Plots.Plot, geometry)
    Xsource = [geometry.source.r, geometry.source.θ, geometry.source.ϕ]
    GWBirefringence.spherical_to_cartesian!(Xsource)
    Plots.scatter!(fig, [Xsource[1]], [Xsource[2]], [Xsource[3]], color="red", label=nothing)

    Xobs = [geometry.observer.r, geometry.observer.θ, geometry.observer.ϕ]
    GWBirefringence.spherical_to_cartesian!(Xobs)
    Plots.scatter!(fig, [Xobs[1]], [Xobs[2]], [Xobs[3]], color="blue", label=nothing)
end


function plot_spinhall_trajectories!(
    fig::Plots.Plot,
    Xspinhall::Array{Float64, 4},
    geometries::Vector{GWBirefringence.Geometry}
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
    grid_func(func::Function, thetas::T=LinRange(0, pi, 50),
              phis::T=LinRange(0, 2pi, 50)) where T<:Union{Vector, LinRange}

Evaluate `func` on a grid given by `thetas` and `phis` if possible attempts
to use multiple threads.
"""
function grid_func(func::Function, thetas::T=LinRange(0, pi, 50),
                   phis::T=LinRange(0, 2pi, 50)) where T<:Union{Vector,
                                                                LinRange}
    N = length(thetas) * length(phis)
    grid = zeros(N, 2)
    Z = zeros(N)
    k = 1
    # Create a meshgrid
    for phi in phis
        for theta in thetas
            grid[k, 1] = theta
            grid[k, 2] = phi
            k += 1
        end
    end
    # Calculate fmin in parallel
    Threads.@threads for i in 1:N
        Z[i] = func([grid[i, 1], grid[i, 2]])
    end
    
    return grid, Z
end
