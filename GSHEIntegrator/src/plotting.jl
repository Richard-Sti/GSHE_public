"""
    cartesiantrajectory(
        init_direction::Vector{<:Real},
        geometry::Geometry,
        ϵ::Real=0.0,
        s::Integer=0
    )

Integrates a trajectory and returns its Cartesian representation.
"""
function cartesiantrajectory(
    init_direction::Vector{<:Real},
    geometry::Geometry,
    ϵ::Real=0.0,
    s::Integer=0
)
    sol = solve_problem(init_direction, geometry, ϵ, s, save_everystep=true)
    g(x::Vector{<:Real}) = spherical_to_cartesian(x, geometry.a)
    return mapslices(g, sol[2:4, :], dims=1)
end


"""
    plotbh!(
        fig::Plots.Plot,
        loc::Tuple{Float64, Float64, Float64}=(0.0, 0.0, 0.0),
        radius::Float64=0.0
    )

Plot the black hole. If radius = 0 use as a point.
"""
function plotbh!(
    fig::Plots.Plot,
    loc::Tuple{Float64, Float64, Float64}=(0.0, 0.0, 0.0),
    radius::Float64=0.0
)
    if radius == 0.0
        Plots.scatter!(fig, loc, c="black", label="Background BH", ms=12.5)
    else
        s = Meshes.Sphere(loc, radius)
        Plots.scatter!(fig, s, alpha=0.01, label="Background BH")
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
    Plots.scatter!(fig, [Xsource[1]], [Xsource[2]], [Xsource[3]], color="red",
        label="Source", ms=5)

    Xobs = [geometry.observer.r, geometry.observer.θ, geometry.observer.ϕ]
    spherical_to_cartesian!(Xobs, geometry.a)
    Plots.scatter!(fig, [Xobs[1]], [Xobs[2]], [Xobs[3]], color="blue",
        label="Observer", ms=5)
    return fig
end