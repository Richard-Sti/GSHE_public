import DiffResults: GradientResult
import Optim: Sphere


"""
    setup_geometry(;
        r_source::GWFloat,
        theta_source::GWFloat,
        phi_source::GWFloat,
        r_obs::GWFloat,
        theta_obs::GWFloat,
        phi_obs::GWFloat,
        a::GWFloat,
        eps::GWFloat,
        s::Int64=2
    )

Setup the geometry.
"""
function setup_geometry(;
    r_source::GWFloat,
    theta_source::GWFloat,
    phi_source::GWFloat,
    r_obs::GWFloat,
    theta_obs::GWFloat,
    phi_obs::GWFloat,
    a::GWFloat,
    eps::GWFloat,
    s::Int64=2
)
    source = Spherical_coords(r=r_source,
                              theta=theta_source,
                              phi=phi_source)
    observer = Spherical_coords(r=r_obs,
                                theta=theta_obs,
                                phi=phi_obs)
    params = Params(a=a, ϵ=eps, s=s)

    return Geometry(source=source, observer=observer, params=params)
end


"""
    setup_problem(geometry::GWBirefringence.Geometry)

Sets up the geodesic problem.
"""
function setup_geodesic_problem2(geometry::GWBirefringence.Geometry)
    # Get callbacks from upthere
    cb = get_callbacks(geometry)
    # ODEProblem
    prob = ode_problem(geodesic_odes!, geometry)
    # Integrator function
    function fsolver(p::Vector{GWFloat}, save_everystep::Bool=false)
        solve_geodesic(p, prob, geometry, cb; save_everystep=save_everystep)
    end

    # Loss function, define with two methods
    function floss(
        p::Vector{GWFloat},
        pfound::Union{Vector{Vector{GWFloat}}, Nothing}=nothing,
    )
        return geodesic_loss(p, pfound, fsolver, geometry)
    end

    return Problem(solve_geodesic=fsolver, loss=floss)
end


function setup_spinhall_problem(geometry::GWBirefringence.Geometry)
    # Get callbacks from upthere
    cb = get_callbacks(geometry)
    prob = ode_problem(spinhall_odes!, geometry)
    # Integrator function
    function fsolver(
        p::Vector{GWFloat},
        pgeo::Vector{GWFloat};
        save_everystep::Bool=false
    )
        solve_spinhall(p, prob, geometry, cb, pgeo; save_everystep=save_everystep)
    end

    # Loss function, define with two methods
    function floss(
        p::Vector{GWFloat},
        pgeo::Vector{GWFloat},
        θmax::GWFloat
    )
        return spinhall_loss(p, pgeo, θmax, fsolver, geometry)
    end

    return Problem(solve_geodesic=fsolver, loss=floss)
end

"""
    solve_config!(
        prob::GWBirefringence.Problem,
        geometry::GWBirefringence.Geometry;
        Nminima::Int64=2
    )

Solves a particular configuration, writing results into `X` at step `step`.
"""
function solve_config(
    prob::GWBirefringence.Problem,
    geometry::GWBirefringence.Geometry;
    Nminima::Int64=2,
)

    Xplus = search_unique_minima(prob.find_min, Nminima)
    Xplus = timing_minima(prob.solve_geodesic, Xplus)

    geometry.params.s *= -1
    Xminus = search_unique_minima(prob.find_min, Nminima)
    Xminus  = timing_minima(prob.solve_geodesic, Xminus)
    geometry.params.s *= -1

    Xminus = match_Xminus(Xplus, Xminus)
    return Xplus, Xminus
end