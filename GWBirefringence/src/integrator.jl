import DifferentialEquations: CallbackSet, ContinuousCallback, DiscreteCallback,
                              terminate!, remake, ODEProblem


"""
    ffield_callback(geometry::GWBirefringence.geometry;
                    interp_points::Int64=10)

Far field callback, terminate integration when observer radius is reached.
"""
function ffield_callback(geometry::Geometry;
                         interp_points::Int64=10)
    f(r, τ, integrator) = r - geometry.observer.r
    terminate_affect!(integrator) = terminate!(integrator)
    return ContinuousCallback(f, terminate_affect!,
                              interp_points=interp_points,
                              save_positions=(false, false),
                              idxs=2)
end


"""
    horizon_callback(geometry::Geometry)

Horizon callback, terminate integration if BH horizon is reached.
"""
function horizon_callback(geometry::Geometry)
    f(x, τ, integrator) = x[2] <= 2.0
    terminate_affect!(integrator) = terminate!(integrator)
    return DiscreteCallback(f, terminate_affect!,
                            save_positions=(false, false))
end


"""
    get_callbacks(geometry::Geometry; interp_points::Int64=10)

Get the far field and horizon callback set.
"""
function get_callbacks(geometry::Geometry; interp_points::Int64=10)
    return CallbackSet(ffield_callback(geometry),
                       horizon_callback(geometry))
end


# """
#     get_callbacks(geometry::Geometry, p::Vector{GWFloat};
#                   interp_points::Int64=10)
# 
# Get the isometry, far field and horizon callback set.
# """
# function get_callbacks(geometry::Geometry, p::Vector{GWFloat};
#                        interp_points::Int64=10)
#     # Initial covector and conserved values
#     x0, time_isometry, phi_isometry = GWBirefringence.init_values(p,
#                                                                   geometry,
#                                                                   true)
#     iso(res, x, p, τ) = GWBirefringence.isometry_residuals!(
#                               res, x, p, τ, geometry, time_isometry,
#                               phi_isometry)
#     return CallbackSet(ffield_callback(geometry),
#                        horizon_callback(geometry),
#                        ManifoldProjection(iso))
# end


"""
    init_values(p::Vector{GWFloat}, geometry::Geometry)

Calculate the vector [x^μ, p_i].
"""
function init_values(p::Vector{GWFloat}, geometry::Geometry)
    @unpack t, r, θ, ϕ = geometry.source
    ψ, ρ = p
    p_r, p_θ, p_ϕ = pi0(ψ, ρ, geometry)
    return [t, r, θ, ϕ, p_r, p_θ, p_ϕ]
end


"""
    init_values(
        p::Vector{GWFloat},
        geometry::Geometry,
        pfound::Vector{GWFloat}
    )

Calculate the vector [x^μ, p_i]. Inverse rotates `p` from the y-axis (0, 1, 0)
under a rotation that transformed `pfound` to the y-axis.
"""
function init_values(
    p::Vector{GWFloat},
    geometry::Geometry,
    pgeo::Vector{GWFloat}
)
    x = rotate_from_y(p, pgeo)
    return init_values(x, geometry)
end


"""
    solve_geodesic(
        p::Vector,
        prob::ODEProblem,
        cb::CallbackSet;
        save_everystep::Bool=false,
        reltol::Float64=1e-12,
        abstol::Float64=1e-12
    )

Solve a geodesic problem, allows changing initial conditions `p` on the fly.
"""
function solve_geodesic(
    p::Vector,
    prob::ODEProblem,
    geometry::Geometry,
    cb::CallbackSet;
    save_everystep::Bool=false,
    reltol::Float64=1e-12,
    abstol::Float64=1e-12
)
    re_prob = remake(prob, u0=init_values(p, geometry))
    return solve(re_prob, Vern9(), callback=cb, save_everystep=save_everystep,
                 reltol=reltol, abstol=abstol)
end

"""
    solve_spinhall(
        p::Vector{GWFloat},
        prob::ODEProblem,
        geometry::Geometry,
        cb::CallbackSet,
        pgeo::Vector{GWFloat};
        save_everystep::Bool=false,
        reltol::Float64=1e-12,
        abstol::Float64=1e-12
)

Solve a Spin-hall problem, allows changing initial conditions `p` on the fly.
"""
function solve_spinhall(
    p::Vector{GWFloat},
    prob::ODEProblem,
    geometry::Geometry,
    cb::CallbackSet,
    pgeo::Vector{GWFloat};
    save_everystep::Bool=false,
    reltol::Float64=1e-12,
    abstol::Float64=1e-12
)
    re_prob = remake(prob, u0=init_values(p, geometry, pgeo))
    return solve(re_prob, Vern9(), callback=cb, save_everystep=save_everystep,
                 reltol=reltol, abstol=abstol)
end


"""
    angular_bounds(p::Vector{GWFloat})

Check whether `p = [θ, ϕ]` satisfies 0 ≤ θ ≤ π and 0 ≤ ϕ < 2π.
"""
function angular_bounds(p::Vector{GWFloat})
    return (0. ≤ p[1] ≤ π) & (0. ≤ p[2]  < 2π)
end


"""
    angular_bounds(p::Vector{GWFloat}, θmax::GWFloat)

Check whether `p = [θ, ϕ]` satisfies 0 ≤ θ ≤ π and 0 ≤ ϕ < 2π and whether the
angular distance of `p` from the Cartesian point (0, 1, 0) is less than θmax.
"""
function angular_bounds(p::Vector{GWFloat}, θmax::GWFloat)
    return (acos(sin(p[1])*sin(p[2])) ≤ θmax) & angular_bounds(p)
end


"""
    loss(
        p::Vector{GWFloat},
        Xfound::Union{Vector{Vector{GWFloat}}, Nothing},
        fsolve::Function,
        geometry::Geometry;
        rtol::Float64=1e-10,
    )

Calculate the angular loss of a geodesic, `fsolve` expected to take
only `p` as input. Checks whether initial condition close to any of `Xfound`.
"""
function geodesic_loss(
    p::Vector{GWFloat},
    pfound::Union{Vector{Vector{GWFloat}}, Nothing},
    fsolve::Function,
    geometry::Geometry;
    rtol::Float64=1e-10,
)
    # Check angular bounds
    if ~angular_bounds(p)
        return Inf64
    end

    # If initial condition too close to old init. conds. do not integrate
    if pfound !== nothing && minimum([angdist(p, x) for x in pfound]) < rtol
        return Inf64
    end

    sol = fsolve(p)
    # Check that the radial distance is within tolerance
    return obs_angdist(sol[:, end], geometry, rtol=rtol)
end


"""
    spinhall_loss(
        p::Vector{GWFloat},
        pgeo::Vector{GWFloat},
        θmax::GWFloat,
        fsolve::Function,
        geometry::Geometry;
        rtol::Float64=1e-10,
    )

Calculate the angular loss of a Spin-hall trajectory, searches for a solution
that is within `θmax` of a geodesic solution `pgeo`.
"""
function spinhall_loss(
    p::Vector{GWFloat},
    pgeo::Vector{GWFloat},
    θmax::GWFloat,
    fsolve::Function,
    geometry::Geometry;
    rtol::Float64=1e-10,
)
    px = GWBirefringence.atan_transform.(p, θmax)
    if ~angular_bounds(px, θmax)
        return Inf64
    end
    sol = fsolve(px, pgeo)
    return obs_angdist(sol[:, end], geometry, rtol=rtol)
end


"""
    obs_angdist(
        sol::Vector{GWFloat},
        geometry::Geometry;
        rtol::Float64=1e-10
    )

Calculate the angular distance between (θ, ϕ) and the observer. Ensures that
the solution's radius is within tolerance close to the observer's radius.
"""
function obs_angdist(
    sol::Vector{GWFloat},
    geometry::Geometry;
    rtol::Float64=1e-10
)
    geometry.xf .= sol
    # Additionally parametrise the time in the observer's proper time
    geometry.xf[1] = obs_proper_time(geometry.xf[1], geometry)
    r, θ, ϕ = geometry.xf[2:4]
    if ~isapprox(r, geometry.observer.r, rtol=rtol)
        return Inf64
    end
    return angdist(θ, ϕ, geometry.observer.θ, geometry.observer.ϕ)
end


"""
    ode_problem(odes!::Function, geometry::Geometry)

ODE problem without specifying the initial conditions.
"""
function ode_problem(odes!::Function, geometry::Geometry)
    return ODEProblem{true}(odes!,
                            rand(7), 
                            (0.0, 100.0geometry.observer.r),
                            geometry)
end


"""
    ode_problem(odes!::Function, geometry::Geometry, x0::Vector{GWFloat})

ODE probelem with specified initial conditins.
"""
function ode_problem(odes!::Function, geometry::Geometry, x0::Vector{GWFloat})
    return ODEProblem{true}(odes!,
                            x0,
                            (0.0, 100.0geometry.observer.r),
                            geometry)
end