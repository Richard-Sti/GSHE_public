"""
    ffield_callback(geometry::Geometry)

Far field callback, terminate integration when observer radius R is reached and interpolate
such that the last saved integration step is at R. Uses the default integrator's
scheme.
"""
function ffield_callback(geometry::Geometry)
    f(r, τ, integrator) = r - geometry.observer.r
    terminate_affect!(integrator) = terminate!(integrator)
    return ContinuousCallback(
        f, terminate_affect!, interp_points=geometry.ode_options.interp_points,
        save_positions=(false, false), idxs=2)
end


"""
    horizon_callback(geometry::Geometry)

Horizon callback, terminate integration if BH horizon is reached.
"""
function horizon_callback(geometry::Geometry)
    @unpack horizon_tol = geometry.ode_options
    @assert horizon_tol >= 1 "Horizon tolerance must be greater or equal to 1."
    R = horizon_tol * (1 + sqrt(1 - geometry.a^2))
    f(x, τ, integrator) = x[2] <= R
    terminate_affect!(integrator) = terminate!(integrator)
    return DiscreteCallback(f, terminate_affect!, save_positions=(false, false))
end


"""
    poles_callback(geometry::Geometry)

Termination integration callback whenever the polar angle θ is within Δθ of
either θ = 0 or θ = π.
"""
function poles_callback(geometry::Geometry)
    θmin = geometry.ode_options.Δθ
    θmax = π - θmin
    f(x, τ, integrator) = (x[3] < θmin) | (x[3] > θmax)
    terminate_affect!(integrator) = terminate!(integrator)
    return DiscreteCallback(f, terminate_affect!, save_positions=(false, false))
end


"""
    max_Δϕ(geometry::Geometry)

Calculate Δϕ for a source and an observer for a solution that does not loop around the BH.
"""
function max_Δϕ(geometry::Geometry)
    ϕ1 = geometry.source.ϕ
    ϕ2 = geometry.observer.ϕ

    Δϕ = abs(ϕ2 - ϕ1)
    if Δϕ < π
        Δϕ = 2π - Δϕ
    end

    return Δϕ
end

"""
    numloops(ϕ0::Real, ϕf::Real)

Calculate the number of complete loops around the BH according to the starting and final ϕ.
"""
function numloops(ϕ0::Real, ϕf::Real)
    Δϕ = abs(ϕf - ϕ0) / (2π)
    return Δϕ - (Δϕ % 1)
end


"""
    loop_callback(geometry::Geometry)

Loop callback that terminates trajectories that loop around the origin.
"""
function loop_callback(geometry::Geometry)
    f(x, τ, integrator) = abs(x[4] - geometry.source.ϕ) > 1.05*max_Δϕ(geometry)
    terminate_affect!(integrator) = terminate!(integrator)
    return DiscreteCallback(f, terminate_affect!, save_positions=(false, false))
end


"""
    get_callbacks(geometry::Geometry)

Get the far field, horizon, polar and loop callbacks.
"""
function get_callbacks(geometry::Geometry)
    cbs = [ffield_callback(geometry), horizon_callback(geometry)]

    if geometry.ode_options.no_loops
        push!(cbs, loop_callback(geometry))
    end

    if geometry.ode_options.Δθ > 0
        push!(cbs, poles_callback(geometry))
    end
    return CallbackSet(cbs...)
end


"""
    init_values(init_direction::Vector{<:Real}, geometry::Geometry)

Calculate the initial values [x^0, x^1, x^2, x^3, p_1, p_2, p_3] for a trajectory emitted
by the source in a given initial direction.
"""
function init_values(init_direction::Vector{<:Real}, geometry::Geometry)
    @unpack t, r, θ, ϕ = geometry.source
    return [[t,r, θ,ϕ]; initial_spatial_comomentum(init_direction, geometry)]
end


"""
    init_values(
        init_direction::Vector{<:Real},
        geometry::Geometry,
        prev_init_direction::Vector{<:Real}
)

Calculate the initial values [x^0, x^1, x^2, x^3, p_1, p_2, p_3] assuming that the initial
direction was sampled near the positive y-axis. Thus inverse rotations it under a rotation
that originally rotated the previvous initial direction to the positive y-axis.
"""
function init_values(
    init_direction::Vector{<:Real},
    geometry::Geometry,
    prev_init_direction::Vector{<:Real}
)
    # TODO: why not a bang here?
    x = rotate_from_y(init_direction, prev_init_direction)
    return init_values(x, geometry)
end


"""
    solve_problem(prob::ODEProblem, geometry::Geometry, cb::CallbackSet;
                  save_everystep::Bool=false)

Solve an ODE problem directly.
"""
function solve_problem(prob::ODEProblem, geometry::Geometry, cb::CallbackSet;
                       save_everystep::Bool=false)
    @unpack reltol, abstol, maxiters, verbose = geometry.ode_options
    return solve(prob, Vern9(), callback=cb, save_everystep=save_everystep, reltol=reltol,
                 abstol=abstol, maxiters=maxiters, verbose=verbose)
end


"""
    solve_problem(
        init_direction::Vector{<:Real},
        prob0::ODEProblem,
        geometry::Geometry,
        cb::CallbackSet;
        save_everystep::Bool=false,
    )

Remake initial conditions and solve an ODE problem.
"""
function solve_problem(
    init_direction::Vector{<:Real},
    prob0::ODEProblem,
    geometry::Geometry,
    cb::CallbackSet;
    save_everystep::Bool=false,
)
    prob = remake(prob0, u0=init_values(init_direction, geometry))
    solve_problem(prob, geometry, cb; save_everystep=save_everystep)
end


"""
    solve_problem(
        init_direction::Vector{<:Real},
        geometry::Geometry;
        ϵ::Real,
        s::Integer,
        save_everystep::Bool=false
    )

Make an ODE problem and solve a GSHE or geodesic problem.
"""
function solve_problem(
    init_direction::Vector{<:Real},
    geometry::Geometry,
    ϵ::Real,
    s::Integer;
    save_everystep::Bool=false
)
    prob = ode_problem(geometry, ϵ, s, init_values(init_direction, geometry))
    return solve_problem(prob, geometry, get_callbacks(geometry); save_everystep=save_everystep)
end


"""
    solve_consecutive_problem(
        init_direction::Vector{<:Real},
        prev_init_direction::Vector{<:Real},
        prob0::ODEProblem,
        geometry::Geometry,
        cb::CallbackSet;
        save_everystep::Bool=false,
    )

Solve a consecutive problem where `init_direction` is specified in the reference frame wherein
`prev_init_direction` is coincident with the positive y-axis.
"""
function solve_consecutive_problem(
    init_direction::Vector{<:Real},
    prev_init_direction::Vector{<:Real},
    prob0::ODEProblem,
    geometry::Geometry,
    cb::CallbackSet;
    save_everystep::Bool=false,
)
    @assert ~(geometry.direction_coords in shadow_coords) "Shadow coords initial conditions  are not supported for consecutive ODE problems."
    prob = remake(prob0, u0=init_values(init_direction, geometry, prev_init_direction))
    return solve_problem(prob, geometry, cb; save_everystep=save_everystep)
end


"""
    angular_bounds(p::Vector{<:Real})

Check whether p = (θ, ϕ) satisfies 0 ≤ θ ≤ π and 0 ≤ ϕ < 2π.
"""
function angular_bounds(p::Vector{<:Real})
    return (0. ≤ p[1] ≤ π) & (0. ≤ p[2]  < 2π)
end


"""
    angular_bounds_y(p::Vector{<:Real}, θmax::Real)

Check whether p = (θ, ϕ) satisfies 0 ≤ θ ≤ π and 0 ≤ ϕ < 2π and whether the angular
distance of p from the Cartesian point (0, 1, 0) is less than θmax.
"""
function angular_bounds_y(p::Vector{<:Real}, θmax::Real)
    return (acos(sin(p[1])*sin(p[2])) ≤ θmax) && angular_bounds(p)
end


"""
    shadow_bounds(p::Vector{<:Real})

Ensure that -1 ≤ k2, k3 ≤ k3 and that k2^2 + k3^2 ≤ 1.
"""
function shadow_bounds(p::Vector{<:Real})
    return (-1 ≤ p[1] ≤ 1) && (-1 ≤ p[2] ≤ 1) && (p[1]^2 + p[2]^2) ≤ 1
end


"""
    in_bounds(p::Vector{<:Real}, geometry::Geometry)

Return whether the initial conditions (ψ, ρ) or (x, y) of a geodesic are in bounds.
"""
function in_bounds(p::Vector{<:Real}, geometry::Geometry)
    if geometry.direction_coords == :spherical && ~angular_bounds(p)
        return false
    elseif geometry.direction_coords in shadow_coords && ~shadow_bounds(p)
        return false
    else
        return true
    end
end


"""
    arrival_stats!(geometry::Geometry, x0::Vector{<:Real}, xf::Vector{<:Real}, ϵ::Real, s::Integer)

Calculate the arrival proper time, gravitational redshift, num of azimuthal loops and store it
in geometry.
"""
function arrival_stats!(geometry::Geometry, x0::Vector{<:Real}, xf::Vector{<:Real}, ϵ::Real, s::Integer)
    geometry.arrival_time = static_observer_proper_time(xf, geometry.a)
    geometry.redshift = obs_redshift(x0, xf, geometry.a)
    geometry.nloops = numloops(x0[4], xf[4])
    geometry.ϕkilling = ϕkilling(xf, geometry, ϵ, s)
end


"""
    is_at_robs(r, geometry)

Check whether r is approximately equal to the observer's radius.
"""
function is_at_robs(r::Real, geometry::Geometry)
    @unpack radius_reltol = geometry.opt_options
    if isapprox(r, geometry.observer.r, rtol=radius_reltol)
        return true
    else
        return false
    end
end


"""
    initial_loss(
        init_direction::Vector{<:Real},
        solver::Function,
        geometry::Geometry,
        init_directions_found::Union{Vector{<:Vector{<:Real}}, Nothing}=nothing,
        ϵ::Real,
        s::Integer
    )

Calculate the angular loss of a solver (GSHE or geodesic), which is expected to take only
the initial direction as input. If close to any previously found initial directions returns
infinity.
"""
function initial_loss(
    init_direction::Vector{<:Real},
    solver::Function,
    geometry::Geometry,
    ϵ::Real,
    s::Integer,
    init_directions_found::Union{Vector{<:Vector{<:Real}}, Nothing}=nothing,
)

    if ~in_bounds(init_direction, geometry)
        return Inf64
    end

    # If initial condition too close to old init. conds. do not integrate
    @unpack angdist_to_old, τ_to_old = geometry.opt_options
    is_first = init_directions_found === nothing
    if ~is_first && minimum([angdist(init_direction, x) for x in init_directions_found]) < angdist_to_old
        return Inf64
    end

    sol = solver(init_direction)
    x0 = sol[:, 1]
    xf = sol[:, end]

    # Check if arrived at the observer radius
    if ~is_at_robs(xf[2], geometry)
        geometry.arrival_time = NaN
        geometry.redshift = NaN
        return Inf64
    end
    arrival_stats!(geometry, x0, xf, ϵ, s)

    # Additionaly check that arrival time is sufficiently different. Might have to be turned
    # off for Schwarzschild.
    τ = geometry.arrival_time
    if ~is_first && minimum(abs(τ - X[3]) for X in init_directions_found) < τ_to_old
        return Inf64
    end

    return obs_angdist(xf, geometry)
end


"""
    consecutive_loss(
        init_direction::Vector{<:Real},
        prev_init_direction::Vector{<:Real},
        solver::Function,
        geometry::Geometry,
        θmax::Real
        nloops::Real
    ϵ::Real,
    s::Integer
    )

Calculate the angular loss of a solver. The initial direction is first arctan transformed to
enforce it is within θmax angular distance of the positive y-axis. Thus, be careful when using
this to calculate the loss directly. The previous initial direction is rotated to coincide
with the positive y-axis.
"""
function consecutive_loss(
    init_direction::Vector{<:Real},
    prev_init_direction::Vector{<:Real},
    solver::Function,
    geometry::Geometry,
    θmax::Real,
    nloops::Real,
    ϵ::Real,
    s::Integer
)
    # Transform back to angles and ensure in bounds
    px = atan_transform.(init_direction, θmax)
    if ~angular_bounds_y(px, θmax)
        return Inf64
    end
    # Solve
    sol = solver(px, prev_init_direction)
    x0 = sol[:, 1]
    xf = sol[:, end]
    # Check if arrived at the observer radius
    if ~is_at_robs(xf[2], geometry)
        geometry.arrival_time = NaN
        geometry.redshift = NaN
        return Inf64
    end
    arrival_stats!(geometry, x0, xf, ϵ, s)

    # Calculate the loss
    return obs_angdist(xf, geometry) + 2π * abs(geometry.nloops - nloops)
end


"""
    obs_angdist(xf::Vector{<:Real}, geometry::Geometry)

Calculate the angular distance an integration solution and the observer when the solution
reaches the observer radius. If it is not returns infinity.
"""
function obs_angdist(xf::Vector{<:Real}, geometry::Geometry)
    r, θ, ϕ = xf[2:4]
    if ~is_at_robs(r, geometry)
        return Inf64
    end

    return angdist(θ, ϕ, geometry.observer.θ, geometry.observer.ϕ)
end


"""
    ode_problem(odes!::Function, geometry::Geometry, x0::Vector{<:Real})

Geodesic or GSHE ODE problem with specified initial conditions.
"""
function ode_problem(geometry::Geometry, ϵ::Real, s::Integer, x0::Vector{<:Real})
    if ϵ == 0
        return geodesic_ode_problem(geometry, x0)
    else
        return gshe_ode_problem(geometry, ϵ, s, x0)
    end

    return ODEProblem{true}(odes!, x0, (0.0, 100.0geometry.observer.r), geometry)
end


"""
    ode_problem(odes!::Function, geometry::Geometry)

Geodesic or GSHE ODE problem without specified initial conditions.
"""
function ode_problem(geometry::Geometry, ϵ::Real, s::Integer)
    return ode_problem(geometry, ϵ, s, rand(geometry.dtype, 7))
end


function geodesic_ode_problem(geometry::Geometry, x0::Vector{<:Real})
    function odes!(dx::Vector{<:Real}, x::Vector{<:Real}, geometry::Geometry, τ::Real)
        return geodesic_odes!(dx, x, geometry)
    end

    return ODEProblem{true}(odes!, x0, (0.0, 100.0geometry.observer.r), geometry)
end


function gshe_ode_problem(geometry::Geometry, ϵ::Real, s::Integer, x0::Vector{<:Real})
    # Ensure type stability
    if ~isa(ϵ, geometry.dtype)
        @warn "Casting ϵ from $(typeof(ϵ)) to $(geometry.dtype)."
        ϵ = geometry.dtype(ϵ)
    end

    function odes!(dx::Vector{<:Real}, x::Vector{<:Real}, geometry::Geometry, tau::Real)
        return gshe_odes!(dx, x, geometry, ϵ, s)
    end

    return ODEProblem{true}(odes!, x0, (0.0, 100.0geometry.observer.r), geometry)
end
