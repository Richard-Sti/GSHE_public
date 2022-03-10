import DifferentialEquations: CallbackSet, ContinuousCallback, DiscreteCallback,
                              terminate!, remake, ODEProblem
#import ForwardDiff: gradient!
#import DiffResults: MutableDiffResult


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

Calculate a vector of the initial vector ``x^mu`` and initial covector ``p_i``,
in this order given the system geometry.
"""
function init_values(p::Vector{GWFloat}, geometry::Geometry)
    @unpack t, r, theta, phi = geometry.source
    ψ, ρ = p
    p_r, p_theta, p_phi = pi0(ψ, ρ, geometry)
    return [t, r, theta, phi, p_r, p_theta, p_phi]
end


"""
    init_values(
        p::Vector{GWFloat},
        geometry::Geometry,
        Rinv::Transpose{GWFloat, Matrix{GWFloat}}
    )

Calculate a vector of the initial vector ``x^mu`` and initial covector ``p_i``,
in this order given the system geometry. Assumes values sampled near the x-axis
and thus performs the inverse rotation.
"""
function init_values(
    p::Vector{GWFloat},
    geometry::Geometry,
    Xfound::Vector{GWFloat}
)
    x = rotate_from_y(p, Xfound)
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

Solve a geodesic problem, allows changing intial conditions ``p`` on the fly.
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


function solve_geodesic(
    p::Vector{GWFloat},
    prob::ODEProblem,
    geometry::Geometry,
    cb::CallbackSet,
    Xfound::Vector{GWFloat};
    save_everystep::Bool=false,
    reltol::Float64=1e-12,
    abstol::Float64=1e-12
)
    re_prob = remake(prob, u0=init_values(p, geometry, Xfound))
    return solve(re_prob, Vern9(), callback=cb, save_everystep=save_everystep,
                 reltol=reltol, abstol=abstol)
end

"""
    angular_bounds(p::Vector{GWFloat}, θxmax::GWFloat)

Check whether θ and ϕ and in range and if they lie sufficiently close to
the y-axis (0, 1, 0).
"""
function angular_bounds(p::Vector{GWFloat})
    return (0. ≤ p[1] ≤ π) & (0. ≤ p[2]  < 2π)
end


function angular_bounds(p::Vector{GWFloat}, θmax::GWFloat)
    return (acos(sin(p[1])*sin(p[2])) ≤ θmax) & angular_bounds(p)
end


"""
    loss(
        p::Vector{GWFloat},
        Xfound::Union{Vector{Vector{GWFloat}}, Nothing},
        solve_geodesic::Function,
        geometry::Geometry;
        θxmax::GWFloat=1π,
        rtol::Float64=1e-10,
    )

Calculate the angular loss of a geodesic, `solve_geodesic` expected to take
only `p` as input. Checks whether initial condition close to any of `Xfound`.
"""
function loss(
    p::Vector{GWFloat},
    Xfound::Union{Vector{Vector{GWFloat}}, Nothing},
    solve_geodesic::Function,
    geometry::Geometry;
    rtol::Float64=1e-10,
)
    # Check angular bounds
    if ~angular_bounds(p)
        return Inf64
    end

    # If initial condition too close to old init. conds. do not integrate
    if Xfound !== nothing && minimum([angdist(p, x) for x in Xfound]) < rtol
        return Inf64
    end

    sol = solve_geodesic(p)
    # Check that the radial distance is within tolerance
    return sol_angdist(sol[:, end], geometry, rtol=rtol)
end


function loss(
    p::Vector{GWFloat},
    Xfound::Vector{GWFloat},
    θmax::GWFloat,
    solve_geodesic::Function,
    geometry::Geometry;
    rtol::Float64=1e-10,
)
    if ~angular_bounds(p, θmax)
        return Inf64
    end
    p0 = copy(p)
    sol = solve_geodesic(p, Xfound)
    return sol_angdist(sol[:, end], geometry, rtol=rtol)
end


function sol_angdist(sol::Vector{GWFloat}, geometry; rtol)
    if ~isapprox(sol[2], geometry.observer.r, rtol=rtol)
        return Inf64
    end

    return angdist(
            sol[3:4],
            [geometry.observer.theta, geometry.observer.phi]
        )
end

"""
    ode_problem(odes!::Function, geometry::Geometry)
"""
function ode_problem(odes!::Function, geometry::Geometry)
    return ODEProblem{true}(odes!,
                            rand(7), 
                            (0.0, 100.0geometry.observer.r),
                            geometry)
end


"""
    ode_problem(odes!::Function, geometry::Geometry, x0::Vector{GWFloat})
"""
function ode_problem(odes!::Function, geometry::Geometry, x0::Vector{GWFloat})
    return ODEProblem{true}(odes!,
                            x0,
                            (0.0, 100.0geometry.observer.r),
                            geometry)
end