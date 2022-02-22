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
    f(r, tau, integrator) = r - geometry.observer.r
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
    f(x, tau, integrator) = x[2] <= 2.0
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


"""
    get_callbacks(geometry::Geometry, p::Vector{GWFloat};
                  interp_points::Int64=10)

Get the isometry, far field and horizon callback set.
"""
function get_callbacks(geometry::Geometry, p::Vector;
                       interp_points::Int64=10)
    # Initial covector and conserved values
    x0, time_isometry, phi_isometry = GWBirefringence.init_values(p,
                                                                  geometry,
                                                                  true)
    iso(res, x, p, tau) = GWBirefringence.isometry_residuals!(
                              res, x, p, tau, geometry, time_isometry,
                              phi_isometry)
    return CallbackSet(ffield_callback(geometry),
                       horizon_callback(geometry),
                       ManifoldProjection(iso))
end


"""
    init_values(p::Vector, geometry::Geometry,
                enforce_isometry::Bool)

Calculate a vector of the initial vector ``x^mu`` and initial covector ``p_i``,
in this order given the system geometry.
"""
function init_values(p::Vector, geometry::Geometry,
                     enforce_isometry::Bool)
    @unpack t, r, theta, phi = geometry.source

    if length(p) == 3 
        __, psi, rho = cartesian_to_spherical(p)
    else
        psi, rho = p
    end
    p_r, p_theta, p_phi = pi0(psi, rho, geometry)
    x0 = [t, r, theta, phi, p_r, p_theta, p_phi]

    if enforce_isometry
        return (x0, time_killing_conservation(x0, geometry),
                phi_killing_conservation(x0, geometry))
    else
        return x0
    end
end


"""
    solve_geodesic(p::Vector, prob::ODEProblem,
                   cb::CallbackSet, init_pos::Function;
                   save_everystep::Bool=false, reltol::Float64=1e-12,
                   abstol::Float64=1e-12)

Solve a geodesic problem, allows changing intial conditions ``p`` on the fly.
"""
function solve_geodesic(p::Vector, prob::ODEProblem,
                        cb::CallbackSet, init_pos::Function;
                        save_everystep::Bool=false, reltol::Float64=1e-12,
                        abstol::Float64=1e-12)
    re_prob = remake(prob, u0=init_pos(p))
    return solve(re_prob, Vern9(), callback=cb, save_everystep=save_everystep,
                 reltol=reltol, abstol=abstol)
end


"""
    loss(
        p::Vector,
        Xfound::Union{Vector{Vector{GWFloat}}, Nothing},
        solve_geodesic::Function,
        geometry::Geometry;
        rtol::Float64=1e-10,
    )

Calculate the angular loss of a geodesic, `solve_geodesic` expected to take
only `p` as input.
"""
function loss(
    p::Vector,
    Xfound::Union{Vector{Vector{GWFloat}}, Nothing},
    solve_geodesic::Function,
    geometry::Geometry;
    rtol::Float64=1e-10,
)
    # Check that angular coords. are in their bounds
    if (length(p) === 2) & ~((0. <= p[1] <= pi) & (0. <= p[2] <= 2pi))
        return Inf
    end
    # If initial condition too close to old init. conds. do not integrate
    if Xfound !== nothing
        min_dist = minimum([angdist(p, x) for x in Xfound])
        if min_dist < rtol
            return min_dist^0.6
        end
    end
    # Solve the solution
    sol = solve_geodesic(p)
    # Check that the radial distance is within tolerance
    if ~isapprox(sol[2, end], geometry.observer.r, rtol=rtol)
        return Inf
    end

    return angdist(
            sol[3:4, end],
            [geometry.observer.theta, geometry.observer.phi]
        )
end


# """
#     loss_gradient!(
#         F,
#         G::Vector,
#         p::Vector,
#         result::DiffResults.MutableDiffResult,
#         floss::Function
#     )
# 
# Calculates the gradient of the loss function and stores it in ``G``.
# """
# function loss_gradient!(
#     F,
#     G,
#     p::Vector,
#     result::MutableDiffResult,
#     floss::Function
# )
#     gradient!(result, floss, p)
#     if G !== nothing
#         G[:] = result.derivs[1]
#     end
# 
#     if F !== nothing
#         return result.value
#     end
# end


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