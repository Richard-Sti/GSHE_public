import DifferentialEquations: CallbackSet, ContinuousCallback, DiscreteCallback,
                              terminate!;

"""
    init_values(p::Vector{Float64}, geometry::GWBirefringence.geometry)

Calculate a vector of the initial vector ``x^mu`` and initial covector ``p_i``,
in this order given the system geometry.
"""
function init_values(p::Vector{GWFloat}, geometry::Geometry,
                     enforce_isometry::Bool)
    @unpack t, r, theta, phi = geometry.source
    # Calculate r, rho and psi
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
    ffield_callback(geometry::GWBirefringence.geometry, interp_points::Int64=10)

Far field callback, terminate integration when observer radius is reached.
"""
function ffield_callback(geometry::Geometry,
                         interp_points::Int64=10)
    f(x, tau, integrator) = x[2] - geometry.observer.r
    terminate_affect!(integrator) = terminate!(integrator)
    return ContinuousCallback(f, terminate_affect!,
                              interp_points=interp_points)
end


"""
    horizon_callback(horizon_radius::Float64=2.0)

Horizon callback, terminate integration if BH horizon is reached.
"""
function horizon_callback(horizon_radius::Float64=2.0)
    f(x, tau, integrator) = x[2] <= horizon_radius
    terminate_affect!(integrator) = terminate!(integrator)
    return DiscreteCallback(f, terminate_affect!)
end


"""
    solve_geodesic(p::Vector, geometry::GWBirefringence.geometry, cbs::Vector;
                   save_everystep::Bool=false, enforce_isometry::Bool=false,
                   reltol::Float64=1e-12, abstol::Float64=1e-12)

Solve a geodesic in Kerr.
"""
function solve_geodesic(p::Vector{GWFloat}, geometry::Geometry, cbs::Vector;
                        save_everystep::Bool=false, enforce_isometry::Bool=false,
                        reltol::Float64=1e-12, abstol::Float64=1e-12)
    # Initial (co) vector
    x0 = GWBirefringence.init_values(p, geometry, enforce_isometry)
    # Optionally enforce isometry
    if enforce_isometry
        x0, time_isometry, phi_isometry = x0
        iso(res, x, p, tau) = GWBirefringence.isometry_residuals!(
                                  res, x, p, tau, geometry, time_isometry,
                                  phi_isometry)
        iso_cb = ManifoldProjection(iso)
        cb = CallbackSet(cbs[1], cbs[2], iso_cb)
    else
        cb = CallbackSet(cbs[1], cbs[2])
    end

    odes!(dx, x, p, tau) = GWBirefringence.geodesic_odes!(dx, x, geometry)
    prob = ODEProblem{true}(odes!, x0, (0.0, 100.0geometry.observer.r), p)
    return solve(prob, Vern9(), callback=cb, save_everystep=save_everystep,
                 reltol=reltol, abstol=abstol)
end


"""
    loss(p::Vector, geometry::GWBirefringence.geometry, cbs::Vector;
              save_everystep::Bool=false, enforce_isometry::Bool=false,
              reltol::Float64=1e-12, abstol::Float64=1e-12)

Calculate the angular loss of a geodesic.
"""
function loss(p::Vector{GWFloat}, geometry::Geometry, cbs::Vector;
              save_everystep::Bool=false, enforce_isometry::Bool=false,
              reltol::Float64=1e-12, abstol::Float64=1e-12)

    if (length(p) == 2) & ~((0. <= p[1] <= pi) & (0. <= p[2] <= 2pi))
        return Inf
    end

    sol = solve_geodesic(p, geometry, cbs, save_everystep=save_everystep,
                         enforce_isometry=enforce_isometry, reltol=reltol, abstol=abstol)
    return angdist(sol, geometry)
end
