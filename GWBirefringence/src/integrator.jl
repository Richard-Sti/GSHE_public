import DifferentialEquations: CallbackSet, ContinuousCallback, DiscreteCallback,
                              terminate!;

"""
    init_values(p::Vector{Float64}, geometry::GWBirefringence.geometry)

Calculate a vector of the initial vector ``x^mu`` and initial covector ``p_i``,
in this order given the system geometry.
"""
function init_values(p::Vector{Float64}, geometry::GWBirefringence.geometry)
    @unpack t, r, theta, phi = geometry.source
    # Calculate r, rho and psi
    __, psi, rho = cartesian_to_spherical(p)
    p_r, p_theta, p_phi = pi0(psi, rho, geometry)
    [t, r, theta, phi, p_r, p_theta, p_phi]
end


"""
    get_callbacks(geometry::GWBirefringence.geometry, interp_points::Int64=10)

Return the callbacks to terminate integration if the far field or horizon
condition is satisfied.
"""

function get_callbacks(geometry::GWBirefringence.geometry,
                       interp_points::Int64=10)
    @unpack r = geometry.observer
    # Far field termination
    ffield_condition(u, tau, integrator) = u[2] - r;
    # BH horizon termination
    horizon_condition(u, tau, integrator) = u[2] <= 2;
    terminate_affect!(integrator) = terminate!(integrator);


    ffield_cb = ContinuousCallback(ffield_condition, terminate_affect!,
                                   interp_points=interp_points)
    horizon_cb = DiscreteCallback(horizon_condition, terminate_affect!)
    CallbackSet(ffield_cb, horizon_cb)
end


"""
    angdist(solution, geometry::GWBirefringence.geometry, rtol::Float64=1e-8)

Calculate the angular distance between the geodesic solution and the observer.
"""
function angdist(solution, geometry::GWBirefringence.geometry,
                 rtol::Float64=1e-8)
    @unpack r, theta, phi = geometry.observer
    # Check that the radii agree within tolerance
    if ~isapprox(solution[2, end], r, rtol=rtol)
        return 2.0pi
    end

    stheta, sphi = solution[3:4, end]

    return acos(cos(theta) * cos(stheta)
                + sin(theta) * sin(stheta)
                * cos(phi - sphi))
end
