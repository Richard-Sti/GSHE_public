import Parameters: @with_kw;

"""
    params(a=::Float64, ϵ=::Float64, s=::Int64)

A keyword struct to hold some trigonometric variables and the Kerr spin
parameter ``a``, perturbation parameter ``ϵ``, and polarisation ``s``.
"""
@with_kw mutable struct params
    s_θ::Float64 = 0.
    c_θ::Float64 = 0.
    s_2θ::Float64 = 0.
    c_2θ::Float64 = 0.
    t_θ::Float64 = 0.
    t_2θ::Float64 = 0.
    p_t::Float64 = 0.
    a::Float64
    ϵ::Float64
    s::Int64
end;


"""
    spherical_coords(t=1.0, r=::Float64, theta=::Float64, phi=::Float64)

Spherical coordinates struct.
"""
@with_kw struct spherical_coords
    t::Float64 = 0.0
    r::Float64
    theta::Float64
    phi::Float64
end


"""
    init_values!(x0::Vector{Float64}, psi::Float64, rho::Float64,
                 source::spherical_coords, a::Float64)

Calculate a vector of the initial vector ``x^mu`` and initial covector ``p_i``,
in this order and write it into x0.
"""
function init_values!(x0::Vector{Float64}, psi::Float64, rho::Float64,
                      source::spherical_coords, a::Float64)
    x0[1] = source.t;
    x0[2] = source.r;
    x0[3] = source.theta;
    x0[4] = source.phi;
    x0[5:end] = pi0(psi, rho, source.r, source.theta, a);
    return
end


function angdist(solution, observer)
    if ~isapprox(solution[2, end], observer["r"], rtol=1e-8)
        return 2.0*pi
    end

    θ, ϕ = sol[3:4, end]

    return acos(cos(θ) * cos(observer["theta"])
                + sin(θ) * sin(observer["theta"])
                * cos(ϕ - observer["phi"]))
end
