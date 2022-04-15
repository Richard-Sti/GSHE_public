"""
    pi0(ψ::Real, ρ::Real, geometry::Geometry)

Calculate the initial covectors ``p_i``.
"""
function pi0(ψ::Real, ρ::Real, geometry::Geometry)
    @unpack r, θ = geometry.source
    @unpack a = geometry.params
    v3 = tetrad_boosting(r, geometry)
    s_θ, c_θ = sin(θ), cos(θ)
    return [sqrt((a^2*c_θ^2 + r^2)/(a^2 + r*(r - 2)))*sin(ψ)*cos(ρ), sqrt(a^2*c_θ^2 + r^2)*sin(ρ)*sin(ψ), s_θ*(a*s_θ*v3*sqrt(a^2 + r*(r - 2))*cos(ψ) + a*s_θ*sqrt(a^2 + r*(r - 2)) + (a^2 + r^2)*(v3 + cos(ψ)))/sqrt(-(v3^2 - 1)*(a^2*c_θ^2 + r^2))]

end


"""
    pt_null(
        r::Real,
        p_r::Real,
        p_θ::Real,
        p_ϕ::Real,
        a::Real,
        s_θ::Real,
        c_θ::Real,
        c_2θ::Real
    )

Calculate the time covector from the null condition.
"""
function pt_null(
    r::Real,
    p_r::Real,
    p_θ::Real,
    p_ϕ::Real,
    a::Real,
    s_θ::Real,
    c_θ::Real,
    c_2θ::Real
)
    return (-4*a*p_ϕ*r - 2*sqrt(4*a^2*p_ϕ^2*r^2 + (p_r^2*(a^2 + r^2 - 2*r)^2 + p_θ^2*(a^2 + r^2 - 2*r) + p_ϕ^2*(a^2*c_θ^2 + r^2 - 2*r)/s_θ^2)*(a^4*c_θ^2 + a^2*r^2*(c_2θ + 3)/2 + 2*a^2*r*s_θ^2 + r^4)))/(2*a^4*c_θ^2 + a^2*r^2*(c_2θ + 3) + 4*a^2*r*s_θ^2 + 2*r^4)
end


"""
    pt_null(x::Vector{<:Real}, a::Real)

Calculate the time covector from the null condition.
"""
function pt_null(x::Vector{<:Real}, a::Real)
    r, θ, __, p_r, p_θ, p_ϕ = @view x[2:7]
    s_θ, c_θ, c_2θ = sin(θ), cos(θ), cos(2*θ)
    return pt_null(r, p_r, p_θ, p_ϕ, a, s_θ, c_θ, c_2θ)
end


"""
    static_observer_proper_time(x::Vector{<:Real}, a::Real)

Calculate the time of a static observer f(τ).
"""
function static_observer_proper_time(x::Vector{<:Real}, a::Real)
    t, r, θ = @view x[1:3]
    return t*sqrt(-2*r/(a^2*cos(θ)^2 + r^2) + 1)
end


"""
    obs_frequency(x::Vector{<:Real}, a::Real)

Calculate the frequency observed by a static observer.
"""
function obs_frequency(x::Vector{<:Real}, a::Real)
    pt = pt_null(x, a)  # p_t
    r, θ = x[2:3]
    g00 = 2r / (r^2 + a^2 * cos(θ)^2) - 1  # 00 metric element
    dγdτ0 = sqrt(-1 / g00)  # 4-velocity of a static observer
    return -pt * dγdτ0 # - p_μ γ^μ
end


"""
    obs_redshift(x0::Vector{<:Real}, xf::Vector{<:Real}, a::Real)

Ratio of wavelength as observed by a static observer at `x0` and `xf`.
"""
function obs_redshift(x0::Vector{<:Real}, xf::Vector{<:Real}, a::Real)
    ωsource = obs_frequency(x0, a)
    ωobs = obs_frequency(xf, a)
    return ωsource / ωobs
end


"""
    tetrad_boosting(r::Real, geometry::Geometry)

Boosting source and observer function.
"""
function tetrad_boosting(r::Real, geometry::Geometry)
    Robs, θobs = geometry.observer.r, geometry.observer.θ
    Rsrc, θsrc = geometry.source.r, geometry.source.θ
    a = geometry.params.a
    return -a*exp(-(-Rsrc + r)^2)*sin(θsrc)/sqrt(Rsrc^2 - 2*Rsrc + a^2) - a*exp(-(-Robs + r)^2)*sin(θobs)/sqrt(Robs^2 - 2*Robs + a^2)

end


"""
    derivative_tetrad_boosting(r::Real, geometry::Geometry)

Derivative of the boosting source and observer function.
"""
function derivative_tetrad_boosting(r::Real, geometry::Geometry)
    Robs, θobs = geometry.observer.r, geometry.observer.θ
    Rsrc, θsrc = geometry.source.r, geometry.source.θ
    a = geometry.params.a
    return 2*a*((-Robs + r)*exp(-(-Robs + r)^2)*sin(θobs)/sqrt(Robs*(Robs - 2) + a^2) + (-Rsrc + r)*exp(-(-Rsrc + r)^2)*sin(θsrc)/sqrt(Rsrc*(Rsrc - 2) + a^2))
end
