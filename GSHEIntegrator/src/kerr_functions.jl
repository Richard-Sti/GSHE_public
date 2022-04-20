"""
    initial_spatial_comomentum(k::Vector{<:Real}, geometry::Geometry)

Calculate the initial covectors [p_1, p_2, p_3] for a given initial direction and geometry.
"""
function initial_spatial_comomentum(k::Vector{<:Real}, geometry::Geometry)
    @unpack r, θ = geometry.source
    a = geometry.a
    v3 = tetrad_boosting(r, geometry)
    s_θ, c_θ = sin(θ), cos(θ)


    if geometry.direction_coords == :spherical
        ψ, ρ = k
        return [sqrt((a^2*c_θ^2 + r^2)/(a^2 + r*(r - 2)))*sin(ψ)*cos(ρ), sqrt(a^2*c_θ^2 + r^2)*sin(ρ)*sin(ψ), s_θ*(a*s_θ*v3*sqrt(a^2 + r*(r - 2))*cos(ψ) + a*s_θ*sqrt(a^2 + r*(r - 2)) + (a^2 + r^2)*(v3 + cos(ψ)))/sqrt(-(v3^2 - 1)*(a^2*c_θ^2 + r^2))]
    elseif geometry.direction_coords == :shadow
        k2, k3 = k
        return [-sqrt(-(a^2*c_θ^2 + r^2)*(k2^2 + k3^2 - 1)/(a^2 + r*(r - 2))), k2*sqrt(a^2*c_θ^2 + r^2), s_θ*(a*k3*s_θ*v3*sqrt(a^2 + r*(r - 2)) + a*s_θ*sqrt(a^2 + r*(r - 2)) + (a^2 + r^2)*(k3 + v3))/sqrt(-(v3^2 - 1)*(a^2*c_θ^2 + r^2))]
    elseif geometry.direction_coords == :shadowpos
        k2, k3 = k
        return [sqrt(-(a^2*c_θ^2 + r^2)*(k2^2 + k3^2 - 1)/(a^2 + r*(r - 2))), k2*sqrt(a^2*c_θ^2 + r^2), s_θ*(a*k3*s_θ*v3*sqrt(a^2 + r*(r - 2)) + a*s_θ*sqrt(a^2 + r*(r - 2)) + (a^2 + r^2)*(k3 + v3))/sqrt(-(v3^2 - 1)*(a^2*c_θ^2 + r^2))]
    else
        return NaN
    end
end


"""
    time_comomentum(x::Vector{<:Real}, a::Real)

Calculate the time covector from the null condition.
"""
function time_comomentum(x::Vector{<:Real}, a::Real)
    r, θ = x[2:3]
    p_r, p_θ, p_ϕ = x[5:7]
    s_θ, c_θ, c_2θ = sin(θ), cos(θ), cos(2θ)
    return (-4*a*p_ϕ*r - 2*sqrt(4*a^2*p_ϕ^2*r^2 + (p_r^2*(a^2 + r^2 - 2*r)^2 + p_θ^2*(a^2 + r^2 - 2*r) + p_ϕ^2*(a^2*c_θ^2 + r^2 - 2*r)/s_θ^2)*(a^4*c_θ^2 + a^2*r^2*(c_2θ + 3)/2 + 2*a^2*r*s_θ^2 + r^4)))/(2*a^4*c_θ^2 + a^2*r^2*(c_2θ + 3) + 4*a^2*r*s_θ^2 + 2*r^4)
end


"""
    static_observer_proper_time(x::Vector{<:Real}, a::Real)

Calculate the proper time of an observer located at constant r, θ at coordinate time t.
"""
function static_observer_proper_time(x::Vector{<:Real}, a::Real)
    t, r, θ = @view x[1:3]
    return t*sqrt(-2*r/(a^2*cos(θ)^2 + r^2) + 1)
end


"""
    obs_frequency(r::Real, θ::Real, a::Real)

Calculate the frequency observed by a static observer: f = - p_μ t^μ, where t^μ is the
observer's 4-velocity.
"""
function obs_frequency(x::Vector{<:Real}, a::Real)
    r, θ = x[2:3]
    g00 = 2r / (r^2 + a^2 * cos(θ)^2) - 1
    # 4-velocity of a static observer. Only the 0 component is non-zero
    dγdτ0 = sqrt(-1 / g00)
    return -time_comomentum(x, a) * dγdτ0
end


"""
    obs_redshift(x0::Vector{<:Real}, xf::Vector{<:Real}, a::Real)

Ratio of wavelength as observed by a static observer at `x0` and `xf`.

TODO: fix this
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
    a = geometry.a
    return -a*exp(-(-Rsrc + r)^2)*sin(θsrc)/sqrt(Rsrc^2 - 2*Rsrc + a^2) - a*exp(-(-Robs + r)^2)*sin(θobs)/sqrt(Robs^2 - 2*Robs + a^2)

end


"""
    derivative_tetrad_boosting(r::Real, geometry::Geometry)

Derivative with respect to radius of the boosting source and observer function.
"""
function derivative_tetrad_boosting(r::Real, geometry::Geometry)
    Robs, θobs = geometry.observer.r, geometry.observer.θ
    Rsrc, θsrc = geometry.source.r, geometry.source.θ
    a = geometry.a
    return 2*a*((-Robs + r)*exp(-(-Robs + r)^2)*sin(θobs)/sqrt(Robs*(Robs - 2) + a^2) + (-Rsrc + r)*exp(-(-Rsrc + r)^2)*sin(θsrc)/sqrt(Rsrc*(Rsrc - 2) + a^2))
end


"""
    kerr_BL(r::Real, θ::Real, a::Real)

The Kerr metric in Boyer-Lingquist coordinates, assuming c = G = M = 1.
"""
function kerr_BL(r::Real, θ::Real, a::Real)
    Rs = 2
    sθ = sin(θ)
    Σ = r^2 + a^2 * cos(θ)^2
    Δ = r^2 - Rs * r + a^2
    A  = (r^2 + a^2)^2 - a^2 * Δ * sθ^2

    gtt = -(1 - Rs * r / Σ)
    grr = Σ / Δ
    gtϕ = -Rs * a * r * sθ^2 / Σ
    gϕϕ = A * sθ^2 / Σ

    return [gtt   0   0  gtϕ
            0    grr  0   0
            0     0   Σ   0
            gtϕ   0   0  gϕϕ]
end