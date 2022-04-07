"""
    pi0(ψ::Real, ρ::Real, geometry::Geometry)

Calculate the initial covectors ``p_i``.
"""
function pi0(ψ::Real, ρ::Real, geometry::Geometry)
    @unpack r, θ = geometry.source
    @unpack a = geometry.params

    v_0, __ = v_r(r, geometry.observer.r, geometry.observer.θ, a)
    s_θ, c_θ = sin(θ), cos(θ)

    return [sqrt((a^2*c_θ^2 + r^2)/(a^2 + r*(r - 2)))*sin(ρ)*sin(ψ), sqrt(a^2*c_θ^2 + r^2)*sin(ψ)*cos(ρ), s_θ*(-a^2*s_θ^2 + a^2 + r^2)*(a*s_θ*v_0*sqrt((a^2 + r*(r - 2))*(a^2*c_θ^2 + r^2))*sqrt(a^2*c_θ^2 + r^2)*cos(ψ) - a*s_θ*sqrt((a^2 + r*(r - 2))*(a^2*c_θ^2 + r^2))*sqrt(a^2*c_θ^2 + r^2) - v_0*(a^2 + r^2)*(a^2*c_θ^2 + r^2) + (a^2 + r^2)*(a^2*c_θ^2 + r^2)*cos(ψ))/(sqrt(1 - v_0^2)*(a^2*c_θ^2 + r^2)^(5/2))]
end


"""
    pt_null(
        r::Real,
        θ::Real,
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
    θ::Real,
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
    return pt_null(r, θ, p_r, p_θ, p_ϕ, a, s_θ, c_θ, c_2θ)
end


"""
    static_observer_proper_time(x::Vector{<:Real}, a::Real)

Calculate the time of a static observer f(τ).
"""
function static_observer_proper_time(x::Vector{<:Real}, a::Real)
    t, r, θ = @view x[1:3]
    return t * sqrt(1 - 2r / (r^2 + a^2 * cos(θ)^2))
end


"""
    obs_frequency(x::Vector{<:Real}, a::Real)

Calculate the frequency observed by a static observer.
"""
function obs_frequency(x::Vector{<:Real}, a::Real)
    pt = pt_null(x, a)
    r, θ = @view x[2:3]
    return - pt * sqrt(1 + 2*r / (r^2 - 2*r + a^2 * cos(θ)^2))
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




# """
#     time_killing_conservation(x, geometry::Geometry)
# 
# Calculate the time isometry.
# """
# function time_killing_conservation(x, geometry::Geometry)
#     t, r, θ, ϕ, p_r, p_θ, p_ϕ = x
#     @unpack a, ϵ, s = geometry.params
#     c_θ = cos(θ)
#     p_t = pt_null(x, geometry)
#    -2*a .*c_θ .*p_r .*r .*s .*ϵ .*(a.^2 + r .*(r - 2)) ./((a .*p_ϕ + p_t .*(a.^2 + r.^2)) .*(a.^2 .*c_θ.^2 + r.^2).^2) + p_t
# end
# 
# 
# """
#     isometry_residuals!(
#         resid::AbstractVector{<:Real},
#         x::AbstractVector{<:Real},
#         p::AbstractVector{<:Real},
#         tau::AbstractFloat,
#         geometry::Geometry,
#         time_isometry::AbstractFloat,
#         phi_isometry::AbstractFloat
#     )
# 
# Calculate the time and phi isometry residuals.
# """
# function isometry_residuals!(
#     resid::AbstractVector{<:Real},
#     x::AbstractVector{<:Real},
#     p::AbstractVector{<:Real},
#     tau::AbstractFloat,
#     geometry::Geometry,
#     time_isometry::AbstractFloat,
#     phi_isometry::AbstractFloat
# )
#     resid[1] = time_killing_conservation(x, geometry) - time_isometry
#     resid[2] = 0.0
#     resid[3] = 0.0
#     resid[4] = phi_killing_conservation(x, geometry) - phi_isometry
#     resid[5] = 0.0
#     resid[6] = 0.0
#     resid[7] = 0.0
#     return resid
# end
# 
# 
# """
#     phi_killing_conservation(
#         x::AbstractVector{<:Real},
#         geometry::Geometry
# )
# 
# Calculate the azimuthal, phi isometry.
# """
# function phi_killing_conservation(
#     x::AbstractVector{<:Real},
#     geometry::Geometry
# )
#     t, r, θ, ϕ, p_r, p_θ, p_ϕ = x
#     @unpack a, ϵ, s = geometry.params
#     c_θ, c_2θ = cos(θ), cos(2.0*θ)
#     s_θ = sin(θ)
#     p_t = pt_null(x, geometry)
#    (4*c_θ .*p_r .*s .*ϵ .*(a.^2 + r .*(r - 2)) .*(2*a.^4 .*c_θ.^2 + a.^2 .*r.^2 .*(c_2θ + 3) + 4*a.^2 .*r .*s_θ.^2 + 2*r.^4) + (4*a.^2 .*c_θ.^2 + 4*r.^2) .*(2*a .*p_ϕ.^2 .*(a.^2 .*c_θ.^2 + r.^2) + 2*p_t .*p_ϕ .*(a.^2 + r.^2) .*(a.^2 .*c_θ.^2 + r.^2) - 2*p_θ .*r .*s .*s_θ .*ϵ .*(a.^2 + r .*(r - 2)))) ./((8*a .*p_ϕ + 8*p_t .*(a.^2 + r.^2)) .*(a.^2 .*c_θ.^2 + r.^2).^2)
# end


"""
    v_r(r, R_o::GWFloat, θ_o::GWFloat, a::GWFloat)

Boosting observer function.
"""
function v_r(r, R_o::Real, θ_o::Real, a::Real)
    
    exp_sin = exp(-(-R_o + r)^2) * sin(θ_o)
    sqrroot = sqrt(R_o^2 - 2*R_o + a^2)
    
    v_0 =  -a * exp_sin / sqrroot
    v_1 = 2*a*(-R_o + r) * exp_sin / sqrroot

    return v_0, v_1
end