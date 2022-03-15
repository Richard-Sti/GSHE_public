"""
    pi0(ψ::GWFloat, ρ::GWFloat, geometry::Geometry)

Calculate the initial covectors ``p_i``.

# Arguments
- `psi::Float64`: the initial direction polar angle
- `rho::Float64`: the initial direction azimuthal angle
- `geometry::GWBirefringence.geometry`: system geometry
"""
function pi0(ψ::GWFloat, ρ::GWFloat, geometry::Geometry)
    @unpack r, θ = geometry.source
    @unpack a = geometry.params

    v_0, __ = v_r(r, geometry.observer.r, geometry.observer.θ, a)
    s_θ, c_θ = sin(θ), cos(θ)

    return [sqrt((a^2*c_θ^2 + r^2)/(a^2 + r*(r - 2)))*sin(ρ)*sin(ψ), sqrt(a^2*c_θ^2 + r^2)*sin(ψ)*cos(ρ), s_θ*(-a^2*s_θ^2 + a^2 + r^2)*(a*s_θ*v_0*sqrt((a^2 + r*(r - 2))*(a^2*c_θ^2 + r^2))*sqrt(a^2*c_θ^2 + r^2)*cos(ψ) - a*s_θ*sqrt((a^2 + r*(r - 2))*(a^2*c_θ^2 + r^2))*sqrt(a^2*c_θ^2 + r^2) - v_0*(a^2 + r^2)*(a^2*c_θ^2 + r^2) + (a^2 + r^2)*(a^2*c_θ^2 + r^2)*cos(ψ))/(sqrt(1 - v_0^2)*(a^2*c_θ^2 + r^2)^(5/2))]
end


"""
    pt_null(r, θ, p_r, p_θ, p_ϕ, a::GWFloat, s_θ, c_θ, c_2θ)

Calculate the time covector from the null condition.
"""
function pt_null(r, θ, p_r, p_θ, p_ϕ, a::GWFloat, s_θ, c_θ, c_2θ)
    return (-4*a*p_ϕ*r - 2*sqrt(4*a^2*p_ϕ^2*r^2 + (p_r^2*(a^2 + r^2 - 2*r)^2 + p_θ^2*(a^2 + r^2 - 2*r) + p_ϕ^2*(a^2*c_θ^2 + r^2 - 2*r)/s_θ^2)*(a^4*c_θ^2 + a^2*r^2*(c_2θ + 3)/2 + 2*a^2*r*s_θ^2 + r^4)))/(2*a^4*c_θ^2 + a^2*r^2*(c_2θ + 3) + 4*a^2*r*s_θ^2 + 2*r^4)
end


function pt_null(x::Vector{GWFloat}, a::GWFloat)
    r, θ = x[2:3]
    p_r, p_θ, p_ϕ = x[5:7]
    s_θ, c_θ, c_2θ = sin(θ), cos(θ), cos(2*θ)
    return pt_null(r, θ, p_r, p_θ, p_ϕ, a, s_θ, c_θ, c_2θ)
end


"""
    obs_proper_time(t::GWFloat, geometry::Geometry)

"""
function static_observer_proper_time(x::Vector{GWFloat}, a::GWFloat)
    t, r, θ = x[1:3]
    return t * sqrt(1 + 2*r / (r^2  - 2*r + a^2 * cos(θ)^2))
end


"""
    obs_geofrequency(pt::GWFloat, r::GWFloat, θ::GWFloat, a::GWFloat)

"""
function obs_frequency(x::Vector{GWFloat}, a::GWFloat)
    pt = pt_null(x, a)
    r, θ = x[2:3]
    return - pt * sqrt(1 + 2*r / (r^2 - 2*r + a^2 * cos(θ)^2))
end



function obs_redshift(
    x0::Vector{GWFloat},
    xf::Vector{GWFloat},
    a::GWFloat
)
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
function v_r(r, R_o::GWFloat, θ_o::GWFloat, a::GWFloat)
    
    exp_sin = exp(-(-R_o + r)^2) * sin(θ_o)
    sqrroot = sqrt(R_o^2 - 2*R_o + a^2)
    
    v_0 =  -a * exp_sin / sqrroot
    v_1 = 2*a*(-R_o + r) * exp_sin / sqrroot

    return v_0, v_1
end