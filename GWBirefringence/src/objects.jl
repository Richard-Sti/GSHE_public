"""
    params(a::Float64, ϵ::Float64, s::Int64)

A keyword struct to hold some trigonometric variables and the Kerr spin
parameter ``a``, perturbation parameter ``ϵ``, and polarisation ``s``.
"""
@with_kw mutable struct params
    a::Float64
    ϵ::Float64
    s::Int64
end


"""
    spherical_coords(t::Float64=0.0 r::Float64, theta::Float64, phi::Float64) 

Spherical coordinates object.
"""
@with_kw mutable struct spherical_coords
    t::Float64 = 0.0
    r::Float64
    theta::Float64
    phi::Float64
end


"""
    geometry(source::GWBirefringence.spherical_coords,
             observer::GWBirefringence.spherical_coords,
             params::GWBirefringence.params)

Geometry object holding the source, observer and params.
"""
@with_kw mutable struct geometry 
    source::spherical_coords
    observer::spherical_coords
    params::params
end
