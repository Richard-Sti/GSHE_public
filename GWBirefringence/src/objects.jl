"""
    Params(a::GWFloat, ϵ::GWFloat, s::Int64)

A keyword struct to hold some trigonometric variables and the Kerr spin
parameter ``a``, perturbation parameter ``ϵ``, and polarisation ``s``.
"""
@with_kw mutable struct Params
    a::GWFloat
    ϵ::GWFloat
    s::Int64
end


"""
    Spherical_coords(t::GWFloat=0.0, r::GWFloat, theta::GWFloat, phi::GWFloat)

Spherical coordinates object.
"""
@with_kw mutable struct Spherical_coords
    t::GWFloat = GWFloat(0.0)
    r::GWFloat
    theta::GWFloat
    phi::GWFloat
end


"""
    geometry(source::GWBirefringence.Spherical_coords,
             observer::GWBirefringence.Spherical_coords,
             params::GWBirefringence.Params)

Geometry object holding the source, observer and params.
"""
@with_kw mutable struct Geometry 
    source::Spherical_coords
    observer::Spherical_coords
    params::Params
end
