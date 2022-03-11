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
    Problem(
        solve::Function,
        loss::Function, 
    )

Problem with a solver and loss for a specific geometry.
"""
@with_kw struct Problem
    solve::Function
    loss::Function
end


"""
    Spherical_coords(
        t::GWFloat=0.0,
        r::GWFloat,
        θ::GWFloat,
        ϕ::GWFloat
     )

Spherical coordinates object.
"""
@with_kw mutable struct Spherical_coords
    t::GWFloat = GWFloat(0.0)
    r::GWFloat
    θ::GWFloat
    ϕ::GWFloat
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
    xf::Vector{GWFloat} = zeros(7)
end


"""
    Base.copy(geometry::GWBirefringence.Geometry)

Deepcopy of geometry.
"""
function Base.copy(geometry::GWBirefringence.Geometry)
    source = GWBirefringence.Spherical_coords(
        @unpack t, r, θ, ϕ = geometry.source)
    observer = GWBirefringence.Spherical_coords(
        @unpack t, r, θ, ϕ= geometry.observer)
    params = GWBirefringence.Params(
        @unpack a, ϵ, s = geometry.params)
    return GWBirefringence.Geometry(source=source,
                                    observer=observer,
                                    params=params)
end
