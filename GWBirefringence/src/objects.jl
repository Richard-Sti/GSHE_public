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
        solve_geodesic::Function,
        loss::Function, 
        find_min::Function,
        find_gradmin::Union{Function, Nothing}=nothing
    )

Problem with solvers and losses for a specific geometry.
"""
@with_kw struct Problem
    solve_geodesic::Function
    loss::Function
    find_min::Function
    find_gradmin::Union{Function, Nothing} = nothing
end


"""
    Spherical_coords(
        t::GWFloat=0.0,
        r::GWFloat,
        theta::GWFloat,
        phi::GWFloat
     )

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


"""
    Base.copy(geometry::GWBirefringence.Geometry)

Deepcopy of geometry.
"""
function Base.copy(geometry::GWBirefringence.Geometry)
    source = GWBirefringence.Spherical_coords(
        @unpack t, r, theta, phi = geometry.source)
    observer = GWBirefringence.Spherical_coords(
        @unpack t, r, theta, phi = geometry.observer)
    params = GWBirefringence.Params(
        @unpack a, ϵ, s = geometry.params)
    return GWBirefringence.Geometry(source=source,
                                    observer=observer,
                                    params=params)
end
