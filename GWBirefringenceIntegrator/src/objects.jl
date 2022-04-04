@with_kw mutable struct Params{T <: Real}
    a::T
    ϵ::T
    s::T
end


@with_kw mutable struct Spherical_coords{T <: Real} <: Number
    t::T = 0.0
    r::T
    θ::T
    ϕ::T
end

@with_kw mutable struct Geometry{T <: Real}
    source::Spherical_coords{T}
    observer::Spherical_coords{T}
    params::Params{T}
    type::DataType
    arrival_time::T = 0.0
    redshift::T = 0.0
end


function Base.copy(geometry::Geometry)
    T = geometry.type
    source = GWBirefringence.Spherical_coords(@unpack t, r, θ, ϕ = geometry.source)
    observer = GWBirefringence.Spherical_coords(@unpack t, r, θ, ϕ= geometry.observer)
    params = GWBirefringence.Params(@unpack a, ϵ, s = geometry.params)
    return GWBirefringence.Geometry{T}(source=source,observer=observer, params=params, type=T)
end