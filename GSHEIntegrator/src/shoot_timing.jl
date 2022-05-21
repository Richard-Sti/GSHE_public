"""
    time_geodesic(
        init_direction::Vector{<:Real},
        geometry::Geometry,
        from_shadow::Bool=true
    )

Time a specific geodesic as a function of the initial direction. Returns τ, the time when
the geodesic intersects the observer sphere, z, the gravitational redshift, θ and ϕ, the
angles where the geodesic intersects the observer sphere.
"""
function time_geodesic(
    init_direction::Vector{<:Real},
    geometry::Geometry,
    from_shadow::Bool=true
)
    if from_shadow
        x, y = init_direction
        init_direction[1] = acos(y)
        init_direction[2] = π + asin(x / sqrt(1 - y^2))
    end
    # If initial conditions out of bounds return NaNs
    if ~is_in_geodesic_init_bounds(init_direction, geometry)
        return NaN, NaN, NaN, NaN
    end

    # Integrate the geodesic
    sol = solve_geodesic(init_direction, geometry)
    x0 = sol[:, 1]
    xf = sol[:, 2]

    r, θ, ϕ = xf[2:4]

    if ~is_at_robs(r, geometry)
        return NaN, NaN, NaN, NaN
    end
    # Calculate the observer proper arrival time and redshift
    τ = static_observer_proper_time(xf, geometry.a)
    z = obs_redshift(x0, xf, geometry.a)
    return τ, z, θ, ϕ
end


"""
    time_gshe(
        Xgeo::Vector{<:Real},
        θ::Real,
        ϕ::Real,
        geometry::Geometry,
        ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
        verbose::Bool=true
    )

Given Xgeo finds GSHE initial conditions that intersect the observer sphere at θ, ϕ.
"""
function time_gshe(
    Xgeo::Vector{<:Real},
    θ::Real,
    ϕ::Real,
    geometry::Geometry,
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
    verbose::Bool=true
)
    # Set the geometry θ and ϕ where the geodesic ended up
    geometry.observer.θ = θ
    geometry.observer.ϕ = ϕ

    return solve_gshe(Xgeo, geometry, ϵs, verbose=verbose)
end


"""
    time_direction!(
        init_direction::Vector{<:Real},
        geometry::Geometry,
        ϵs::Union{Vector{<:Real}, LinRange{<:Real}}
        from_shadow::Bool=true;
        verbose::Bool=true
    )

Time a direction. Shoots a geodesic in a specific direction and finds GSHE solutions that
intersect where the geodesic intersects the observer sphere.
"""
function time_direction!(
    init_direction::Vector{<:Real},
    geometry::Geometry,
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
    from_shadow::Bool=true;
    verbose::Bool=true
)
    # Figure out where the geodesic intersects the observer radius
    τ, z, θobs, ϕobs = time_geodesic(init_direction, geometry, from_shadow)
    push!(init_direction, τ, z)

    if isnan(θobs) || isnan(ϕobs) || any(isnan.(init_direction))
        Xgshe = fill!(Array{geometry.dtype, 3}(undef, 2, length(ϵs), 4), NaN)
    else
        Xgshe = time_gshe(init_direction, θobs, ϕobs, geometry, ϵs, verbose)
    end

    return init_direction, Xgshe
end
