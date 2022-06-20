"""
    time_initial!(
        init_direction::Vector{<:Real},
        ϵ::Real,
        s::Integer,
        geometry::Geometry,
        from_shadow::Bool=true
    )

Time an initial direction and record where it hits a sphere of observer's radius. If ϵ is
non-zero then do this for the specified polarisation. If the BH horizon is hit returns NaNs.
"""
function time_initial!(
    init_direction::Vector{<:Real},
    ϵ::Real,
    s::Integer,
    geometry::Geometry,
    from_shadow::Bool=true
)
    if from_shadow
        x, y = init_direction
        # Check the radius, though it might have already been checked elsewhere.
        if x^2 + y^2 > 1
            return push!(init_direction, fill!(zeros(geometry.dtype, 7), NaN)...)
        end
        init_direction[1] = acos(y)
        init_direction[2] = π + asin(x / sqrt(1 - y^2))
    end

    # If initial conditions out of bounds return NaNs
    if ~in_bounds(init_direction, geometry)
        return push!(init_direction, fill!(zeros(geometry.dtype, 7), NaN)...)
    end

    # Integrate the geodesic
    sol = solve_problem(init_direction, geometry, ϵ, s)
    x0 = sol[:, 1]
    xf = sol[:, end]

    if ~is_at_robs(xf[2], geometry)
        return push!(init_direction, fill!(zeros(geometry.dtype, 7), NaN)...)
    end

    # Calculate the observer proper arrival time and redshift
    push!(init_direction,
        static_observer_proper_time(xf, geometry.a),obs_redshift(x0, xf, geometry.a),
        0.0, xf[3], xf[4], numloops(x0[4], xf[4]), ϕkilling(xf, geometry, ϵ, s))
end


"""
    time_gshe(
        X0::Vector{<:Real},
        geometry::Geometry,
        ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
        s::Integer,
        increasing_ϵ::Bool,
        verbose::Bool=true
    )

Time the GSHE trajectories that reach a specific point on a distant sphere.
"""
function time_gshe(
    X0::Vector{<:Real},
    geometry::Geometry,
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
    s::Integer,
    increasing_ϵ::Bool,
    verbose::Bool=true
)
    # Set the geometry θ and ϕ where the geodesic ended up. Watch out about pop ordering
    ϕkill = pop!(X0)
    nloops = pop!(X0)
    geometry.observer.ϕ = pop!(X0)
    geometry.observer.θ = pop!(X0)
    push!(X0, nloops, ϕkill)

    if increasing_ϵ
        Xgeo = X0
        Xgshe = solve_increasing(X0, geometry, s, ϵs; verbose=verbose)
    else
        Xgeo, Xgshe = solve_decreasing(X0, geometry, s, ϵs; verbose=verbose)
    end

    return Xgeo, Xgshe

end


"""
    time_direction(
        init_direction::Vector{<:Real},
        geometry::Geometry,
        s::Integer,
        ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
        increasing_ϵ::Bool,
        from_shadow::Bool=true;
        verbose::Bool=true
    )

Time a direction. Shoots a trajectory in a specific direction and finds the GSHE solutions
that intersect the same point. If increasing_ϵ is true the initial trajectory is geodesic,
otherwise it is the one of the maximum ϵ.
"""
function time_direction(
    init_direction::Vector{<:Real},
    geometry::Geometry,
    s::Integer,
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
    increasing_ϵ::Bool,
    from_shadow::Bool=true;
    verbose::Bool=true
)
    init_direction = copy(init_direction)
    time_initial!(init_direction, increasing_ϵ ? 0 : ϵs[end], s, geometry, from_shadow)

    if any(isnan.(init_direction))
        # Pop the two angular locations
        pop!(init_direction), pop!(init_direction)
        fill!(init_direction, NaN)
        Xgeo = init_direction
        Xgshe = fill!(Matrix{geometry.dtype}(undef, length(ϵs), 7), NaN)
    else
        Xgeo, Xgshe = time_gshe(init_direction, geometry, ϵs, s, increasing_ϵ, verbose)
    end

    return Xgeo, Xgshe
end
