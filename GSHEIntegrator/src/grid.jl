"""
    make_2dmesh(
        xs::Union{Vector{<:Real}, LinRange{<:Real}},
        ys::Union{Vector{<:Real}, LinRange{<:Real}}
    )

Create a 2D meshgrid from 1D vectors of length `m`` and `n`. Output shape is (`m x n`, 2).
"""
function make_2dmesh(
    xs::Union{Vector{<:Real}, LinRange{<:Real}},
    ys::Union{Vector{<:Real}, LinRange{<:Real}}
)
    N = length(xs) * length(ys)
    grid = zeros(N, 2)

    k = 1
    for x in xs, y in ys
        grid[k, 1] = x
        grid[k, 2] = y
        k += 1
    end

    return grid
end


"""
    grid_evaluate_scalar(
        f::Function,
        x::Union{Vector{<:Real}, LinRange{<:Real}},
        y::Union{Vector{<:Real}, LinRange{<:Real}};
        like_grid::Bool=true
    )

Evaluate function f(<:Vector{Real}) on a 2D grid.
"""
function grid_evaluate_scalar(
    f::Function,
    x::Union{Vector{<:Real}, LinRange{<:Real}},
    y::Union{Vector{<:Real}, LinRange{<:Real}};
    like_grid::Bool=true
)
    grid = make_2dmesh(x, y)
    N = size(grid, 1)
    Z = zeros(N)

    # Calculate the entries
    Threads.@threads for k in 1:N
        Z[k] = f(grid[k, :])
    end

    if like_grid
        return reshape(Z, length(x), length(y))
    end

    return Z
end


"""
    grid_evaluate_timing(
        directions::Matrix{<:Real},
        ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
        geometry::Geometry,
        from_shadow::Bool=true;
        direction_verbose::Bool=true,
        gshe_verbose::Bool=false
    )

Evaluate time delays for given directions. For each direction shoots a geodesic, notes at
which θ, ϕ it intersects the observer sphere and calculates the GSHE delays to that point.
"""
function grid_evaluate_timing(
    directions::Matrix{<:Real},
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
    s::Integer,
    increasing_ϵ::Bool,
    geometry::Geometry,
    from_shadow::Bool=true;
    direction_verbose::Bool=true,
    gshe_verbose::Bool=false
)
    @assert geometry.direction_coords == :spherical "Direction coords must be spherical."

    N = size(directions, 1)
    # Initialise the output arrays
    Xgeos = fill(NaN, N, 9)
    Xgshes = fill(NaN, N, length(ϵs), 9)

    # Optionally check which x^2 + y^2 > 1 and do not calculate those
    if from_shadow
        mask = reshape(mapslices(x->sum(x) ≤ 1 , directions.^2, dims=2), N)
        iters = [i for i in 1:N if mask[i]]
    else
        iters = [i for i in 1:N]
    end

    if direction_verbose
        println("Evaluation of $(length(iters)) points.")
    end

    # Shuffle so that some workers don't get preferably easy jobs
    if Threads.nthreads() > 1
        shuffle!(iters)
    end

    Threads.@threads for n in iters
        # Optionally print status
        if direction_verbose
            println("n = $n")
            flush(stdout)
        end

        Xgeo, Xgshe = time_direction(
            directions[n, :], deepcopy(geometry), s, ϵs, increasing_ϵ, from_shadow; verbose=gshe_verbose)
        Xgeos[n, ..] .= Xgeo
        Xgshes[n, ..] .= Xgshe
    end

    return Xgeos, Xgshes
end


"""
    grid_evaluate_timing(
        xs::Union{Vector{<:Real}, LinRange{<:Real}},
        ys::Union{Vector{<:Real}, LinRange{<:Real}},
        ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
        s::Integer,
        increasing_ϵ::Bool,
        geometry::Geometry,
        from_shadow::Bool=true;
        direction_verbose::Bool=true,
        gshe_verbose::Bool=false
    )
Evaluate time delays for given directions. For each direction shoots a trajectory, notes at
which θ, ϕ it intersects the observer sphere and calculates the GSHE delays to that point.

Creates a meshgrid from `xs` and `ys`.
"""
function grid_evaluate_timing(
    xs::Union{Vector{<:Real}, LinRange{<:Real}},
    ys::Union{Vector{<:Real}, LinRange{<:Real}},
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
    s::Integer,
    increasing_ϵ::Bool,
    geometry::Geometry,
    from_shadow::Bool=true;
    direction_verbose::Bool=true,
    gshe_verbose::Bool=false
)
    grid = make_2dmesh(xs, ys)
    return grid_evaluate_timing(
        grid, ϵs, s, increasing_ϵ, geometry, from_shadow;
        direction_verbose=direction_verbose, gshe_verbose=gshe_verbose
        )
end
