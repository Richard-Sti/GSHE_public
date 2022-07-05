"""
    setup_geometry(
        dtype::DataType=Float64;
        rsource::Real,
        θsource::Real,
        ϕsource::Real,
        robs::Real,
        θobs::Real,
        ϕobs::Real,
        a::Real,
        s::Integer=2,
        direction_coords::Symbol=:spherical,
        ode_options::ODESolverOptions=ODESolverOptions(),
        opt_options::OptimiserOptions=OptimiserOptions()
    )

Setup the geometry.
"""
function setup_geometry(
    dtype::DataType=Float64;
    rsource::Real,
    θsource::Real,
    ϕsource::Real,
    robs::Real,
    θobs::Real,
    ϕobs::Real,
    a::Real,
    s::Integer=2,
    direction_coords::Symbol=:spherical,
    ode_options::ODESolverOptions=ODESolverOptions(),
    opt_options::OptimiserOptions=OptimiserOptions(),
    postproc_options::PostprocOptions=PostprocOptions()
)
    coords_choices = [:spherical, :shadow, :shadowpos]
    @assert direction_coords in coords_choices "`direction_coords` must be one of `$coords_choices`"

    source = SphericalCoords{dtype}(r=rsource, θ=θsource, ϕ=ϕsource)
    observer = SphericalCoords{dtype}(r=robs, θ=θobs, ϕ=ϕobs)
    return Geometry{dtype}(dtype=dtype, source=source, observer=observer, s=s, a=a,
                           direction_coords=direction_coords, ode_options=ode_options,
                           opt_options=opt_options, postproc_options=postproc_options)
end


"""
    setup_geometries(
        dtype::DataType=Float64;
        rsource::Union{Vector{T}, LinRange{T}, T},
        θsource::Union{Vector{T}, LinRange{T}, T},
        ϕsource::Union{Vector{T}, LinRange{T}, T},
        robs::Union{Vector{T}, LinRange{T}, T},
        θobs::Union{Vector{T}, LinRange{T}, T},
        ϕobs::Union{Vector{T}, LinRange{T}, T},
        a::Union{Vector{T}, LinRange{T}, T},
        s::Integer=2,
        direction_coords::Symbol=:spherical,
        ode_options::ODESolverOptions=ODESolverOptions(),
        opt_options::OptimiserOptions=OptimiserOptions()
    ) where T <: Real

Setup a vector of geometries.
"""
function setup_geometries(
    dtype::DataType=Float64;
    rsource::Union{Vector{T}, LinRange{T}, T},
    θsource::Union{Vector{T}, LinRange{T}, T},
    ϕsource::Union{Vector{T}, LinRange{T}, T},
    robs::Union{Vector{T}, LinRange{T}, T},
    θobs::Union{Vector{T}, LinRange{T}, T},
    ϕobs::Union{Vector{T}, LinRange{T}, T},
    a::Union{Vector{T}, LinRange{T}, T},
    s::Integer=2,
    direction_coords::Symbol=:spherical,
    ode_options::ODESolverOptions=ODESolverOptions(),
    opt_options::OptimiserOptions=OptimiserOptions(),
    postproc_options::PostprocOptions=PostprocOptions()
) where T <: Real
    geometries = Vector{Geometry{dtype}}()
    for rs in rsource, θs in θsource, ϕs in ϕsource, ro in robs, θo in θobs, ϕo in ϕobs, ai in a
        geo = setup_geometry(dtype;
            rsource=rs, θsource=θs, ϕsource=ϕs, robs=ro, θobs=θo, ϕobs=ϕo, a=ai, s=s,
            direction_coords=direction_coords, ode_options=ode_options,
            opt_options=opt_options, postproc_options=postproc_options)
        push!(geometries, geo)
    end
    return geometries
end


"""
    check_geometry_types(geometries::Vector{<:Geometry{<:Real}})

Check that each geometry has the same data type.
"""
function check_geometry_dtypes(geometries::Vector{<:Geometry{<:Real}})
    dtype = geometries[1].dtype
    @assert all([dtype == geo.dtype for geo in geometries]) ("All geometry dtypes must be the same.")
end


"""
    setup_initial_solver(geometry::Geometry, ϵ::Real, s::Integer)

Setup the initial geodesics or GSHE solver.
"""
function setup_initial_solver(geometry::Geometry, ϵ::Real, s::Integer)
    cb = get_callbacks(geometry)
    prob = ode_problem(geometry, ϵ, s)

    f(init_direction::Vector{<:Real}) = solve_problem(init_direction, prob, geometry, cb)
    return f
end

"""
    setup_consecutive_solver(geometry::Geometry, ϵ::Real, s::Integer)

Setup a consecutive solver (GSHE or geodesic) whose `init_direction` is specified in the
reference frame where the `prev_init_direction` coincides with the positive y-axis.
"""
function setup_consecutive_solver(geometry::Geometry, ϵ::Real, s::Integer)
    cb = get_callbacks(geometry)
    prob = ode_problem(geometry, ϵ, s)
    # Integrator function
    function solver(init_direction::Vector{<:Real}, prev_init_direction::Vector{<:Real};
                    save_everystep::Bool=false)
        return solve_consecutive_problem(init_direction, prev_init_direction, prob, geometry, cb;
                                         save_everystep=save_everystep)
    end
    return solver
end


"""
    setup_initial_loss(geometry::Geometry, ϵ::Real, s::Integer)

Setup the initial loss (GSHE or geodesic).
"""
function setup_initial_loss(geometry::Geometry, ϵ::Real, s::Integer)
    solver = setup_initial_solver(geometry, ϵ, s)
    # Loss function, define with two methods
    function loss(
        init_direction::Vector{<:Real},
        init_directions_found::Union{Vector{<:Vector{<:Real}}, Nothing}=nothing,
    )
        return initial_loss(init_direction, solver, geometry, ϵ, s, init_directions_found)
    end

    return loss
end


"""
    setup_consecutive_loss(geometry::Geometry, ϵ::Real, s::Integer, nloops::Real)

Setup a consecutive loss.
"""
function setup_consecutive_loss(geometry::Geometry, ϵ::Real, s::Integer, nloops::Real)
    solver = setup_consecutive_solver(geometry, ϵ, s)
    # Loss function, define with two methods
    function loss(init_direction::Vector{<:Real}, prev_init_direction::Vector{<:Real}, θmax::Real)
        return consecutive_loss(init_direction, prev_init_direction, solver, geometry, θmax, nloops, ϵ, s)
    end
    return loss
end


"""
    sort_configurations!(
        Xgeos::Array{<:Real, 3},
        Xgshes::Union{Array{<:Real, 5}, Nothing}=nothing
    )

Sort the different configurations to achieve continuity when varying an extrinsic paramater.
"""
function sort_configurations!(
    Xgeos::Array{<:Real, 3},
    Xgshes::Union{Array{<:Real, 5}, Nothing}=nothing
)
    flip_geo = zero(Xgeos[1, 1, ..])
    if ~isnothing(Xgshes)
        flip_gshe = zero(Xgshes[1, 1, ..])
    end

    for i in 1:(size(Xgeos, 1)-1)
        Δσ = [angdist(Xgeos[i, 1, 1:2], Xgeos[i+1, jj, 1:2]) for jj in 1:2]
        Δt = [abs(Xgeos[i, 1, 3] - Xgeos[i+1, jj, 3]) for jj in 1:2]

        if (argmin(Δσ) != argmin(Δt))
            @warn "Δσ and Δt do not match for i = $i."
            flush(stdout)
        end

        # In this case the geodesics match. Continue
        if argmin(Δt) == 1
            continue
        end

        # Flip the array rows
        flip_geo .= Xgeos[i+1, 1, ..]
        Xgeos[i+1, 1, ..] = Xgeos[i+1, 2,..]
        Xgeos[i+1, 2, ..] .= flip_geo
        # Optionalli flip GSHE
        if ~isnothing(Xgshes)
            flip_gshe .= Xgshes[i+1, 1, ..]
            Xgshes[i+1, 1, ..] = Xgshes[i+1, 2, ..]
            Xgshes[i+1, 2, ..] .= flip_gshe
        end

    end
end


"""
    toarray(X::Union{Vector{<:Matrix{<:Real}}, Vector{<:Array{<:Real}}})

Convert a vector of matrices or arrays of real numbers in to an array. The first index of
the output array is that of the original vector elements.
"""
function toarray(X::Union{Vector{<:Matrix{<:Real}}, Vector{<:Array{<:Real}}})
    N = length(X)
    out = zeros(N, size(X[1])...)
    for i in 1:N
        out[i, ..] = X[i][..]
    end

    return out
end


"""
    fit_timing(
        ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
        Xgeo::Matrix{<:Real},
        Xgshe::Array{<:Real, 4},
        geometry::Geometry;
        fit_gshe_gshe::Bool=false
    )

Calculate power law fits to a `Xgeo` and `Xgshe`.
"""
function fit_timing(
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
    Xgeo::Matrix{<:Real},
    Xgshe::Array{<:Real, 4},
    geometry::Geometry;
    fit_gshe_gshe::Bool=false
)
    N = size(Xgeo, 1)
    dim = fit_gshe_gshe ? 3 : 2

    αs = fill(NaN, N, dim, 2)
    βs = fill(NaN, N, dim, 2)

    for n in 1:N
        if isnan(Xgeo[n, 3])
            continue
        end
        # Calculate the GSHE to geodesic fit
        fit_gshe_to_geo = fit_Δts(ϵs, Xgshe[n, ..], Xgeo[n, :], geometry)
        for i in 1:2
            αs[n, i, :] .= fit_gshe_to_geo[i]["alpha"]
            βs[n, i, :] .= fit_gshe_to_geo[i]["beta"]
        end

        if fit_gshe_gshe
            # Calculate the GSHE to GSHE fit
            fit_gshe_to_gshe = fit_Δts(ϵs, Xgshe[n, ..], geometry)
            αs[n, 3, :] = fit_gshe_to_gshe["alpha"]
            βs[n, 3, :] = fit_gshe_to_gshe["beta"]
        end
    end

    return αs, βs
end


"""
    fit_timing(
        ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
        Xgeo::Matrix{<:Real},
        Xgshe::Array{<:Real, 3},
        geometry::Geometry
    )

Fit timing between a geodesic and a single polarisation GSHE trajectories.
"""
function fit_timing(
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
    Xgeo::Matrix{<:Real},
    Xgshe::Array{<:Real, 3},
    geometry::Geometry
)
    N = size(Xgeo, 1)
    αs = fill(NaN, N, 2)
    βs = fill(NaN, N, 2)

    for n in 1:N
        if isnan(Xgeo[n, 3])
            continue
        end

        fit = fit_Δts(ϵs, Xgshe[n, ..], Xgeo[n, ..], geometry)
        αs[n, ..] = fit["alpha"]
        βs[n, ..] = fit["beta"]
    end

    return αs, βs
end


"""
    fit_timing(
        ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
        Xgeos::Array{<:Real, 3},
        Xgshes::Array{<:Real, 5},
        geometries::Vector{<:Geometry{<:Real}};
        fit_gshe_gshe::Bool=false
    )

Calculate power law fits to a `Xgeos` and `Xgshes`.
"""
function fit_timing(
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
    Xgeos::Array{<:Real, 3},
    Xgshes::Array{<:Real, 5},
    geometries::Vector{<:Geometry{<:Real}};
    fit_gshe_gshe::Bool=false
)
    @assert size(Xgeos, 1) == size(Xgshes, 1) == length(geometries) "Inconsistent sizes."
    Nconfs = size(Xgeos, 1)
    Ngeos = size(Xgeos, 2)
    dim = fit_gshe_gshe ? 3 : 2

    αs = fill(NaN, Nconfs, Ngeos, dim, 2)
    βs = fill(NaN, Nconfs, Ngeos, dim, 2)

    # One by one calculate the α and β for each configuration
    for i in 1:Nconfs
        α, β = fit_timing(ϵs, Xgeos[i, ..], Xgshes[i, ..], geometries[i];
                          fit_gshe_gshe=fit_gshe_gshe)
        αs[i, ..] .= α
        βs[i, ..] .= β
    end

    return αs, βs
end
