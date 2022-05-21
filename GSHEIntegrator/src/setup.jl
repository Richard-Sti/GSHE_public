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
    @assert all([dtype == geo.dtype for geo in geometries]) ("All geometry dtypes must "
                                                             *"be the same.")
end


"""
    setup_geodesic_solver(geometry::Geometry)

Setup the geodesic solver for a given geometry.
"""
function setup_geodesic_solver(geometry::Geometry)
    cb = get_callbacks(geometry)
    prob = geodesic_ode_problem(geometry)
    f(init_direction::Vector{<:Real}) = solve_geodesic(init_direction, prob, geometry, cb)
    return f
end


"""
    setup_geodesic_loss(geometry::Geometry)

Setup the geodesic loss function for a given geometry.
"""
function setup_geodesic_loss(geometry::Geometry)
    # Loss function, define with two methods
    solver = setup_geodesic_solver(geometry)
    function loss(
        init_direction::Vector{<:Real},
        init_directions_found::Union{Vector{<:Vector{<:Real}}, Nothing}=nothing,
    )
        return geodesic_loss(init_direction, solver, geometry, init_directions_found)
    end

    return loss
end


"""
    setup_gshe_solver(geometry::Geometry, ϵ::Real, s::Integer)

Setup the GSHE trajectory solver for a given geometry, including reference frame rotations.
"""
function setup_gshe_solver(geometry::Geometry, ϵ::Real, s::Integer)
    cb = get_callbacks(geometry)
    prob = gshe_ode_problem(geometry, ϵ, s)
    # Integrator function
    function solver(init_direction::Vector{<:Real}, geodesic_init_direction::Vector{<:Real};
                    save_everystep::Bool=false)
        solve_gshe(init_direction, geodesic_init_direction, prob, geometry, cb;
                   save_everystep=save_everystep)
    end
    return solver
end


"""
    setup_gshe_loss(geometry::Geometry, ϵ::Real, s::Integer)

Setup the spin Hall trajectory loss function for a given geometry.
"""
function setup_gshe_loss(geometry::Geometry, ϵ::Real, s::Integer)
    solver = setup_gshe_solver(geometry, ϵ, s)
    # Loss function, define with two methods
    function loss(init_direction::Vector{<:Real}, geodesic_init_direction::Vector{<:Real}, θmax::Real)
        return gshe_loss(init_direction, geodesic_init_direction, solver, geometry, θmax)
    end
    return loss
end


"""
    solve_geodesics(
        geometries::Vector{<:Geometry{<:Real}},
        Nsols::Integer=2,
        verbose::Bool=true
        to_sort::Bool=true
    )

Find the geodesic initial direction and time of arrivals for a list of geometries.
"""
function solve_geodesics(
    geometries::Vector{<:Geometry{<:Real}},
    Nsols::Integer=2,
    verbose::Bool=true,
    to_sort::Bool=true
)
    check_geometry_dtypes(geometries)
    dtype = geometries[1].dtype

    N = length(geometries)
    Xgeos = Vector{Matrix{dtype}}(undef, N)

    if verbose
        println("Solving $N configurations' geodesics.")
        flush(stdout)
    end

    # Shuffle workers jobs
    if Threads.nthreads() > 1
        iters = shuffle!([i for i in 1:N])
    else
        iters = 1:N
    end

    Threads.@threads for i in iters
        if verbose
            println("Solving geodesics for geometry $i/$N")
            flush(stdout)
        end
        Xgeos[i] = find_geodesic_minima(geometries[i], Nsols)
    end

    if Nsols > 1 && to_sort
        if verbose
            println("Sorting the configurations.")
            flush(stdout)
        end
        sort_configurations!(Xgeos)
    end

    return Xgeos
end


"""
    sort_configurations!(Xgeos::Vector{<:Matrix{<:Real}})

Sort the different configurations to achieve continuity when varying some extrinsic paramater.
"""
function sort_configurations!(Xgeos::Vector{<:Matrix{<:Real}})
    flip_geo = zero(Xgeos[1][1, :])

    for i in 1:(length(Xgeos)-1)
        Δσ = [angdist(Xgeos[i][1, 1:2], Xgeos[i+1][jj, 1:2]) for jj in 1:2]
        Δt = [abs(Xgeos[i][1, 3] - Xgeos[i+1][jj, 3]) for jj in 1:2]

        if (argmin(Δσ) != argmin(Δt))
            @warn "Δσ and Δt do not match for i = $i."
            flush(stdout)
        end

        # In this case the geodesics match. Continue
        if argmin(Δt) == 1
            continue
        end

        # Flip the array rows
        flip_geo .= Xgeos[i+1][1, :]
        Xgeos[i+1][1,:] = Xgeos[i+1][2,:]
        Xgeos[i+1][2,:] = flip_geo
    end

end


"""
    is_strictly_increasing(x::Union{Vector{<:Real}, LinRange{<:Real}})

Check if a vector is strictly increasing.
"""
function is_strictly_increasing(x::Union{Vector{<:Real}, LinRange{<:Real}})
    return all((x[i+1] - x[i]) > 0 for i in 1:length(x)-1)
end


"""
    solve_gshe(
        Xgeo::Vector{<:Real},
        geometry::Geometry{<:Real},
        ϵs::Union{Vector{<:Real}, LinRange{<:Real}};
        verbose::Bool=true,
    )

Find the GSHE solutions of a specific configuration and geodesic. The shape of the output
array is (s = ± 2, Nϵs, 4) where the last index stores the initial direction, time of
arrival and redshift.
"""
function solve_gshe(
    Xgeo::Vector{<:Real},
    geometry::Geometry{<:Real},
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}};
    verbose::Bool=true,
)
    @assert is_strictly_increasing(ϵs) "`ϵs` must be strictly increasing."
    Nϵs = length(ϵs)
    X = zeros(geometry.dtype, 2, Nϵs, 5)

    s = geometry.s
    for (i, ϵ) in enumerate(ϵs)
        if verbose
            @printf "%.2f%%, ϵ=%.2e\n" (i / Nϵs * 100) ϵ
            flush(stdout)
        end

        # A loop over the +- polarisation states
        for (j, sx) in enumerate([+s, -s])
            # Loop over the previously found solutions in reverse
            for k in reverse(1:i)
                # If previous GSHE solution available and is not NaN set it as
                # initial direction.
                if k > 1
                    p0 = X[j, k - 1, 1:2]
                    if ~any(isnan.(p0))
                        X[j, i, :] .= find_restricted_minimum(geometry, ϵ, sx, p0)
                        break
                    end
                end

                # For the first GSHE set the geodesic solution as initial direction
                if k == 1
                    X[j, i, :] .= find_restricted_minimum(geometry, ϵ, sx, Xgeo[1:2])
                end
            end
        end

    end

    if geometry.postproc_options.check_gshe_sols
        check_gshes!(X, Xgeo, geometry, ϵs)
    end

    return X
end


"""
    function solve_gshe(
        Xgeo::Matrix{<:Real},
        geometry::Geometry{<:Real},
        ϵs::Union{Vector{<:Real}, LinRange{<:Real}};
        verbose::Bool=true,
    )

Find the GSHE solutions of a specific configuration. Iterates over geodesics. The shape of
the output array is (Ngeodesics, s = ± 2, Nϵs, 4) where the last index stores the initial
direction, time of arrival and redshift.
"""
function solve_gshe(
    Xgeo::Matrix{<:Real},
    geometry::Geometry{<:Real},
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}};
    verbose::Bool=true,
)
    Nϵs = length(ϵs)
    Ngeos = size(Xgeo)[1]
    X = zeros(geometry.dtype, Ngeos, 2, Nϵs, 5)

    for n in 1:Ngeos
        if verbose
            println("n = $n")
            flush(stdout)
        end

        X[n, :, :, :] .= solve_gshe(Xgeo[n, :], geometry, ϵs; verbose=verbose)
    end

    return X
end


"""
    solve_gshes(
        Xgeos::Vector{<:Matrix{<:Real}},
        geometries::Vector{<:Geometry{<:Real}},
        ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
        configuration_verbose::Bool=true,
        perturbation_verbose::Bool=true,
    )

Find the GSHE solutions. Iterates over configurations. The shape of the output array
is (Nconfigurations, Ngeodesics, s = ± 2, Nϵs, 4).
"""
function solve_gshes(
    Xgeos::Vector{<:Matrix{<:Real}},
    geometries::Vector{<:Geometry{<:Real}},
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
    configuration_verbose::Bool=true,
    perturbation_verbose::Bool=true,
)
    Nconfs = length(Xgeos)
    @assert Nconfs === length(geometries) ("`Xgeos` and `geometries` must have the same length.")
    check_geometry_dtypes(geometries)
    Xgshes = Vector{Array{geometries[1].dtype, 4}}(undef, Nconfs)

    if configuration_verbose
        println("Solving GSHE for $Nconfs configurations.")
        flush(stdout)
    end

    # Shuffle workers jobs
    if Threads.nthreads() > 1
        iters = shuffle!([i for i in 1:Nconfs])
    else
        iters = 1:Nconfs
    end

    Threads.@threads for i in iters
        if configuration_verbose
            println("Solving GSHE for geometry $i/$Nconfs")
            flush(stdout)
        end
        Xgshes[i] = solve_gshe(Xgeos[i], geometries[i], ϵs; verbose=perturbation_verbose)
    end

    return Xgshes

end


"""
    Xgeos_to_array(Xgeos::Vector{<:Matrix{<:Real}})

Convert a vector of geodesic solutions to an array. The first index indexes the vector
elements.
"""
function Xgeos_to_array(Xgeos::Vector{<:Matrix{<:Real}})
    N = length(Xgeos)
    out = zeros(N, size(Xgeos[1])...)
    for i in 1:N
        out[i, :, :] = Xgeos[i][:, :]
    end

    return out
end


"""
    Xgshes_to_array(Xgshes::Vector{<:Array{<:Real, 4}})

Convert a vector of GSHE solutions to an array. The first index indexes the vector
elements.
"""
function Xgshes_to_array(Xgshes::Vector{<:Array{<:Real, 4}})
    N = length(Xgshes)
    out = zeros(N, size(Xgshes[1])...)
    for i in 1:N
        out[i, :, :, :, :] = Xgshes[i][:, :, :, :]
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
        fit_gshe_to_geo = fit_Δts(ϵs, Xgshe[n, :, :, :], Xgeo[n, :], geometry)
        for i in 1:2
            αs[n, i, :] .= fit_gshe_to_geo[i]["alpha"]
            βs[n, i, :] .= fit_gshe_to_geo[i]["beta"]
        end

        if fit_gshe_gshe
            # Calculate the GSHE to GSHE fit
            fit_gshe_to_gshe = fit_Δts(ϵs, Xgshe[n, :, :, :], geometry)
            αs[n, 3, :] = fit_gshe_to_gshe["alpha"]
            βs[n, 3, :] = fit_gshe_to_gshe["beta"]
        end
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
        α, β = fit_timing(ϵs, Xgeos[i, :, :], Xgshes[i, :, :, :, :], geometries[i];
                          fit_gshe_gshe=fit_gshe_gshe)
        αs[i, :, :, :] .= α
        βs[i, :, :, :] .= β
    end

    return αs, βs
end
