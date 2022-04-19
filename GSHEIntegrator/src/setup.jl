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
    ode_options::ODESolverOptions=ODESolverOptions(),
    opt_options::OptimiserOptions=OptimiserOptions()
)
    source = SphericalCoords{dtype}(r=rsource, θ=θsource, ϕ=ϕsource)
    observer = SphericalCoords{dtype}(r=robs, θ=θobs, ϕ=ϕobs)
    return Geometry{dtype}(dtype=dtype, source=source, observer=observer, s=s, a=a,
                           ode_options=ode_options, opt_options=opt_options)
end


"""
    setup_geometries(
        dtype::DataType=Float64;
        rsource::Union{Vector{T}, LinRange{T}},
        θsource::Union{Vector{T}, LinRange{T}},
        ϕsource::Union{Vector{T}, LinRange{T}},
        robs::Union{Vector{T}, LinRange{T}},
        θobs::Union{Vector{T}, LinRange{T}},
        ϕobs::Union{Vector{T}, LinRange{T}},
        a::Union{Vector{T}, LinRange{T}},
        s::Integer=2,
        ode_options::ODESolverOptions=ODESolverOptions(),
        opt_options::OptimiserOptions=OptimiserOptions()
    ) where T <: Real

Setup a vector of geometries.
"""
function setup_geometries(
    dtype::DataType=Float64;
    rsource::Union{Vector{T}, LinRange{T}},
    θsource::Union{Vector{T}, LinRange{T}},
    ϕsource::Union{Vector{T}, LinRange{T}},
    robs::Union{Vector{T}, LinRange{T}},
    θobs::Union{Vector{T}, LinRange{T}},
    ϕobs::Union{Vector{T}, LinRange{T}},
    a::Union{Vector{T}, LinRange{T}},
    s::Integer=2,
    ode_options::ODESolverOptions=ODESolverOptions(),
    opt_options::OptimiserOptions=OptimiserOptions()
) where T <: Real
    geometries = Vector{Geometry{dtype}}()
    for rs in rsource, θs in θsource, ϕs in ϕsource, ro in robs, θo in θobs, ϕo in ϕobs, ai in a
        geo = setup_geometry(dtype;
            rsource=rs, θsource=θs, ϕsource=ϕs, robs=ro, θobs=θo, ϕobs=ϕo, a=ai, s=s,
            ode_options=ode_options, opt_options=opt_options)
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
    )

Find the geodesic initial direction and time of arrivals for a list of geometries.
"""
function solve_geodesics(
    geometries::Vector{<:Geometry{<:Real}},
    Nsols::Integer=2,
    verbose::Bool=true
)
    check_geometry_dtypes(geometries)
    dtype = geometries[1].dtype

    N = length(geometries)
    Xgeos = Vector{Matrix{dtype}}(undef, N)
    Threads.@threads for i in 1:N
        if verbose
            print("Solving geodesics for geometry $i/$N\n")
            flush(stdout)
        end
        Xgeos[i] = find_geodesic_minima(geometries[i], Nsols)
    end

    return Xgeos
end


"""
    is_strictly_increasing(x::Union{Vector{<:Real}, LinRange{<:Real}})

Check if a vector is strictly increasing.
"""
function is_strictly_increasing(x::Union{Vector{<:Real}, LinRange{<:Real}})
    return all((x[i+1] - x[i]) > 0 for i in 1:length(x)-1)
end


"""
    solve_gshe(Xinit::Array{<:Real, 3}, geometry::Geometry, ϵ::Real)

Find the s = ± |s| GSHE solutions for a configuration and its (typically 2) geodesics at a fixed
value of ϵ.
"""
function solve_gshe(Xinit::Array{<:Real, 3}, geometry::Geometry, ϵ::Real)
    Ngeos = size(Xinit)[2]  # Number of geodesics for this particular configuration
    X = zeros(geometry.dtype, 2, Ngeos, 4)

    for i in 1:Ngeos
        X[1, i, :] .= find_restricted_minimum(geometry, ϵ, geometry.s, Xinit[1, i, 1:2])
        X[2, i, :] .= find_restricted_minimum(geometry, ϵ, -geometry.s, Xinit[2, i, 1:2])
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

Find the GSHE initial conditions and time of arrival for a given configuration. Iterates
over the geodesics and ϵ. The shape of the output array is (Ngeodesics, s = ± 2, Nϵs, 4)
where the last index stores the initial direction, time of arrival and redshift.
"""
function solve_gshe(
    Xgeo::Matrix{<:Real},
    geometry::Geometry{<:Real},
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}};
    verbose::Bool=true,
)
    @assert is_strictly_increasing(ϵs) "`ϵs` must be strictly increasing."
    Nϵs = length(ϵs)
    Ngeos = size(Xgeo)[1]
    X = zeros(geometry.dtype, Ngeos, 2, Nϵs, 4)

    s = geometry.s
    for n in 1:Ngeos, (i, ϵ) in enumerate(ϵs)
        if verbose
            @printf "n=%d, %.2f%%, ϵ=%.2e\n" n (i / Nϵs * 100) ϵ
            flush(stdout)
        end

        if i > 1
            X[n, 1, i, :] .= find_restricted_minimum(geometry, ϵ, +s, X[n, 1, i - 1, 1:2])
            X[n, 2, i, :] .= find_restricted_minimum(geometry, ϵ, -s, X[n, 2, i - 1, 1:2])
        else
            X[n, 1, i, :] .= find_restricted_minimum(geometry, ϵ, +s, Xgeo[n, 1:2])
            X[n, 2, i, :] .= find_restricted_minimum(geometry, ϵ, -s, Xgeo[n, 1:2])
        end

    end

    if geometry.postproc_options.check_gshe_sols
        check_gshes!(X, Xgeo, geometry, ϵs)
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

Find the GSHE solutions. Iterates over configurations.
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

    Threads.@threads for i in 1:Nconfs
        if configuration_verbose
            print("Solving GSHE, geometry $i/$Nconfs\n")
            flush(stdout)
        end
        Xgshes[i] = solve_gshe(Xgeos[i], geometries[i], ϵs; verbose=perturbation_verbose)
    end

    return Xgshes

end


"""
    is_equally_spaced(x::Vector{<:Real})

Check whether a vector is equally spaced up to some default tolerance.
"""
function is_equally_spaced(x::Vector{<:Real})
    dx = [x[i+1] - x[i] for i in 1:length(x)-1]
    return all(isapprox(dx[1], dx[i]) for i in 2:length(dx))
end


"""
    check_gshes!(
        Xgshe::Array{<:Real, 4},
        Xgeo::Matrix{<:Real},
        geometry::Geometry,
        ϵs::Union{Vector{<:Real}, LinRange{<:Real}}
    )

Checks the continuity of the power law solution for the difference between GSHE
and a geodesic. Optionally recalculates odd solutions.
"""
function check_gshes!(
    Xgshe::Array{<:Real, 4},
    Xgeo::Matrix{<:Real},
    geometry::Geometry,
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}}
)
    log_ϵs = log10.(ϵs)
    @assert is_strictly_increasing(log_ϵs) "ϵ must be strictly increasing."
    if ~is_equally_spaced(log_ϵs)
        @warn "ϵs are not logarithimically spaced. Skipping checks, proceed carefully."
        return
    end
    flush(stdout)

    @unpack integration_error, average_tol, Ncorrect = geometry.postproc_options

    Ngeos = size(Xgeo)[1]  # Number of geodesic solutions for this configuration
    Nϵs = length(log_ϵs)

    for n in 1:Ngeos, s in 1:2, k in 1:(Ncorrect+ 1)
        y = abs.(Xgshe[n, s, :, 3] .- Xgeo[n, 3])
        y = y[y .> integration_error]

        outliers = Vector{Int64}()

        for i in 2:(Nϵs-1)
            mu = (y[i + 1] + y[i - 1]) / 2
            if abs(y[i] - mu)  > average_tol
                push!(outliers, i)
            end
        end

        # If no outliers exit
        if length(outliers) == 0
            continue
        end

        # Exit if too many attempts
        if k == Ncorrect + 1
            @warn ("Failed to recalculate outliers $outliers for geodesic $n, s=$s. "
                   *"Setting to NaN, either inspect the solutions or increase `normlinear_tol`.")
            flush(stdout)
            for j in outliers
                Xgshe[n, s, j, :] .= NaN
            end
            # Continue to the next upper level loop
            continue
        else
            @info "Detected outliers $outliers for geodesic $n, s=$s. Attempting to recalculate."
            flush(stdout)
        end

        good_gshes = [i for i in 1:Nϵs if ~(i in outliers)]
        # Ensure outliers are sorted
        sort!(outliers)
        for k in outliers
            # Get the previous good solution
            if k == 1
                p0 = Xgeo[n, 1:2]
            else
                p0 = Xgshe[n, s, argmin(abs.(good_gshes .- k)), 1:2]
            end

            Xgshe[n, s, k, :] .= find_restricted_minimum(
                geometry, ϵs[k], s == 1 ? geometry.s : -geometry.s, p0)
        end
    end

    return nothing
end


"""
    grid_evaluate(f::Function, x::T, y::T) where T <: Union{Vector{<:Real}, LinRange{<:Real}}

Evaluate function f(<:Vector{Real}) on a 2D grid.
"""
function grid_evaluate(f::Function, x::T, y::T) where T <: Union{Vector{<:Real}, LinRange{<:Real}}
    N = length(x) * length(y)
    grid = zeros(N, 2)
    Z = zeros(N)

    # Create a meshgrid
    k = 1
    for xi in x, yi in y
        grid[k, 1] = xi
        grid[k, 2] = yi
        k += 1
    end

    # Calculate the entries
    Threads.@threads for k in 1:N
        Z[k] = f(grid[k, :])
    end

    Z = transpose(reshape(Z, length(x), length(y)))
    return Z
end