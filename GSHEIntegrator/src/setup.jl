"""
    setup_geometry(;
        rsource::Real,
        θsource::Real,
        ϕsource::Real,
        robs::Real,
        θobs::Real,
        ϕobs::Real,
        a::Real,
        ϵ::Real,
        s::Integer=2
    )

Setup the geometry.
"""
function setup_geometry(
    type::DataType=Float64;
    rsource::Real,
    θsource::Real,
    ϕsource::Real,
    robs::Real,
    θobs::Real,
    ϕobs::Real,
    a::Real,
    ϵ::Real,
    s::Integer=2
)
    source = Spherical_coords{type}(r=rsource, θ=θsource, ϕ=ϕsource)
    observer = Spherical_coords{type}(r=robs, θ=θobs, ϕ=ϕobs)
    params = Params{type}(a=a, ϵ=ϵ, s=s)
    return Geometry{type}(source=source, observer=observer, params=params, type=type)
end


"""
    setup_geometries(
        type::DataType=Float64;
        rsource::Union{Vector{T}, LinRange{T}},
        θsource::Union{Vector{T}, LinRange{T}},
        ϕsource::Union{Vector{T}, LinRange{T}},
        robs::Union{Vector{T}, LinRange{T}},
        θobs::Union{Vector{T}, LinRange{T}},
        ϕobs::Union{Vector{T}, LinRange{T}},
        as::Union{Vector{T}, LinRange{T}},
        s::Integer=2
    ) where T<: Real

Setup a vector of base geometries.
"""
function setup_geometries(
    type::DataType=Float64;
    rsource::Union{Vector{T}, LinRange{T}},
    θsource::Union{Vector{T}, LinRange{T}},
    ϕsource::Union{Vector{T}, LinRange{T}},
    robs::Union{Vector{T}, LinRange{T}},
    θobs::Union{Vector{T}, LinRange{T}},
    ϕobs::Union{Vector{T}, LinRange{T}},
    as::Union{Vector{T}, LinRange{T}},
    s::Integer=2
) where T<: Real
    base_geometries = Vector{Geometry{type}}()
    for rs in rsource, θs in θsource, ϕs in ϕsource, ro in robs, θo in θobs, ϕo in ϕobs, a in as
        geo = setup_geometry(type;
            rsource=rs, θsource=θs, ϕsource=ϕs, robs=ro, θobs=θo, ϕobs=ϕo, a=a, ϵ=0.01, s=s)
        push!(base_geometries, geo)
    end
    base_geometries
end


"""
    vary_ϵ(ϵ::Real, geometry::Geometry)

Copy geometry and replace its ϵ with a new value specified in the function input.
"""
function vary_ϵ(ϵ::Real, geometry::Geometry)
    new_geometry = copy(geometry)
    new_geometry.params.ϵ = ϵ
    return new_geometry
end


"""
    check_geometry_types(geometries::Vector{<:Geometry{<:Real}})

Check that each geometry has the same data type.
"""
function check_geometry_types(geometries::Vector{<:Geometry{<:Real}})
    dtype = geometries[1].type
    @assert all([dtype == geo.type for geo in geometries]) "All geometry data types must be the same."
end

"""
    setup_geodesic_solver(geometry::Geometry)

Setup the geodesic solver for a given geometry.
"""
function setup_geodesic_solver(geometry::Geometry)
    # Get callbacks from upthere
    cb = get_callbacks(geometry)
    # ODEProblem
    prob = ode_problem(geodesic_odes!, geometry)
    # Integrator function
    function solver(p::Vector{<:Real}, save_everystep::Bool=false;
                    reltol::Real=1e-14, abstol::Real=1e-14)
        solve_geodesic(p, prob, geometry, cb;
                       save_everystep=save_everystep, reltol=reltol, abstol=abstol)
    end
    return solver
end


"""
    setup_geodesic_loss(geometry::Geometry)

Setup the geodesic loss function for a given geometry.
"""
function setup_geodesic_loss(geometry::Geometry)
    # Loss function, define with two methods
    f = setup_geodesic_solver(geometry)
    function loss(
        p::Vector{<:Real},
        pfound::Union{Vector{<:Vector{<:Real}}, Nothing}=nothing,
    )
        return geodesic_loss(p, pfound, f, geometry)
    end
    return loss
end


"""
    setup_spinhall_solver(geometry::Geometry)

Setup the spin Hall trajectory solver for a given geometry without reference frame
rotations.
"""
function setup_spinhall_solver_norot(geometry::Geometry)
    # Get callbacks from upthere
    cb = get_callbacks(geometry)
    prob = ode_problem(spinhall_odes!, geometry)
    # Integrator function
    function solver(p::Vector{<:Real}, save_everystep::Bool=false;
                    reltol::Real=1e-14, abstol::Real=1e-14)
        solve_geodesic(p, prob, geometry, cb;
                       save_everystep=save_everystep, reltol=reltol, abstol=abstol)
    end
    return solver
end


"""
    setup_spinhall_solver(geometry::Geometry)

Setup the spin Hall trajectory solver for a given geometry including reference frame
rotations.
"""
function setup_spinhall_solver(geometry::Geometry)
    # Get callbacks from upthere
    cb = get_callbacks(geometry)
    prob = ode_problem(spinhall_odes!, geometry)
    # Integrator function
    function solver(p::Vector{<:Real}, pgeo::Vector{<:Real};
                    save_everystep::Bool=false, reltol::Real=1e-14, abstol::Real=1e-14)
        solve_spinhall(p, prob, geometry, cb, pgeo;
                       save_everystep=save_everystep, reltol=reltol, abstol=abstol)
    end
    return solver
end


"""
    setup_spinhall_loss(geometry::Geometry)

Setup the spin Hall trajectory loss function for a given geometry.
"""
function setup_spinhall_loss(geometry::Geometry)
    solver = setup_spinhall_solver(geometry)
    # Loss function, define with two methods
    function loss(p::Vector{<:Real}, pgeo::Vector{<:Real}, θmax::Real)
        return spinhall_loss(p, pgeo, θmax, solver, geometry)
    end
    return loss
end


"""
    solve_geodesics(
        geometries::Vector{<:Geometry{<:Real}},
        alg::NelderMead,
        options::Options;
        Nsols::Integer=2,
        verbose::Bool=true
    )
Find the geodesic solutions for a list of geometries.
"""
function solve_geodesics(
    geometries::Vector{<:Geometry{<:Real}},
    alg::NelderMead,
    options::Options;
    Nsols::Integer=2,
    verbose::Bool=true
)
    check_geometry_types(geometries)
    dtype = geometries[1].type

    N = length(geometries)
    Xgeos = Vector{Matrix{dtype}}(undef, N)
    Threads.@threads for i in 1:N
        if verbose
            print("Solving geodesics for geometry $i/$N\n")
            flush(stdout)
        end

        Xgeos[i] = find_geodesic_minima(geometries[i], alg, options; Nsols=Nsols)
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
    solve_gshe(
        Xgeo::Matrix{<:Real},
        geometry::Geometry,
        alg::NelderMead,
        options::Options;
        θmax0::Real=0.025
        verbose::Bool=false
    )

Find the s = ± |s| GSHE solutions for a configuration and its (typically 2) geodesics at a fixed
value of ϵ.
"""
function solve_gshe(
    Xinit::Array{<:Real, 3},
    geometry::Geometry,
    alg::NelderMead,
    options::Options;
    θmax0::Real=0.025,
    verbose::Bool=false
)
    Nsols = size(Xinit)[2]
    X = zeros(geometry.type, 2, Nsols, 4)
    
    for i in 1:Nsols
        if verbose
            println("Iteration $i")
        end
        X[1, i, :] .= find_restricted_minimum(
            geometry, Xinit[1, i, 1:2], alg, options; θmax0=θmax0, Nmax=50)
        # Flip polarisation sign
        geometry.params.s *= -1
        X[2, i, :] .= find_restricted_minimum(
            geometry, Xinit[1, i, 1:2], alg, options; θmax0=θmax0, Nmax=50)
        geometry.params.s *= -1
    end
    return X
end


"""
    solve_gshe(
        Xgeo::Matrix{<:Real},
        base_geometry::Geometry{<:Real},
        ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
        alg::NelderMead,
        options::Options;
        θmax0::Real=0.025,
        verbose=true,
        normlinear_tol::Real=1e-1,
        integration_error::Real=1e-12,
        Nmax::Integer=10,
        check_sols::Bool=true
    )

Find the s = ± |s| GSHE trajectories for a given geodesic. Iterates over ϵ values.
Checks whether the ϵ dependence is sensible.
"""
function solve_gshe(
    Xgeo::Matrix{<:Real},
    base_geometry::Geometry{<:Real},
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
    alg::NelderMead,
    options::Options;
    θmax0::Real=0.015,
    verbose=true,
    normlinear_tol::Real=1e-1,
    integration_error::Real=1e-12,
    Nmax::Integer=10,
    check_sols::Bool=true
)
    @assert is_strictly_increasing(ϵs) "`ϵs` must be strictly increasing."
    N = length(ϵs)
    Nsols = size(Xgeo)[1]
    Xspinhall = zeros(base_geometry.type, N, Nsols, 2, 4)
    geometries = [vary_ϵ(ϵ, base_geometry) for ϵ in ϵs]

    # Initial direction where to search. We assume to be searching in increasing values of
    # ϵ and iteratively update this to be the new found solution for previous ϵ.
    Xinit = zeros(base_geometry.type, 2, Nsols, 2)
    for i in 1:Nsols, s in 1:2
        Xinit[s, i, :] .= Xgeo[i, 1:2]
    end

    for (i, geometry) in enumerate(geometries)
        if verbose
            @printf "%.2f%%, ϵ=%.2e\n" (i / N *100) geometry.params.ϵ
            flush(stdout)
        end

        Xspinhall[i, :, : ,:] .= solve_gshe(Xinit, geometry, alg, options; θmax0=θmax0)
        # Update Xinit
        Xinit .= Xspinhall[i, :, :, 1:2]
        
    end
    # Check we have no strange outliers
    if check_sols
        check_perturbed_config!(Xspinhall, Xgeo, geometries, alg, options;
            θmax0=θmax0, normlinear_tol=normlinear_tol, integration_error=integration_error,
            Nmax=Nmax)
    end

    return Xspinhall
end


"""
    solve_gshes(
        Xgeos::Vector{<:Matrix{<:Real}},
        geometries::Vector{<:Geometry{<:Real}},
        ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
        alg::NelderMead,
        options::Options;
        θmax0::Real=0.025,
        verbose::Bool=true,
        normlinear_tol::Real=1e-1,
        integration_error::Real=1e-12,
        Nmax::Integer=10,
        check_sols::Bool=true
    )
Find the s = ± |s| GSHE solutions. Iterates over configurations.
"""
function solve_gshes(
    Xgeos::Vector{<:Matrix{<:Real}},
    geometries::Vector{<:Geometry{<:Real}},
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
    alg::NelderMead,
    options::Options;
    θmax0::Real=0.025,
    verbose::Bool=true,
    normlinear_tol::Real=1e-1,
    integration_error::Real=1e-12,
    Nmax::Integer=10,
    check_sols::Bool=true
)
    @assert length(Xgeos) === length(geometries) "`Xgeos` and `geometries` must have the same length."
    @assert is_strictly_increasing(ϵs) "`ϵs` must be strictly increasing."
    check_geometry_types(geometries)
    dtype = geometries[1].type

    Xspinhalls = Vector{Array{dtype, 4}}(undef, length(Xgeos))
    Ngeo = length(geometries)

    Threads.@threads for i in 1:Ngeo
        if verbose
            print("Solving GSHE, geometry $i/$Ngeo\n")
            flush(stdout)
        end
        Xspinhalls[i] = solve_gshe(Xgeos[i], geometries[i], ϵs, alg, options;
            θmax0=θmax0, verbose=false, normlinear_tol=normlinear_tol,
            integration_error=integration_error, Nmax=Nmax, check_sols=check_sols)
    end

    return Xspinhalls

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
    check_perturbed_config!(
        Xspinhall::Array{<:Real, 4},
        Xgeo::Matrix{<:Real},
        geometries::Vector{<:Geometry{<:Real}},
        alg::NelderMead,
        options::Options;
        θmax0::Real=0.025,
        normlinear_tol::Real=1e-1,
        integration_error::Real=1e-12,
        Nmax::Integer=10
    )

Check the outliers of the Δt - ϵ relation between the s = ± 2 polarisations. Picks out
the outliers with `LinRegOutliers.smr98` and checks whether they are above
`normlinear_tol`. If yes attempts to recalculate it for `Nmax` attempts. If no solution
is found replaces with NaNs.
"""
function check_perturbed_config!(
    Xspinhall::Array{<:Real, 4},
    Xgeo::Matrix{<:Real},
    geometries::Vector{<:Geometry{<:Real}},
    alg::NelderMead,
    options::Options;
    θmax0::Real=0.025,
    normlinear_tol::Real=1e-1,
    integration_error::Real=1e-12,
    Nmax::Integer=10
)
    ϵs = [geo.params.ϵ for geo in geometries]
    log_ϵs = log10.(ϵs)

    @assert is_strictly_increasing(log_ϵs) "ϵ must be strictly increasing."
    if ~is_equally_spaced(log_ϵs)
        @warn "ϵs are not logarithimically spaced. Skipping checks, proceed carefully."
        return
    end
    flush(stdout)

    Nsols = size(Xgeo)[1]
    Nϵs = length(log_ϵs)

    for igeo in 1:Nsols, s in 1:2, n in 1:(Nmax + 1)
        y = abs.(Xspinhall[:, s, igeo, 3] .- Xgeo[igeo, 3])
        y = y[y .> integration_error]

        outliers = Vector{Int64}()

        for i in 2:(Nϵs-1)
            mu = (y[i + 1] + y[i - 1]) / 2
            if abs(y[i] - mu) / ϵs[i] > normlinear_tol
                push!(outliers, i)
            end
        end

        # If no outliers exit
        if length(outliers) == 0
            continue
        end

        # Exit if too many attempts
        if n == Nmax + 1
            @warn ("Failed to recalculate outliers $outliers for igeo=$igeo, s=$s. "
                   *"Setting to NaN, either inspect the solutions or increase `normlinear_tol`.")
            flush(stdout)
            for k in outliers
                Xspinhall[k, s, igeo, :] .= NaN
            end
            # Continue to the next upper level loop
            continue
        else
            @info "Detected outliers $outliers for igeo=$igeo, s=$s. Attempting to recalculate."
            flush(stdout)
        end

        good_gshes = [i for i in 1:Nϵs if ~(i in outliers)]
        # Ensure outliers are sorted
        sort!(outliers)
        for k in outliers
            # Get the previous good solution
            if k == 1
                p0 = Xgeo[igeo, 1:2]
            else
                p0 = Xspinhall[argmin(abs.(good_gshes .- k)), s, igeo, 1:2]
            end

            # s = 2 corresponds to the negative polarisation
            s == 2 ? geometries[k].params.s *= -1 : nothing 
            Xspinhall[k, s, igeo, :] .= find_restricted_minimum(
                geometries[k], p0, alg, options; θmax0=θmax0, Nmax=50)
            s == 2 ? geometries[k].params.s *= -1 : nothing 
        end
    end

    return nothing 
end