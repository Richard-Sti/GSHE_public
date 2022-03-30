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
    base_geometries = Vector{GWBirefringence.Geometry{type}}()
    for rs in rsource, θs in θsource, ϕs in ϕsource, ro in robs, θo in θobs, ϕo in ϕobs, a in as
        geo = GWBirefringence.setup_geometry(type;
            rsource=rs, θsource=θs, ϕsource=ϕs, robs=ro, θobs=θo, ϕobs=ϕo, a=a, ϵ=0.01, s=s)
        push!(base_geometries, geo)
    end
    base_geometries
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
    setup_spinhall_solver(geometry::GWBirefringence.Geometry)

Setup the spin Hall trajectory solver for a given geometry without reference frame
rotations.
"""
function setup_spinhall_solver_norot(geometry::GWBirefringence.Geometry)
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
    setup_spinhall_solver(geometry::GWBirefringence.Geometry)

Setup the spin Hall trajectory solver for a given geometry including reference frame
rotations.
"""
function setup_spinhall_solver(geometry::GWBirefringence.Geometry)
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
    setup_spinhall_loss(geometry::GWBirefringence.Geometry)

Setup the spin Hall trajectory loss function for a given geometry.
"""
function setup_spinhall_loss(geometry::GWBirefringence.Geometry)
    solver = setup_spinhall_solver(geometry)
    # Loss function, define with two methods
    function loss(p::Vector{<:Real}, pgeo::Vector{<:Real}, θmax::Real)
        return spinhall_loss(p, pgeo, θmax, solver, geometry)
    end
    return loss
end


"""
    solve_perturbed_config(
        Xgeo::Matrix{<:Real},
        geometry::Geometry,
        alg::NelderMead,
        options::Options;
        θmax0::Real=0.025
        verbose::Bool=false
    )

Find the s = ± 2 spin-Hall perturbations for geodesics specified in `Xgeo`.
"""
function solve_perturbed_config(
    Xgeo::Matrix{<:Real},
    geometry::Geometry,
    alg::NelderMead,
    options::Options;
    θmax0::Real=0.025,
    verbose::Bool=false
)
    Nsols = size(Xgeo)[1]

    X = zeros(2, Nsols, 4)
    
   for i in 1:Nsols
        if verbose
            println("Iteration $i")
        end
        X[1, i, :] .= GWBirefringence.find_restricted_minimum(
            geometry, Xgeo[i, 1:2], alg, options; θmax0=θmax0, Nmax=50)
        geometry.params.s *= -1
        X[2, i, :] .= GWBirefringence.find_restricted_minimum(
            geometry, Xgeo[i, 1:2], alg, options; θmax0=θmax0, Nmax=50)
        geometry.params.s *= -1
   end
   return X
end


"""
    solve_perturbed_config(
        Xgeo::Matrix{<:Real},
        base_geometry::Geometry{<:Real},
        ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
        alg::NelderMead,
        options::Options;
        θmax0::Real=0.025,
        verbose=true,
        residuals_tolerance::Real=1e-2,
        integration_error::Real=1e-12,
        Nmax::Integer=10,
    )

Find the s = ± 2 spin-Hall perturbations for geodesics specified in `Xgeo`. Varies the
geometry.
"""
function solve_perturbed_config(
    Xgeo::Matrix{<:Real},
    base_geometry::Geometry{<:Real},
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
    alg::NelderMead,
    options::Options;
    θmax0::Real=0.025,
    verbose=true,
    residuals_tolerance::Real=1e-2,
    integration_error::Real=1e-12,
    Nmax::Integer=10,
)
    N = length(ϵs)
    Nsols = size(Xgeo)[1]
    Xspinhall = zeros(base_geometry.type, N, Nsols, 2, 4)
    geometries = [vary_ϵ(ϵ, base_geometry) for ϵ in ϵs]

    for (i, geometry) in enumerate(geometries)
        if verbose
            @printf "%.2f%%, ϵ=%.2e\n" (i / N *100) geometry.params.ϵ
            flush(stdout)
        end

        Xspinhall[i, :, : ,:] .= solve_perturbed_config(
            Xgeo, geometry, alg, options; θmax0=θmax0)
    end
    # Check we have no strange outliers
    check_perturbed_config!(Xspinhall, Xgeo, geometries, alg, options;
        θmax0=θmax0, residuals_tolerance=residuals_tolerance, integration_error=integration_error,
        Nmax=Nmax)

    return Xspinhall
end


"""
    check_perturbed_config!(
        Xspinhall::Array{<:Real, 4},
        Xgeo::Matrix{<:Real},
        geometries::Vector{<:Geometry{<:Real}},
        alg::NelderMead,
        options::Options;
        θmax0::Real=0.025,
        residuals_tolerance::Real=1e-2,
        integration_error::Real=1e-12,
        Nmax::Integer=10
    )

Check the outliers of the Δt - ϵ relation between the s = ± 2 polarisations. Picks out
the outliers with `LinRegOutliers.smr98` and checks whether they are above
`residuals_tolerance`. If yes attempts to recalculate it for `Nmax` attempts. If no solution
is found replaces with NaNs.
"""
function check_perturbed_config!(
    Xspinhall::Array{<:Real, 4},
    Xgeo::Matrix{<:Real},
    geometries::Vector{<:Geometry{<:Real}},
    alg::NelderMead,
    options::Options;
    θmax0::Real=0.025,
    residuals_tolerance::Real=1e-2,
    integration_error::Real=1e-12,
    Nmax::Integer=10
)
    log_ϵs = log10.([geo.params.ϵ for geo in geometries])
    Nsols = size(Xgeo)[1]
    df = DataFrame(x=log_ϵs, y=log_ϵs)
    formula = @formula y ~ x


    for igeo in 1:Nsols, j in 1:(Nmax + 1)
        y = abs.(Xspinhall[:, 1, igeo, 3] - Xspinhall[:, 2, igeo, 3])
        x, y = cut_below_integration_error(log_ϵs, y, integration_error)

        df.x .= x
        df.y .= log10.(y)
        reg = createRegressionSetting(formula, df)
        outliers= smr98(reg)["outliers"]

        # If no outliers exit
        if length(outliers) == 0
            break
        end

        # Check the outliers residuals
        # Fit LLSQ on non-outliers
        mask = .~[(i in outliers) for i in 1:length(log_ϵs)]
        fit = llsq(df.x[mask], df.y[mask])
        # Calculate residuals
        res = abs.(df.y[.~mask] .- (fit[1] .* df.x[.~mask] .+ fit[2]))
        # Rule out small residuals
        outliers = outliers[res .> residuals_tolerance]

        # If all these outliers below tolerance exit the check
        if length(outliers) == 0
            break
        end

        # If reached Nmax exit check
        if j > Nmax
            @warn "Failed to recalculate $(length(outliers)) outliers. Setting to NaN."
            flush(stdout)
            for k in outliers, s in [2, -2]
                Xspinhall[k, s == 2 ? 1 : 2, igeo, :] .= NaN
            end
            break
        else
            @info "Detected $(length(outliers)) outlier. Recalculating."
            flush(stdout)
        end

        for k in outliers, s in [2, -2]
            s < 0 ? geometries[k].params.s *= -1 : nothing 

            Xspinhall[k, s == 2 ? 1 : 2, igeo, :] .= GWBirefringence.find_restricted_minimum(
                geometries[k], Xgeo[igeo, 1:2], alg, options; θmax0=θmax0, Nmax=50)

            s < 0 ? geometries[k].params.s *= -1 : nothing 
        end
    end

    return nothing 
end


"""
    vary_ϵ(ϵ::Real, geometry::GWBirefringence.Geometry)

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
    solve_geodesics_from_geometries(
        geometries::Vector{<:Geometry{<:Real}},
        alg::NelderMead,
        options::Options;
        Nsols::Integer=2,
        verbose::Bool=true
    )
Find the geodesic solutions for a list of geometries.
"""
function solve_geodesics_from_geometries(
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

        Xgeos[i] = GWBirefringence.find_minima(geometries[i], alg, options; Nsols=Nsols)
    end

    return Xgeos
end


"""
    solve_perturbed_config(
        Xgeos::Vector{<:Matrix{<:Real}},
        geometries::Vector{<:Geometry{<:Real}},
        ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
        alg::NelderMead,
        options::Options;
        θmax0::Real=0.025,
        verbose::Bool=true,
        residuals_tolerance::Real=1e-2,
        integration_error::Real=1e-12,
        Nmax::Integer=10
    )
Find the s = ± 2 perturbed solutions for each geodesic.
"""
function solve_perturbed_config(
    Xgeos::Vector{<:Matrix{<:Real}},
    geometries::Vector{<:Geometry{<:Real}},
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
    alg::NelderMead,
    options::Options;
    θmax0::Real=0.025,
    verbose::Bool=true,
    residuals_tolerance::Real=1e-2,
    integration_error::Real=1e-12,
    Nmax::Integer=10
)
    @assert length(Xgeos) === length(geometries) "`Xgeos` and `geometries` must have the same length."
    check_geometry_types(geometries)
    dtype = geometries[1].type

    Xspinhalls = Vector{Array{dtype, 4}}(undef, length(Xgeos))
    Ngeo = length(geometries)

    Threads.@threads for i in 1:Ngeo
        if verbose
            print("Solving perturbations for geometry $i/$Ngeo\n")
            flush(stdout)
        end
        Xspinhalls[i] = solve_perturbed_config(Xgeos[i], geometries[i], ϵs, alg, options;
            θmax0=θmax0, verbose=false, residuals_tolerance=residuals_tolerance,
            integration_error=integration_error, Nmax=Nmax)
    end

    return Xspinhalls

end