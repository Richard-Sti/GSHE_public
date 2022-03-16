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
    vary_ϵ(ϵ::Real, geometry::GWBirefringence.Geometry)

Copy geometry and replace its ϵ with a new value specified in the function input.
"""
function vary_ϵ(ϵ::Real, geometry::GWBirefringence.Geometry)
    new_geometry = copy(geometry)
    new_geometry.params.ϵ = ϵ
    return new_geometry
end