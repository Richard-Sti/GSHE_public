"""
    setup_geometry(;
        rsource::GWFloat,
        θsource::GWFloat,
        ϕsource::GWFloat,
        robs::GWFloat,
        θobs::GWFloat,
        ϕobs::GWFloat,
        a::GWFloat,
        eps::GWFloat,
        s::Int64=2
    )

Setup the geometry.
"""
function setup_geometry(;
    rsource::GWFloat,
    θsource::GWFloat,
    ϕsource::GWFloat,
    robs::GWFloat,
    θobs::GWFloat,
    ϕobs::GWFloat,
    a::GWFloat,
    eps::GWFloat,
    s::Int64=2
)
    source = Spherical_coords(r=rsource, θ=θsource, ϕ=ϕsource)
    observer = Spherical_coords(r=robs, θ=θobs, ϕ=ϕobs)
    params = Params(a=a, ϵ=eps, s=s)
    return Geometry(source=source, observer=observer, params=params)
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
    function solver(p::Vector{GWFloat}, save_everystep::Bool=false)
        solve_geodesic(p, prob, geometry, cb; save_everystep=save_everystep)
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
        p::Vector{GWFloat},
        pfound::Union{Vector{Vector{GWFloat}}, Nothing}=nothing,
    )
        return geodesic_loss(p, pfound, f, geometry)
    end
    return loss
end


"""
    setup_spinhall_solver(geometry::GWBirefringence.Geometry)

Setup the spin Hall trajectory solver for a given geometry without reference
frame rotations.
"""
function setup_spinhall_solver_norot(geometry::GWBirefringence.Geometry)
    # Get callbacks from upthere
    cb = get_callbacks(geometry)
    prob = ode_problem(spinhall_odes!, geometry)
    # Integrator function
    function solver(
        p::Vector{GWFloat},
        save_everystep::Bool=false
    )
        solve_geodesic(p, prob, geometry, cb; save_everystep=save_everystep)
    end
    return solver
end


"""
    setup_spinhall_solver(geometry::GWBirefringence.Geometry)

Setup the spin Hall trajectory solver for a given geometry including reference
frame rotations.
"""
function setup_spinhall_solver(geometry::GWBirefringence.Geometry)
    # Get callbacks from upthere
    cb = get_callbacks(geometry)
    prob = ode_problem(spinhall_odes!, geometry)
    # Integrator function
    function solver(
        p::Vector{GWFloat},
        pgeo::Vector{GWFloat};
        save_everystep::Bool=false
    )
        solve_spinhall(p, prob, geometry, cb, pgeo; save_everystep=save_everystep)
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
    function loss(
        p::Vector{GWFloat},
        pgeo::Vector{GWFloat},
        θmax::GWFloat
    )
        return spinhall_loss(p, pgeo, θmax, solver, geometry)
    end
    return loss
end


"""
    solve_perturbed_config(
        Xgeo::Matrix{Float64},
        geometry::Geometry,
        alg::NelderMead,
        options::Options;
        θmax0::Float64=0.025
        verbose::Bool=false
    )

Find the s = ± 2 spin-Hall perturbations for geodesics specified in `Xgeo`.
"""
function solve_perturbed_config(
    Xgeo::Matrix{Float64},
    geometry::Geometry,
    alg::NelderMead,
    options::Options;
    θmax0::Float64=0.025,
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
    vary_ϵ(ϵ::GWFloat, geometry::GWBirefringence.Geometry)

Copy geometry and replace its ϵ with a new value specified in the function
input.
"""
function vary_ϵ(ϵ::GWFloat, geometry::GWBirefringence.Geometry)
    new_geometry = copy(geometry)
    new_geometry.params.ϵ = ϵ
    return new_geometry
end