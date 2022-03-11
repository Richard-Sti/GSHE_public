import DiffResults: GradientResult
import Optim: Sphere


"""
    setup_geometry(;
        r_source::GWFloat,
        theta_source::GWFloat,
        phi_source::GWFloat,
        r_obs::GWFloat,
        theta_obs::GWFloat,
        phi_obs::GWFloat,
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


function solve_perturbed_config(
    Xgeo::Matrix{Float64},
    geometry::Geometry,
    alg::NelderMead,
    options::Options,
    θmax::Float64=0.15;
    verbose::Bool=false
)
    Nsols = size(Xgeo)[1]

    X = zeros(2, Nsols, 3)
    
   for i in 1:Nsols
        if verbose
            println("Iteration $i")
        end
        X[1, i, :] .= GWBirefringence.find_restricted_minimum(
            geometry, Xgeo[i, 1:2], θmax, alg, options, Nmax=500)
        geometry.params.s *= -1
        X[2, i, :] .= GWBirefringence.find_restricted_minimum(
            geometry, Xgeo[i, 1:2], θmax, alg, options, Nmax=500)
        geometry.params.s *= -1
   end
   return X
end


function vary_ϵ(ϵ::Float64, geometry::GWBirefringence.Geometry)
    new_geometry = copy(geometry)
    new_geometry.params.ϵ = ϵ
    return new_geometry
end



# """
#     solve_config!(
#         prob::GWBirefringence.Problem,
#         geometry::GWBirefringence.Geometry;
#         Nminima::Int64=2
#     )
# 
# Solves a particular configuration, writing results into `X` at step `step`.
# """
# function solve_config(
#     prob::GWBirefringence.Problem,
#     geometry::GWBirefringence.Geometry;
#     Nminima::Int64=2,
# )
# 
#     Xplus = search_unique_minima(prob.find_min, Nminima)
#     Xplus = timing_minima(prob.solve_geodesic, Xplus)
# 
#     geometry.params.s *= -1
#     Xminus = search_unique_minima(prob.find_min, Nminima)
#     Xminus  = timing_minima(prob.solve_geodesic, Xminus)
#     geometry.params.s *= -1
# 
#     Xminus = match_Xminus(Xplus, Xminus)
#     return Xplus, Xminus
# end