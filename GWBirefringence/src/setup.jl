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
    source = Spherical_coords(r=r_source,
                              theta=theta_source,
                              phi=phi_source)
    observer = Spherical_coords(r=r_obs,
                                theta=theta_obs,
                                phi=phi_obs)
    params = Params(a=a, ϵ=eps, s=s)

    return Geometry(source=source, observer=observer, params=params)
end


"""
    setup_problem(
        geometry::GWBirefringence.Geometry,
        options::Options;
        geodesic::Bool=false
    )

Sets up the problem.
"""
function setup_problem(
    geometry::GWBirefringence.Geometry,
    options::Options;
#    use_gradients::Bool=false,
    geodesic::Bool=false
)
    # Initial vector
    init(p) = init_values(p, geometry, false)
    # Problem and callbacks
    if geodesic
        prob = ode_problem(geodesic_odes!, geometry)
    else
        prob = ode_problem(spinhall_odes!, geometry)
    end
    cb = get_callbacks(geometry)
    # Geodesic solver
    fsolver(p, save_everystep::Bool=false) = solve_geodesic(
                                                p, prob, cb, init,
                                                save_everystep=save_everystep)
    # Loss and minimizer
    floss(p, Xfound::Union{Vector{Vector{GWFloat}}, Nothing}) = loss(
                                                                p, Xfound,
                                                                fsolver,
                                                                geometry)


    problem = Problem(solve_geodesic=fsolver,
                      loss=floss)

#     if !use_gradients
#         fmin() = find_minimum(floss, NelderMead(), options)
#         problem = Problem(solve_geodesic=fsolver,
#                           loss=floss,
#                           find_min=fmin)
#     else
#         result = GradientResult(zeros(GWFloat, 3))
#         floss_gradient!(F, G, p) = loss_gradient!(F, G, p, result, floss)
#     
#         grad_fmin() = find_minimum(floss_gradient!,
#                               ConjugateGradient(manifold=Sphere()),
#                               options)
#         problem = Problem(solve_geodesic=fsolver,
#                           loss=floss,
#                           find_min=grad_fmin)
#     end
    
    return problem
end


"""
    solve_config!(
        prob::GWBirefringence.Problem,
        geometry::GWBirefringence.Geometry;
        Nminima::Int64=2
    )

Solves a particular configuration, writing results into `X` at step `step`.
"""
function solve_config(
    prob::GWBirefringence.Problem,
    geometry::GWBirefringence.Geometry;
    Nminima::Int64=2,
)

    Xplus = search_unique_minima(prob.find_min, Nminima)
    Xplus = timing_minima(prob.solve_geodesic, Xplus)

    geometry.params.s *= -1
    Xminus = search_unique_minima(prob.find_min, Nminima)
    Xminus  = timing_minima(prob.solve_geodesic, Xminus)
    geometry.params.s *= -1

    Xminus = match_Xminus(Xplus, Xminus)
    return Xplus, Xminus
end