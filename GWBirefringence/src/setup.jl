import DiffResults: GradientResult
import Optim: NelderMead, ConjugateGradient, Sphere


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
        options::Options,
        gradient_search::Bool=false
    )

Sets up the problem.
"""
function setup_problem(
    geometry::GWBirefringence.Geometry,
    options::Options,
    gradient_search::Bool=false
)
    # Initial vector
    init(p) = init_values(p, geometry, false)
    # Problem and callbacks
    prob = geodesic_ode_problem(geometry)
    cb = get_callbacks(geometry)
    # Geodesic solver
    fsolver(p, save_everystep::Bool=false) = solve_geodesic(p, prob, cb, init, save_everystep=save_everystep)
    # Loss and minimizer
    floss(p) = loss(p, fsolver, geometry)
    fmin() = find_minimum(floss, NelderMead(), options)
    
    problem = Problem(solve_geodesic=fsolver,
                      loss=floss,
                      find_min=fmin)

    
    if gradient_search == true
        result = GradientResult(zeros(GWFloat, 3))
        floss_gradient!(F, G, p) = loss_gradient!(F, G, p, result, loss)
    
        fmingrad() = find_minimum(floss_gradient!,
                                  ConjugateGradient(manifold=Sphere()),
                                  options)
        
        problem.find_gradmin = fmingrad
    end
    
    return problem
end
