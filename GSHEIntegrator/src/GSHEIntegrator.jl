module GSHEIntegrator


export cartesian_to_spherical, cartesian_to_spherical!, spherical_to_cartesian,
    spherical_to_cartesian!, rvs_sphere, angdist
export get_callbacks, init_values, solve_geodesic, solve_gshe, geodesic_loss, gshe_loss,
    obs_angdist
export SphericalCoords, ODESolverOptions, OptimiserOptions, PostprocOptions, Geometry
export find_geodesic_minima, find_restricted_minima
export plot_initial_conditions!, plot_arrival_times!, plot_time_difference!,
    plot_geodesics!, plot_gshe_trajectories!, plot_blackhole!, plot_start_end!
export fit_Δts
export setup_geometry, setup_geometries, setup_geodesic_solver, setup_geodesic_loss,
    setup_gshe_solver, setup_gshe_loss, solve_geodesics, solve_gshe, solve_gshes,
    check_gshes!


shadow_coords = [:shadow, :shadowpos]

import Parameters: @with_kw, @unpack
import Optim: NelderMead, Options, optimize
import DifferentialEquations: CallbackSet, ContinuousCallback, DiscreteCallback,
                              terminate!, remake, ODEProblem, solve, Vern9
using LaTeXStrings
import Plots
import Meshes
import MultivariateStats: llsq
import Printf: @printf, @sprintf
import Random: shuffle!

include("./objects.jl")
include("./integrator.jl")
include("./coords.jl")
include("./kerr_functions.jl")
include("./kerr_trajectories.jl")
include("./minimas.jl")
include("./plotting.jl")
include("./setup.jl")
include("./powerlaw.jl")
include("./outliers.jl")
include("./shoot_timing.jl")
include("./grid.jl")
include("./io.jl")

end
