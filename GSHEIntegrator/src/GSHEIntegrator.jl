module GSHEIntegrator

# coords.jl
export azimuthal_angle, cartesian_to_spherical, cartesian_to_spherical!, spherical_to_cartesian,
    spherical_to_cartesian!, rvs_sphere, rvs_sphere_restricted, rvs_sphere_y, angdist,
    rotate_to_y, rotate_to_y!, rotate_from_y, rotate_from_y!, atan_transform, atan_invtransform,
    shadow2angle, angle2shadow
# grid.jl
export make_2dmesh, grid_evaluate_scalar, grid_evaluate_timing
# integrator.jl
export ffield_callback, horizon_callback, poles_callback, max_Δϕ, numloops, loops_callback,
    get_callbacks, init_values, solve_problem, solve_consecutive_problem, angular_bounds,
    angular_bounds_y, shadow_bounds, in_bounds, arrival_stats!, initial_loss, consecutive_loss,
    is_at_robs, obs_angdist, ode_problem, magnification
# io.jl
export save_geometry_info, save_config_info
# kerr_functions.jl
export initial_spatial_comomentum, time_comomentum, static_observer_proper_time, obs_frequency,
    obs_redshift, tetrad_boosting, derivative_tetrad_boosting, kerr_BL, ϕkilling
# kerr_trajectories.jl
export geodesic_odes!, gshe_odes!
# minimas.jl
export find_initial_minima, find_initial_minimum, getθmax, find_consecutive_minimum
# mpi_support.jl
export setup_geometry, Nconfigs, checkpointdir, make_checkpointdir, MPI_solve_configuration,
    MPI_sort_solutions, fit_timing, MPI_solve_shooting, MPI_collect_shooting
# objects.jl
export SphericalCoords, ODESolverOptions, OptimiserOptions, PostprocOptions, Geometry
# outliers.jl
export predict_llsq, residuals_llsq, R2_llsq, findoutlier_fixedslope, move_point!, find_outliers,
    check_solutions!, check_geodesics
# powerlaw.jl
export bootstrap_powerlaw, cut_below_integration_error, fit_Δts
# setup.jl
export setup_geometry, setup_geometries, check_geometry_dtypes, setup_initial_solver, setup_consecutive_solver,
    setup_initial_loss, setup_consecutive_loss, sort_configurations!, toarray, fit_timing
# shoot_timing.jl
export time_initial!, time_gshe, time_direction
# solver.jl
export solve_initial, is_strictly_increasing, solve_decreasing, solve_increasing, solve_full
# plotting.jl
export cartesiantrajectory, plotbh!, plot_start_end!



shadow_coords = [:shadow, :shadowpos]


import Parameters: @with_kw, @unpack
import Optim: NelderMead, Options, optimize
import DifferentialEquations: CallbackSet, ContinuousCallback, DiscreteCallback,
                              terminate!, remake, ODEProblem, solve, Vern9
import MultivariateStats: llsq
import Printf: @printf, @sprintf
import Random: shuffle!
import NPZ: npzwrite, npzread
using EllipsisNotation
import StatsBase: mean, std
import Clustering
import ForwardDiff: jacobian
import LinearAlgebra: det
import Plots
import Meshes

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
include("./mpi_support.jl")
include("./solver.jl")

end
