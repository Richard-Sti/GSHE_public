module GSHEIntegrator

# coords.jl
export cartesian_to_spherical, cartesian_to_spherical!, spherical_to_cartesian,
    spherical_to_cartesian!, rvs_sphere, rvs_sphere_y, angdist,
    rotate_from_y, rotate_from_y!, atan_transform, atan_invtransform,
    shadow2angle
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
    MPI_sort_solutions, fit_timing, MPI_solve_shooting, MPI_collect_shooting, magnification_save,
    magnification_collect
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
import MultivariateStats: llsq  # TODO remove dependence
import DifferentialEquations: CallbackSet, ContinuousCallback, DiscreteCallback,
                              terminate!, remake, ODEProblem, solve, Vern9
import Clustering: kmeans, nclusters, assignments, counts
import Random: shuffle!
import NPZ: npzwrite, npzread
using EllipsisNotation
import ForwardDiff: jacobian
# import Plots
# import Meshes

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
include("./mpi_support.jl")
include("./solver.jl")


################################################################################
#                               Statistics                                     #
################################################################################


"""
    mean(x::Vector{<:Real})

Compute the mean of a vector of real numbers.
"""
function mean(x::Vector{<:Real})
    return sum(x)/length(x)
end

"""
    std(x::Vector{<:Real})
Compute the standard deviation of a vector of real numbers.
"""
function std(x::Vector{<:Real})
    mu = mean(x)
    return sqrt((sum((x[i] - mu)^2 for i in 1:length(x)) / (length(x)-1)))
end


################################################################################
#                                  I/O                                         #
################################################################################


"""
    save_geometry_info(cdir::String, geometry::Geometry, msg::String)

Save information about geometry.
"""
function save_geometry_info(cdir::String, geometry::Geometry, msg::String)
    fpath = joinpath(cdir, "Description.txt")
    open(fpath, "w") do f
        println(f, msg)
        println(f, "Source:")
        println(f, geometry.source)
        println(f, "Observer:")
        println(f, geometry.observer)
        println(f, "BH spin")
        println(f, "a = $(geometry.a)")
        println(f, "ODE Options")
        println(f, geometry.ode_options)
        println(f, geometry.opt_options)
    end
end


"""
    save_config_info(config::Dict{Symbol, Any})

Save information about the configuration file.
"""
function save_config_info(config::Dict{Symbol, Any})
    fpath = joinpath(checkpointdir(config), "Description.txt")
    open(fpath, "w") do f
        for (key, value) in config
            println(f, "$key:")
            println(f, value)
            println(f, "\n")
        end
    end
end





end
