module GWBirefringence

export init_values
export Params, Spherical_coords, Geometry
export pi0, geodesic_odes!
export GWFloat
export geodesic_ode_problem
export setup_geometry, setup_problem


import Parameters: @with_kw, @unpack
import Optim: NelderMead, ConjugateGradient, Options, optimize
import DifferentialEquations: CallbackSet, ContinuousCallback, DiscreteCallback,
                              terminate!, remake, ODEProblem, solve, Vern9
using LaTeXStrings
import Plots
import Meshes
import MultivariateStats: llsq
import Printf: @printf, @sprintf
import LinRegOutliers: @formula
import LinRegOutliers: createRegressionSetting, smr98
import DataFrames: DataFrame
import LinRegOutliers

include("./objects.jl")
include("./integrator.jl")
include("./coords.jl")
include("./kerr_functions.jl")
include("./kerr_trajectories.jl")
include("./minimas.jl")
include("./plotting.jl")
include("./setup.jl")
include("./powerlaw.jl")

end
