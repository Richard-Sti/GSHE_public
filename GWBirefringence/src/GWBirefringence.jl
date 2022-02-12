module GWBirefringence

import DoubleFloats: Double64
GWFloat = Float64

export init_values
export Params, Spherical_coords, Geometry
export pi0, geodesic_odes!
export GWFloat, Double64
export geodesic_ode_problem

import Parameters: @with_kw, @unpack
using DifferentialEquations

include("./objects.jl")
include("./integrator.jl")
include("./coords.jl")
include("./kerr_geodesics.jl")
include("./minimas.jl")
include("./plotting.jl")

end
