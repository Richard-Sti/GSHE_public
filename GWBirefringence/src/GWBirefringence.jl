module GWBirefringence

import DoubleFloats: Double64
Float = Union{Float64, Double64, BigFloat}

export init_values
export params, spherical_coords, geometry
export pi0, geodesic_odes!
export Float, Double64

import Parameters: @with_kw, @unpack
using DifferentialEquations

include("./objects.jl")
include("./integrator.jl")
include("./coords.jl")
include("./kerr_geodesics.jl")

end
