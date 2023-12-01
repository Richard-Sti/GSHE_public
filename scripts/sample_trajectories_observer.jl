"""
#!/usr/bin/env julia
# -*- coding: utf-8 -*-
#Created on Tue Oct 10 16:01:31 2023
#@author: miguel
# export JULIA_NUM_THREADS=4
"""


import BenchmarkTools: @btime, @benchmark;

import Pkg: activate, build
activate("../GSHEIntegrator/.")

using GSHEIntegrator

import Plots
using LaTeXStrings
using Measures
using NPZ
using EllipsisNotation

using Dates


#now many points to sample 
n_points = 5

rsource=5
θsource=0.5π
ϕsource=0
robs=25
θobs=0.5π
ϕobs=-π
akerr=0.99


Nsols = 2 




function generate_filename(base::String="GSHE", counter::Int=0)
    """
    generate_filename(base::String="trajectory", counter::Int=0)

    Generate a unique filename based on the current date and time.

    # Arguments
    - `base::String="trajectory"`: The base name for the file.
    - `counter::Int=0`: Counter to append to the filename if a file with the same name already exists.

    # Returns
    - `String`: A unique filename in the format of `base_date_time[_counter].jld2`

    # Examples
    ```julia
    generate_filename()  # Returns "GSHE_2023_10_03_14_30_00"
    generate_filename("experiment", 1)  # Returns "experiment_2023_10_03_14_30_01"
    """
    date_str = Dates.format(now(), "yyyy_mm_dd")
    time_str = Dates.format(now(), "HH_MM_SS")
    
    filename = "$(base)_$(date_str)_$(time_str)"
    
    # Check if file exists, if yes, append counter
    if isfile("$(filename).jld2") || counter > 0
        filename *= "_$(counter)"
    end
    
    # Increment counter if file with that name exists
    while isfile("$(filename).jld2")
        counter += 1
        filename = "$(base)_$(date_str)_$(time_str)_$(counter)"
    end
    
    return "$(filename)"
end




print("Sampling N=",n_points)


#use a file_lock to avoid problems
file_lock = ReentrantLock()

# Initialize an array of Dicts to store the results for each iteration
results = []

# Save the partial results array to a JLD2 or HDF5 file
using JLD2
generate_filename()
filename = "../data/trajectories/"*generate_filename("random_trajectories")
print(filename)

# for i in 1:n_points
Threads.@threads for i in 1:n_points
    θsrc = acos(1 - 2 * rand())
    θobs = acos(1 - 2 * rand())
    ϕobs = 2 * pi * rand()
    
    # Your existing code to set up geometry and solve...
    geometry = GSHEIntegrator.setup_geometry(
       rsource=rsource, θsource=θsrc, ϕsource=0,
       robs=robs, θobs=θobs, ϕobs=ϕobs,
       a=akerr,getmagnification=true)
    
    Xgeo = GSHEIntegrator.solve_initial(geometry, 0.0, Nsols)
    
    ϵ0=1e-3 
    ϵs = (10).^LinRange(-3, -1, 5)

    geometry_no_mag = GSHEIntegrator.setup_geometry(
        rsource=rsource, θsource=θsrc, ϕsource=0,
        robs=robs, θobs=θobs, ϕobs=ϕobs,
        a=akerr, getmagnification=false)
    
    
    Xgshe = GSHEIntegrator.solve_increasing(Xgeo, geometry_no_mag, ϵs; verbose=false);
    #Xeps = GSHEIntegrator.solve_initial(geometry_no_mag, ϵ0, Nsols)
    αs, βs = GSHEIntegrator.fit_timing(ϵs, Xgeo, Xgshe, geometry_no_mag; fit_gshe_gshe=true);
    
    # Create a dictionary to store the results of this iteration
    #TODO: add more quantities: INITIAL THETA_I FOR EACH TRAJECTORY (NOT JUST SOURCE LOCATION)
    result = Dict(
        "θ_src" => θsrc,
        "ϕ_src" => 0,
        "θ_obs" => θobs,
        "ϕ_obs" => ϕobs,
        "r_src" => rsource,
        "r_obs" => robs,
        "Xgeo" => Xgeo,
        "μ" => Xgeo[:,9],
        "Δt" => Xgeo[2,3] - Xgeo[1,3],
        "α" => αs,
        "β" => βs,
        # "Xeps" => Xeps,
        # "β" => (Xgeo[:,3] - Xeps[:,1,3]) / ϵ0^2,
        # "βLR" => (Xeps[:,1,3] - Xeps[:,2,3]) / ϵ0^3,
        # "ϵ0" => ϵ0,
        "ϵs" => ϵs,
        "a_kerr" => akerr,
        # ... add other quantities here
    )
    print(i," ")
    
    # Add this dictionary to the results array
    push!(results, result)
    
    # Lock the file for writing
    lock(file_lock)
    
    try
        @save filename*".jld2" results
    finally
        # Release the lock
        unlock(file_lock)
    end
end
