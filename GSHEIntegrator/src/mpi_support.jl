"""
    setup_geometry(index::Integer, config::Dict{Symbol, Any})

Setup the system geometry from a configuration dictionary. Select the param that is being
varied at position `index`.
"""
function setup_geometry(index::Integer, config::Dict{Symbol, Any})
    # Source, observer and BH pars
    pars = [:rsource, :θsource, :ϕsource, :robs, :θobs, :ϕobs, :a]
    # Check all are present
    @assert all([par in keys(config) for par in pars])
    # Create the kwargs object. For the varied param select the appropriate position
    kwargs = Dict{Symbol, Any}()
    for par in pars
        if par == config[:varparam]
            kwargs[par] = config[par][index]
        else
            kwargs[par] = config[par]
        end
    end

    # Optional params
    optional_pars = [:opt_options, :ode_options, :postproc_options, :s, :direction_coords]
    for par in optional_pars
        if par in keys(config)
            kwargs[par] = config[par]
        end
    end

    return setup_geometry(config[:dtype]; kwargs...)
end


"""
    Nconfigs(config::Dict{Symbol, Any})

Return the number of configurations for given config dictionary where one param is being
varied.
"""
function Nconfigs(config::Dict{Symbol, Any})
    return length(config[config[:varparam]])
end


"""
    checkpointdir(config::Dict{Symbol, Any})

Join paths to get to the checkpoint directory.
"""
function checkpointdir(config::Dict{Symbol, Any})
    return joinpath(config[:cdir], "run_$(config[:runID])")
end


"""
    make_checkpointdir(config::Dict{Symbol, Any})
Check whether a checkpoint directory path exists, if not create it.
"""
function make_checkpointdir(config::Dict{Symbol, Any})
    cdir = checkpointdir(config)
    if ~isdir(cdir)
        mkdir(cdir)
    end
end


"""
    MPI_solve_configuration(index::Integer, config::Dict{Symbol, Any})

MPI solve geodesic and GSHE of the config file at `index` and write to the checkpointing
directory.
"""
function MPI_solve_configuration(index::Integer, config::Dict{Symbol, Any})
    geometry = setup_geometry(index, config)
    cdir = checkpointdir(config)
    Xgeo, Xgshe = solve_full(geometry, config[:ϵs], config[:increasing_ϵ], config[:Nsols]; perturbation_verbose=false)
    npzwrite(joinpath(cdir, "$(index)_Xgeo.npy"), Xgeo)
    npzwrite(joinpath(cdir, "$(index)_Xgshe.npy"), Xgshe)
end


"""
    MPI_sort_solutions(config::Dict{Symbol, Any})

Sort solutions from MPI. Loads them, sorts, and puts into a single file.
"""
function MPI_sort_solutions(config::Dict{Symbol, Any})
    N = Nconfigs(config)
    cdir = checkpointdir(config)
    # Read the geodesic solutions
    Xgeos = [npzread(joinpath(cdir, "$(i)_Xgeo.npy")) for i in 1:N]
    Xgshes = [npzread(joinpath(cdir, "$(i)_Xgshe.npy")) for i in 1:N]
    # Convert to arrays
    Xgeos = toarray(Xgeos)
    Xgshes = toarray(Xgshes)
    # Sort them
    GSHEIntegrator.sort_configurations!(Xgeos, Xgshes)
    # Write the new files
    npzwrite(joinpath(cdir, "Xgeos.npy"), Xgeos)
    npzwrite(joinpath(cdir, "Xgshes.npy"), Xgshes)

    # Delete the old intermediary files
    for i in 1:N
        rm(joinpath(cdir, "$(i)_Xgeo.npy"))
        rm(joinpath(cdir, "$(i)_Xgshe.npy"))
    end
end


"""
    fit_timing(config::Dict{Symbol, Any})

Fit timing of the collected MPI results and save to disk.
"""
function fit_timing(config::Dict{Symbol, Any})
    N = Nconfigs(config)
    cdir = checkpointdir(config)

    Xgeos = npzread(joinpath(cdir, "Xgeos.npy"))
    Xgshes = npzread(joinpath(cdir, "Xgshes.npy"))

    geometries = [setup_geometry(i, config) for i in 1:N]

    αs, βs = GSHEIntegrator.fit_timing(config[:ϵs], Xgeos, Xgshes, geometries;
                                       fit_gshe_gshe=config[:fit_gshe_gshe])
    npzwrite(joinpath(cdir, "alphas.npy"), αs)
    npzwrite(joinpath(cdir, "betas.npy"), βs)
end


"""
    MPI_solve_shooting(i::Integer, j::Integer, config::Dict{Symbol, Any})

Solve a shooting problem in the direction given by `i` and `j`th indices of the direction
vectors.
"""
function MPI_solve_shooting(i::Integer, j::Integer, config::Dict{Symbol, Any})
    # Unpack either k_x and k_y or ψ and ρ
    x = config[:dir1][i]
    y = config[:dir2][j]

    if config[:from_shadow] && (x^2 + y^2) > 1
        Xgeo = fill(NaN, 6)
        Xgshe = fill(NaN, length(config[:ϵs]), 6)
    else
        geometry = GSHEIntegrator.setup_geometry(-1, config)
        Xgeo, Xgshe = GSHEIntegrator.time_direction(
            [x, y], geometry, config[:s], config[:ϵs], config[:increasing_ϵ], config[:from_shadow];
            verbose=false)
    end
    # Write everything down
    npzwrite(joinpath(GSHEIntegrator.checkpointdir(config), "$(i)_$(j)_Xgeo.npy"), Xgeo)
    npzwrite(joinpath(GSHEIntegrator.checkpointdir(config), "$(i)_$(j)_Xgshe.npy"), Xgshe)
end


"""
    MPI_collect_shooting(config::Dict{Symbol, Any})

Collect the MPI shooting results and fit α and β.
"""
function MPI_collect_shooting(config::Dict{Symbol, Any})
    # Unpack either k_x and k_y or ψ and ρ
    xs = config[:dir1]
    ys = config[:dir2]

    cdir = GSHEIntegrator.checkpointdir(config)
    # Get their lengths
    Nx = length(xs)
    Ny = length(ys)
    directions = fill(NaN, Nx * Ny, 2)
    Xgeos = fill(NaN, Nx * Ny, 6)
    Xgshes = fill(NaN, Nx * Ny, length(config[:ϵs]), 6)

    k = 1
    for i in 1:Nx
        for j in 1:Ny
            directions[k, 1] = xs[i]
            directions[k, 2] = ys[i]
            Xgeos[k, ..] .= npzread(joinpath(cdir, "$(i)_$(j)_Xgeo.npy"))
            Xgshes[k, ..] .= npzread(joinpath(cdir, "$(i)_$(j)_Xgshe.npy"))
            k += 1
        end
    end

    npzwrite(joinpath(cdir, "Xgeos.npy"), Xgeos)
    npzwrite(joinpath(cdir, "Xgshes.npy"), Xgshes)

    # Fit α and β
    geometry = GSHEIntegrator.setup_geometry(-1, config)
    αs, βs = GSHEIntegrator.fit_timing(config[:ϵs], Xgeos, Xgshes, geometry)

    npzwrite(joinpath(cdir, "alphas.npy"), αs)
    npzwrite(joinpath(cdir, "betas.npy"), βs)

    # Remove intermediary results
    for i in 1:Nx, j in 1:Ny
        rm(joinpath(cdir, "$(i)_$(j)_Xgeo.npy"))
        rm(joinpath(cdir, "$(i)_$(j)_Xgshe.npy"))
    end

end