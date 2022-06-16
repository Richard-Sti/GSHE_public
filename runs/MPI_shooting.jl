import MPI
import NPZ: npzwrite, npzread
using TaskmasterMPI

import_start = time()
# Activate GSHE code
import Pkg: activate
activate("/mnt/zfsusers/rstiskalek/GSHE/GSHEIntegrator/.")
import GSHEIntegrator
import_end = time()


MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

if rank == 0
    println("Initialised GSHEIntegrator in $(import_end - import_start) s.")
end


config = Dict(
    :runID => "12",
    :cdir => "/mnt/extraspace/rstiskalek/gshe",
    :msg => "Geodesic shooting. Ignore observer angular position.",
    :varparam => [:dir1, :dir2],
    :ϵs => (10).^LinRange(-3, -1, 30),
    :rsource => 10.0,
    :θsource => 0.5π,
    :ϕsource => 0.0,
    :robs => 100.0,
    :θobs => π/2,
    :ϕobs => 1π,
    :dir1 => LinRange(-1, 1, 250),
    :dir2 => LinRange(-1, 1, 250),
    :from_shadow => true,
    :increasing_ϵ => false,
    :s => 2,
    :a => 0.99,
    :opt_options => GSHEIntegrator.OptimiserOptions(Ninit=51, Nconsec=30),
    :fit_gshe_gshe => false,
    :dtype => Float64
    )


if MPI.Comm_size(comm) == 1
    println("Exiting as the MPI size must be > 1.")
elseif rank == 0
    # Save stuff first
    println("Saving information."); flush(stdout)
    GSHEIntegrator.make_checkpointdir(config)
    GSHEIntegrator.save_config_info(config)
    npzwrite(joinpath(GSHEIntegrator.checkpointdir(config), "Epsilons.npy"), config[:ϵs])
    npzwrite(joinpath(GSHEIntegrator.checkpointdir(config), "dir1.npy"), config[:dir1])
    npzwrite(joinpath(GSHEIntegrator.checkpointdir(config), "dir2.npy"), config[:dir2])

    # Go on do the work
    println("Solving directions."); flush(stdout)
    tasks = []
    for i in 1:length(config[:dir1])
        for j in 1:length(config[:dir2])
            push!(tasks, [i, j])
        end
    end
    # Start the task delegator
    master_process(tasks, comm, verbose=true)
    # Once all tasks have been delegated sort the geodesics
    println("Collecting results."); flush(stdout)
    GSHEIntegrator.MPI_collect_shooting(config)
else
    f(task::Vector) = GSHEIntegrator.MPI_solve_shooting(task..., config)
    worker_process(f, comm)
end
