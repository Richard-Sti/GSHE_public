import MPI
import NPZ: npzwrite, npzread
using TaskmasterMPI

# Activate GSHE code
import_start = time()
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
    :runID => "02",
    :cdir => "/mnt/extraspace/rstiskalek/gshe",
    :msg => "Varying rsrc",
    :varparam => :rsource,
    :Nsols => 2,
    :ϵs => (10).^LinRange(-3, -1, 31),
    :rsource => LinRange(3, 35, 200),
#    :rsource => 4.0,
    :θsource => 0.5π,
    :ϕsource => 0.0,
    :robs => 100.0,
#    :θobs => sort!(push!(acos.(LinRange(-0.9, 0.9, 301)), π/2)),
    :θobs => 0.4π,
    # :ϕobs => LinRange(0.75π, 1.25π, 301),
    :ϕobs => 0.75π,
    :a => 0.99,
    :increasing_ϵ => true,
    :getmagnification => true,
    :opt_options => GSHEIntegrator.OptimiserOptions(Ninit=101),
    :ode_options => GSHEIntegrator.ODESolverOptions(no_loops=true),
    :fit_gshe_gshe => true,
    :dtype => Float64
    )

if MPI.Comm_size(comm) == 1
    println("Exiting as the MPI size must be > 1.")
elseif rank == 0
    println("Saving information."); flush(stdout)
    # Save stuff first
    GSHEIntegrator.make_checkpointdir(config)
    GSHEIntegrator.save_config_info(config)
    npzwrite(joinpath(GSHEIntegrator.checkpointdir(config), "Epsilons.npy"), config[:ϵs])
    npzwrite(joinpath(GSHEIntegrator.checkpointdir(config), "VaryParam.npy"), config[config[:varparam]])

    # Go on do the work
    println("Solving geodesics and GSHE."); flush(stdout)
    tasks = Vector(1:GSHEIntegrator.Nconfigs(config))
    # Start the task delegator
    master_process(tasks, comm, verbose=true)
    # Once all tasks have been delegated sort the geodesics
    println("Collecting results."); flush(stdout)
    GSHEIntegrator.MPI_sort_solutions(config)
    println("Fitting timing."); flush(stdout)
    GSHEIntegrator.fit_timing(config)
else
    f(index::Integer) = GSHEIntegrator.MPI_solve_configuration(index, config)
    worker_process(f, comm)
end
