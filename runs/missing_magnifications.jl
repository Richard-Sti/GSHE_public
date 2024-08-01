import MPI
import NPZ: npzwrite, npzread
using TaskmasterMPI
using GSHEIntegrator

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)


runID = 13
geometry = GSHEIntegrator.setup_geometry(
   rsource=10, θsource=0.5π, ϕsource=0,
   robs=100, θobs=0.5π, ϕobs=π,
   a=0.99)

cdir = "/mnt/extraspace/rstiskalek/gshe/run_$runID"

Xgeos = npzread(joinpath(cdir, "Xgeos.npy"))
N = size(Xgeos, 1)
Xgshes = npzread(joinpath(cdir, "Xgshes.npy"))
ϵs = npzread(joinpath(cdir, "Epsilons.npy"))


if MPI.Comm_size(comm) == 1
    println("Exiting as the MPI size must be > 1.")
elseif rank == 0
    tasks = [i for i in 1:N if ~any(isnan.(Xgeos[i, 1:3]))]
    tasks_copy = copy(tasks)
    # Start the task delegator
    master_process(tasks, comm, verbose=true)
    # Once all tasks have been delegated sort the geodesics
    println("Collecting results."); flush(stdout)
    GSHEIntegrator.magnification_collect(Xgeos, Xgshes, tasks_copy, cdir)
else
    f(index::Integer) = GSHEIntegrator.magnification_save(Xgeos, Xgshes, index, geometry, ϵs, 2, cdir)
    worker_process(f, comm)
end
