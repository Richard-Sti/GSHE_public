import BenchmarkTools: @btime, @benchmark;

start = time()
import Pkg: activate, build
activate("../GSHEIntegrator/.")
import GSHEIntegrator;

dur = time() - start

println("Finished loading in $dur s.")



geometry = GSHEIntegrator.setup_geometry(
    rsource=3, θsource= π/2, ϕsource=0,
    robs=50, θobs=π/3, ϕobs=π,
    a=0.9);


p = [0.2, 0.3]
x0 = GSHEIntegrator.init_values(p, geometry)


dx2 = zeros(7)


println("All finished.")


GSHEIntegrator.gshe_odes!(dx2, x0, geometry, 0.01, 2)

println(@benchmark GSHEIntegrator.gshe_odes!(dx2, x0, geometry, 0.11, 2))
# println(@benchmark zeros(7))
