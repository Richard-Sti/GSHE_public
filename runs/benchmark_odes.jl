using GSHEIntegrator
using BenchmarkTools
using Profile


geometry = GSHEIntegrator.setup_geometry(
    rsource=3, θsource= π/2, ϕsource=0,
    robs=50, θobs=π/3, ϕobs=π,
    a=0.9);

p = [0.2, 0.3]
x0 = GSHEIntegrator.init_values(p, geometry)

@show x0

dx = zeros(7)

# geodesic_odes!(dx, x0, geometry)
#
# @show @btime geodesic_odes!(dx, x0, geometry)
#
# Profile.clear()
# for _ in 1:1000000
#     @profile geodesic_odes!(dx, x0, geometry)
# end
# Profile.print()

gshe_odes!(dx, x0, geometry, 0.01, 2)

@show @btime gshe_odes!(dx, x0, geometry, 0.01, 2)

Profile.clear()
for _ in 1:10000000
    @profile gshe_odes!(dx, x0, geometry, 0.01, 2)
end
Profile.print()


# GSHEIntegrator.