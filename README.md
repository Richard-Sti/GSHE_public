# Gravitational spin-Hall equations

## Immediate TO-DO
- [ ] Clean up the plotting modules.
- [ ] Add capabilities to fit the power law $f(\epsilon) = \alpha~\epsilon^\beta$. Can this be done analytically?
- [ ] Add support for varying other parameters.

## Long TO-DO

- [ ] Model the waveform, likely as such that the time delay is calculated with respect to the geodesic.

$$
f(t) = \int \frac{\mathrm{d}\omega}{2\pi} e^{- i\omega\left[t + \Delta t(\omega)\right] \tilde{f}(\omega)}
$$


## Completed TO-DO
- [x] Clean up the interface.
- [x] Start saving the arrival time on the optimiser level
- [x] Clear up geodesic nomenclature.
- [x] Perform arctan coordinate transformation when looking for perturbed solutions.
- [x] Remove the dependence on the Rinv
- [x] How to efficiently pass geometry etc. into the functions?
- [x] Begin looking for $s=\pm 2$ within some radius of its geodesic. Make sure that that $[x^\mu, p_i]$ initialisation works.
- [x] Have a look at how I do coordinate transformations. Can we avoid some memory allocations?
- [x] Start searching within small $\theta_{\rm max}$ and if no solution found gradually increase.