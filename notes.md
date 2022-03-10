# Gravitational spin-Hall equations

## Immediate TO-DO

- [ ] Perform arctan coordinate transformation when looking for perturbed solutions.
- [ ] Clear up geodesic nomenclature.


## Long TO-DO

- [ ] Model the waveform, likely as such that the time delay is calculated with respect to the geodesic.

$$
f(t) = \int \frac{\mathrm{d}\omega}{2\pi} e^{- i\omega\left[t + \Delta t(\omega)\right] \tilde{f}(\omega)}
$$


## Completed TO-DO
- [x] Remove the dependence on the Rinv
- [x] How to efficiently pass geometry etc. into the functions?
- [x] Begin looking for $s=\pm 2$ within some radius of its geodesic. Make sure that that $[x^\mu, p_i]$ initialisation works.
- [x] Have a look at how I do coordinate transformations. Can we avoid some memory allocations?