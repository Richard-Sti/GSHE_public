# Numerical gravitational spin Hall equations


## TODO:


### Geodesic extrapolation
- [ ] Add support for the increase of t along the straight line
- [ ] Add support for the cross product norm to determine at which point the geodesic is straight (give some tolerance)


### Target hitting
- [ ] Calculate how close that straight line passes to a target location 
- [ ] Add some solver that minimises the impact parameter


### Documentation & Debugging
- [x] Document the code that does the translation from mathematica
- [x] Create a chain class
- [ ] Document the extrapolation stuff
- [ ] Debug Kerr
- [ ] Generalise the solver from the integrator, make it possible to switch the solvers (i.e. RK45 -> sympleptic, ...)


### Integration
- [ ] Add a sympleptic integrator (more suitable for integrating a Hamiltonian system)
