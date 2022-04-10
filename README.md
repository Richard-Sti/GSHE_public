# Gravitational spin-Hall effect

The gravitational spin-Hall effect (GSHE) describes the frequency and spin-dependent trajectories of wavepackets, such as gravitational-waves emitted from a binary black hole merger, propagating in strong gravitational fields as described in [1]. We calculate the GSHE-induced deviations from a null geodesic and the observed detector strain for a gravitational wavepacket. 


## TO-DO
- [ ] When looping over arrays stop assuming $s=\pm 2$ when summarising results.
- [ ] Add an example notebook to calculate the deviations.
- [ ] Clean up some older code and think about Kerr-Schild coordinates


## Things to be resolved later
- GW emitter anisotropy
- JuliaSymbolics.jl to generate high-performance code from symbolic expressions? But the current code (ODEs themselves) is already very fast. Had a brief look at the package but didn't make much sense out of it.


## References
[1] Andersson, Lars, Jérémie Joudioux, Marius A. Oancea, and Ayush Raj. "Propagation of polarized gravitational waves." Physical Review D 103, no. 4 (2021): 044053.