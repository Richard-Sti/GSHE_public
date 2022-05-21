# Gravitational spin-Hall effect

The gravitational spin-Hall effect (GSHE) describes the frequency and spin-dependent trajectories of wavepackets, such as gravitational-waves emitted from a binary black hole merger, propagating in strong gravitational fields as described in [1]. We calculate the GSHE-induced deviations from a null geodesic and the observed detector strain for a gravitational wavepacket.


## Things to be resolved later
- GW emitter anisotropy
- JuliaSymbolics.jl to generate high-performance code from symbolic expressions? But the current code (ODEs themselves) is already very fast. Had a brief look at the package but didn't make much sense out of it.
- Look into precompilation.
- Why does the code sometimes struggle to find solutions? Too many iterations near the horizon?
- Asymptotic behaviour far from source need better understood?


## References
[1] Andersson, Lars, Jérémie Joudioux, Marius A. Oancea, and Ayush Raj. "Propagation of polarized gravitational waves." Physical Review D 103, no. 4 (2021): 044053.