# Numerical gravitational spin-Hall effect

The gravitational spin-Hall effect (GSHE) describes the frequency and spin-dependent trajectories of wavepackets, such as gravitational-waves emitted from a binary black hole merger, propagating in strong gravitational fields as described in [1]. In this  package, we calculate the GSHE-induced deviations from a null geodesic and the observed detector strain for a gravitational wavepacket as introduced in [2, 3].


## Installation
The package is split amongst three parts:
1. `GSHEIntegrator.jl`: Julia code for numerically solved the GSHE equations of motion for a wavepacket in a Kerr spacetime.
2. `GSHESymbolical`: Mathematica code for the symbolic derivation of the GSHE equations of motion, which are translated to Julia to be used in `GSHEIntegrator.jl`.
3. `GSHEWaveform`: Python code for the calculation of the effect of the GSHE on the detector strain.

The Julia and Python code can be installed following the instructions below, while the Mathematica code are just self-contained notebooks.





## License and Citation
If you use or find useful any of the code in this repository, please cite [2, 3].

```
Copyright (C) 2024 Richard Stiskalek
This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
```


## Contributors
- Richard Stiskalek (University of Oxford)
- Marius Oancea (University of Vienna)
- Miguel Zumalacarregui (Max Planck Institute for Gravitational Physics)


## Examples
...

## References
[1] Andersson, Lars, Jérémie Joudioux, Marius A. Oancea, and Ayush Raj. "Propagation of polarized gravitational waves." Physical Review D 103, no. 4 (2021): 044053.
[2] ...
[3] ...


## Things to be resolved later
- GW emitter anisotropy
- JuliaSymbolics.jl to generate high-performance code from symbolic expressions? But the current code (ODEs themselves) is already very fast. Had a brief look at the package but didn't make much sense out of it.
- Look into precompilation.
- Why does the code sometimes struggle to find solutions? Too many iterations near the horizon?
- Asymptotic behaviour far from source need better understood?

