# Gravitational spin-Hall equations

## Powerlaw outlier TO-DO
- [x] Outlier detection when fitting the power law. If mean residuals too large fit a slope = 2 line (optionally) and throw away the biggest outlier and try recalculating, repeat until errors hit some threshold. 

- [ ] This doesn't workkkkkkkkkk!


## Plotting TO-DO
- [x] Write down tools to simply calculate several configurations
- [x] Decide whether to keep carrying on geometries or only a list of several base geometries and epsilons
- [x] Plot the intercept as the function of the source-lens-observer configuration
- [ ] When plotting time differences remove 0.0 (since log)

## Waveform TO-DO

- [ ] Model the waveform, likely compose backwards the frequency components as

$$
f(t) = \int \frac{\mathrm{d}\omega}{2\pi} e^{- i\omega\left[t + \Delta t(\omega)\right] \tilde{f}(\omega)}
$$

- [ ] The BH is initially shooting in diffferent diretions and we have an anisotropic source, different detector amplitudes? Understand why the polar angle $\rho$ appears to vary so much in some cases.


## Other TO-DO
- [x] Add support for varying other parameters.
- [ ] When looping over arrays stop assuming $s=\pm 2$ when summarising results.