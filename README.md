# Gravitational spin-Hall equations

## Powerlaw outlier TO-DO
- [ ] Outlier detection when fitting the power law. If mean residuals too large fit a slope = 2 line (optionally) and throw away the biggest outlier and try recalculating, repeat until errors hit some threshold. 
- [ ] In fitting set some lower sensitivity on $\Delta t$ (e.g. 1e-10)


## Plotting TO-DO
- [ ] Plot the intercept as the function of the source-lens-observer configuration
- [ ] When plotting time differences remove 0.0 (since log)

## Waveform TO-DO

- [ ] Model the waveform, likely compose backwards the frequency components as
$$
f(t) = \int \frac{\mathrm{d}\omega}{2\pi} e^{- i\omega\left[t + \Delta t(\omega)\right] \tilde{f}(\omega)}
$$
- [ ] The BH is initially shooting in diffferent diretions and we have an anisotropic source, different detector amplitudes?


## Other TO-DO
- [ ] Add support for varying other parameters.