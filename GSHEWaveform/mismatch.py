import numpy
from scipy.integrate import quad
from scipy.interpolate import interp1d
from warnings import warn

from .utils import mixing


def circular_mismatch(fhcirc, fmin, fmax, delay, minrelerr=1e-6):
    r"""
    Calculate the mismatch of a circular waveform.

    Arguments
    ---------
    fhcirc: :py:class:`pycbc.types.frequencyseries.FrequencySeries`
        Frequency-domain circular basis waveform.
    fmin: float
        Minimum frequency [Hz].
    fmax: float
        Maximum frequency [Hz].
    delay: :py:function
        Function whose sole argument is frequency [Hz] and returns the
        time delay [s].
    maxrelerr: float, optional
        The maximum relative error of the integration defined as the ratio
        between the integration error and its result. If sends a warning.

    Returns
    -------
    mismatch: float
        The mismatch of the circular waveform.
    """
    fs = fhcirc.sample_frequencies
    # Make sure we have the right frequency range
    m = (fs >= fmin) & (fs <= fmax)
    fs = fs[m]
    h = fhcirc.data[m]
    # Amplitude squared
    h2 = numpy.real(numpy.conj(h) * h)
    # Normalise, cancels in mismatch and helps numeric integration
    h2 /= numpy.max(h2)

    dtau = delay(fs)
    cos_mix = numpy.real(mixing(fs, dtau))

    num, errnum = quad(interp1d(fs, cos_mix * h2), fmin, fmax)
    den, errden = quad(interp1d(fs, h2), fmin, fmax)

    relerrden = errden / den
    relerrnum = errnum / num
    if relerrden > minrelerr or relerrnum > minrelerr:
        warn("Integ. error to result ratios are {} and {}. Proceed carefully."
             .format(relerrnum, relerrden))

    return 1 - num / den
