# Copyright (C) 2022 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
Waveform mismatch calculation.
"""
from copy import deepcopy
from warnings import warn

import numpy
from pycbc.psd.analytical import flat_unity
from pycbc.filter import optimized_match
from scipy.integrate import quad
from scipy.interpolate import interp1d

from .utils import mixing


def circular_mismatch(fhcirc, fmin, fmax, delay):
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
    fhcirc = deepcopy(fhcirc)
    fhcirc_gshe = deepcopy(fhcirc)

    freqs = fhcirc.sample_frequencies.data

    psd = flat_unity(freqs.size, delta_f=fhcirc.delta_f, low_freq_cutoff=fmin)
    psd.data[:] = 1

    dt = delay(freqs)
    fhcirc_gshe.data *= mixing(freqs, dt)

    match = optimized_match(
        fhcirc_gshe, fhcirc, psd=psd, low_frequency_cutoff=fmin,
        high_frequency_cutoff=fmax)[0]

    return 1 - match


def circular_mismatch_nominim(fhcirc, fmin, fmax, delay, minrelerr=1e-6):
    fhcirc = deepcopy(fhcirc)
    fhcirc_shifted = deepcopy(fhcirc)

    fs = fhcirc_shifted.sample_frequencies
    m = (fs >= fmin) & (fs <= fmax)
    fs = fs[m]

    h_GO = fhcirc.data[m]
    h_GSHE = fhcirc_shifted.data[m]

    # Amplitude squared
    h2 = numpy.real(numpy.conj(h_GO) * h_GSHE)
    # Normalise, cancels in mismatch and helps numeric integration
    h2 /= numpy.max(h2)

    dtau = delay(fs)
    cos_mix = numpy.real(mixing(fs, dtau))

    num, errnum = quad(interp1d(fs, cos_mix * h2), fmin, fmax)
    den, errden = quad(interp1d(fs, h2), fmin, fmax)

    relerrden = errden / den
    relerrnum = errnum / num
    if relerrden > minrelerr or relerrnum > minrelerr:
        warn(f"Integ. error to result ratios are {relerrnum} and "
             "{relerrnum}. Proceed carefully.")

    return 1 - num / den
