from copy import deepcopy
import numpy
from pycbc.waveform.utils import apply_fd_time_shift, fd_to_td

from .utils import mixing, linear_to_circular, circular_to_linear


def gshe_to_circular(hcirc_tilde, time_delay):
    r"""
    Apply GSHE correction to a null geodesic frequency-domain waveform in the
    circular basis.

    Arguments
    ---------
    hcirc_tilde : :py:class:`pycbc.types.frequencyseries.FrequencySeries`
        Frequency-domain circular basis waveform.
    time_delay : :py:func:`time_delay(f)`
        Observer time delay [s] of the particular circular polarisation whose
        only argument is frequeyncy [Hz]. Must act on arrays.

    Returns
    -------
    hcirc_tilde_gshe : :py:class:`pycbc.types.frequencyseries.FrequencySeries`
        Frequency-domain circular basis waveform with GSHE corrections applied.
    """
    freqs = hcirc_tilde.sample_frequencies.data
    dt = time_delay(freqs)
    # Deepcopy the waveform
    hcirc_tilde_gshe = deepcopy(hcirc_tilde)
    # Multiply its data points by the mixing exp^{2pi i f dt}
    hcirc_tilde_gshe.data *= mixing(freqs, dt)
    return hcirc_tilde_gshe


def gshe_to_linear(hptilde, hctilde, right_time_delay, left_time_delay):
    r"""
    Apply GSHE corrections to null geodesic linearly polarised frequency-domain
    waveforms.

    Arguments
    ---------
    hptilde : :py:class:`pycbc.types.frequencyseries.FrequencySeries`
        Frequency-domain plus polarisation waveform.
    hctilde : :py:class:`pycbc.types.frequencyseries.FrequencySeries`
        Frequency-domain cross polarisation waveform.
    right_time_delay : :py:func:`time_delay(f)`
        Observer time delay [s] of the right circular polarisation whose
        only argument is frequeyncy [Hz]. Must act on arrays.
    left_time_delay : :py:func:`time_delay(f)`
        Observer time delay [s] of the left circular polarisation whose
        only argument is frequeyncy [Hz]. Must act on arrays.

    Returns
    -------
    hptilde_gshe : :py:class:`pycbc.types.frequencyseries.FrequencySeries`
        Frequency-domain plus polarisation waveform with GSHE corrections.
    hctilde_gshe : :py:class:`pycbc.types.frequencyseries.FrequencySeries`
        Frequency-domain cross polarisation waveform with GSHE corrections.
    """
    hrtilde, hltilde = linear_to_circular(hptilde, hctilde)
    # Apply correctoins in the circular basis
    hrtilde_gshe = gshe_to_circular(hrtilde, right_time_delay)
    hltilde_gshe = gshe_to_circular(hltilde, left_time_delay)
    # Inverse transform back to the linear basis
    return circular_to_linear(hrtilde_gshe, hltilde_gshe)


def fd_to_td_fiducialshift(htilde, left_window=None, right_window=None,
                           tshift=-1):
    """
    Inverse Fourier transform a frequency-domain waveform to the time domain
    with a time shift to correct for :py:func:`pycbc.waveforms.utils.fd_to_td`
    terminating the time series at :math:`t=0`.

    Arguments
    ---------
    htilde : :py:class:`pycbc.types.frequencyseries.FrequencySeries`
        Frequency-domain waveform.
    left_window : tuple of float, optional
        A tuple giving the start and end frequency of the FD taper to apply on
        the left side. If None, no taper will be applied on the left.
    right_window : tuple of float, optional
        A tuple giving the start and end frequency of the FD taper to apply on
        the right side. If None, no taper will be applied on the right.
    tshift : float, optional
        Fiducial time shift. Initially shifts the waveform backwards before
        the IFT and then forward after. Must be negative, by default -1.

    Returns
    -------
    htilde : :py:class:`pycbc.types.timeseries.TimeSeries`
        Time-domain waveform.
    """
    if tshift > 0:
        raise ValueError("`tshift` must be less than or equal to 0. "
                         "Currently {}".format(tshift))
    # Initial backwards shift
    htilde = apply_fd_time_shift(htilde, tshift)
    h = fd_to_td(htilde, left_window=left_window, right_window=right_window)
    # Shift back forwards
    h.start_time -= tshift
    return h


def waveform_to_strain(hp, hc, Fp, Fc):
    """
    Turn plus and cross waveforms into strain given some detector response
    coefficients. Multiplise the polarisation waveform with its antenna
    response and sums over the two states.

    Arguments 
    ---------
    hp : :py:class:`pycbc.types.timeseries.TimeSeries`
        Time-domain plus polarisation waveform.
    hc : :py:class:`pycbc.types.timeseries.TimeSeries`
        Time-domain cross polarisation waveform.
    Fp : float
        Plus polarisation sensitivity.
    Fc : float
        Cross polarisation sensitivity.

    Returns
    -------
    strain : :py:class:`pycbc.types.timeseries.TimeSeries`
        Time-domain detector strain.
    """
    return hp * Fp + hc * Fc
