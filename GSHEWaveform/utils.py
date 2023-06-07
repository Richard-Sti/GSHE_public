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
Utility functions for primarily handling units.
"""
from copy import deepcopy

import matplotlib as mpl
import numpy
from scipy import constants as c
from scipy.interpolate import interp1d

Msun = 1.989e30  # kg


def coordinate_time_to_seconds(t, M):
    r"""
    Convert coordinate time (:math:`R_s=2`, :math:`c=1`)  to seconds by
    multiplying it by a factor of :math:`R_s / c / 2`.

    Arguments
    ---------
    t : float
        Coordinate time [natural units].
    M : float
        Background object mass [Msun].

    Returns
    -------
    t : float
        Time [s].
    """
    Rs = 2 * c.G * Msun * M / c.c**2
    return t * Rs / c.c / 2


def epsilon_from_freq(f, M):
    r"""
    Calculate the perturbation strength :math:`\epsilon`.

    Arguments
    ---------
    f: float
        Frequency [Hz]
    M:  float
        Background mass [Msun].

    Returns
    -------
    epsilon : float
        Perturbation strength.
    """
    return c.c**3 / (c.G * Msun) / M / f


def M_from_epsfreq(eps, f):
    r"""
    Calculate the background mass :math:`M` in units of solar mass from
    :math:`\epsilon` and :math:`f` in units of seconds. This is the inverse of
    :py:func:`epsilon_from_freq`.

    Arguments
    ---------
    eps: float
        Perturbation strength parameter.
    f: float
        Frequency [Hz].

    Returns
    -------
    M: float
        Background mass [Msun].
    """
    return c.c**3 / (c.G * Msun) / (eps * f)


def time_delay_analytical(f, M, alpha, beta):
    r"""
    Calculate the analytical time delay :math:`\Delta \tau(f)`.

    Arguments
    ---------
    f: float
        Frequency [Hz]
    M: float
        Background mass [Msun]
    beta: float
        Power law proportionality constant [s]
    alpha: float
        Power law slope.

    Returns
    -------
    time_delay: float
        Time delay [s]
    """
    # Dimension-full beta in seconds
    dimbeta = coordinate_time_to_seconds(beta, M)

    def delay(x):
        return dimbeta * epsilon_from_freq(x, M)**alpha

    if isinstance(f, (float, int)):
        return delay(f)
    # Lists have to be converted to arrays
    if isinstance(f, list):
        f = numpy.asarray(f, dtype=float)
    return numpy.piecewise(f, [f <= 0, f > 0], [1e16, delay])


def linear_to_circular(plus, cross):
    """
    Convert PyCBC series from linear to circular polarisation.

    Input type:
        T = pycbc.types.frequencyseries.FrequencySeries
            or pycbc.types.frequencyseries.TimeSeries

    Arguments
    ---------
    plus: T
        Plus polarisation.
    cross: T
        Cross polarisation.

    Returns
    -------
    right: T
        Right polarisation.
    Left: T
        Left polarisation.
    """
    right = deepcopy(plus)
    left = deepcopy(cross)

    right.data = (plus.data + 1j*cross.data)/numpy.sqrt(2)
    left.data = (plus.data - 1j*cross.data)/numpy.sqrt(2)
    return right, left


def circular_to_linear(right, left):
    """
    Convert PyCBC series from circular to linear polarisation.

    Input type:
        T = pycbc.types.frequencyseries.FrequencySeries
            or pycbc.types.frequencyseries.TimeSeries

    Arguments
    ---------
    right : T
        Right circular polarisation.
    left : T
        Left ciruclar polarisation.

    Returns
    -------
    plus : T
        Plus linear polarisation.
    cross : T
        Cross linear polarisation.
    """
    plus = deepcopy(right)
    cross = deepcopy(left)

    plus.data = (right.data + left.data)/numpy.sqrt(2)
    cross.data = -1j * (right.data - left.data)/numpy.sqrt(2)
    return plus, cross


def mixing(freqs, time_delay):
    r"""
    Calculate the mixing factor

    .. math::

        exp^{-2\pi i f \Delta \tau (f)}.

    Arguments
    ---------
    freqs: float
        Frequency [Hz]
    time_delay: float
        Time delay [s]

    Returns
    -------
    mixing: float
        Mixing factor.
    """
    arg = 2 * numpy.pi * freqs * time_delay
    return numpy.cos(arg) - 1j * numpy.sin(arg)


class GSHEtoGeodesicDelayInterpolator:
    r"""
    Interpolates the GSHE to geodesic time delay in log-log.

    Arguments
    ---------
    epsilons : 1-dimensional array
        Values of :math:`\epsilon` at which the time delay was calculated.
    Xgshe : 2- or 4-dimensional array
        Array containing the GSHE arrival times, as outputted from Julia. If
        both `n` and `s` are not `None` then must be 4-dimensional.
    Xgeo : 1- or 3-dimensional array
        Array containing the geodesic arrival times, as outputted from Julia.
        If both `n` and `s` are not `None` then must be 4-dimensional.
    n : int
        The geodesic index.
    s : int
        The polarisation index.
    M: float
        Background mass [Msun]
    """

    def __init__(self, epsilons, Xgshe, Xgeo, M, n=None, s=None):
        if n is None and s is None:
            dt = Xgshe[:, 2] - Xgeo[2]
        else:
            dt = Xgshe[n, s, :, 2] - Xgeo[n, 2]
        # Get the sign and check all equal
        sign = numpy.sign(dt)
        assert numpy.alltrue(sign[0] == sign[1:])
        self.sign = sign[0]
        # Get the interpolator, we live dangerously and allow extrapolation
        # since we expect to fit straight lines.
        self.finterp = interp1d(numpy.log(epsilons), numpy.log(numpy.abs(dt)),
                                fill_value="extrapolate", bounds_error=False)
        self.M = M

    def delay_from_epsilon(self, epsilon):
        return self.sign * numpy.exp(self.finterp(numpy.log(epsilon)))

    def __call__(self, f, units="seconds"):
        """The interpolated delay."""
        if units not in ["seconds", "natural"]:
            raise ValueError("Allowed units choices are `seconds` and "
                             "`natural`.")
        eps = epsilon_from_freq(f, self.M)
        delay = self.delay_from_epsilon(eps)

        if units == "seconds":
            delay = coordinate_time_to_seconds(delay, self.M)

        if isinstance(f, numpy.ndarray):
            delay[f <= 0] = self.sign * 1e16

        if isinstance(f, (float, int)):
            if f == 0:
                delay = self.sign * 1e16
        return delay


def setmplstyle(fpath=None):
    """
    Set the matplotlib style. If `fpath` is `None` uses the default style.

    Arguments
    ---------
    fpath: str, optional
        Path to the style text file.
    """
    if fpath is None:
        mpl.rcParams.update(mpl.rcParamsDefault)
    else:
        mpl.style.use(fpath)


def ylabel_withoffset(ax, label):
    """
    Draw the y-label of the given axis with its offset.

    Arguments
    ---------
    ax: py:class:`matplotlib.axes._subplots.AxesSubplot`
        Matplotlib axis.
    label: str
        They y-axis label.
    """
    ax.yaxis.offsetText.set_visible(False)
    offset = ax.yaxis.get_major_formatter().get_offset()
    ax.set_ylabel(r"{} {}".format(label, offset))
