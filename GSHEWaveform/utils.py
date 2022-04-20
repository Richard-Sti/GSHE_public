import numpy
from scipy import constants as c
from scipy.interpolate import interp1d
from copy import deepcopy
# Append the solar mass
c.Msun = 1.989e30


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
    Rs = 2 * c.G * c.Msun * M / c.c**2
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
    return c.c**3 / (c.G * c.Msun) / M / f


def time_delay_analytical(f, M, alpha, beta):
    r"""
    Calculate the analytical time delay :math:`\Delta \tau(f)`.

    Arguments
    ---------
    f: float
        Frequency [Hz]
    M: float
        Background mass [Msun]
    alpha: float
        Power law proportionality constant [s]
    beta: float
        Power law slope.

    Returns
    -------
    time_delay: float
        Time delay [s]
    """
    return numpy.piecewise(
        f, [f <= 0, f > 0],
        [1e16, lambda x: alpha * epsilon_from_freq(x, M)**beta])


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
    return numpy.cos(arg) - 1j *numpy.sin(arg)


class GSHEtoGeodesicDelayInterpolator:
    """
    Interpolates the GSHE to geodesic time delay in log-log.

    Arguments
    ---------
    epsilons : 1-dimensional array
        Values of :math:`\epsilon` at which the time delay was calculated.
    Xgshe : 4-dimensional array
        Array containing the GSHE arrival times, as outputted from Julia.
    Xgeo : 3-dimensional array
        Array containing the geodesic arrival times, as outputted from Julia.
    n : int
        The geodesic index.
    s : int
        The polarisation index.
    M: float
        Background mass [Msun]
    """

    def __init__(self, epsilons, Xgshe, Xgeo, n, s, M):

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

