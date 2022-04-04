import numpy
from scipy import constants as c
from copy import deepcopy
c.Msun = 1.989e30


def epsilon(f, M):
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


def time_delay_analytical(f, M, alpha=1, beta=3):
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
    return numpy.piecewise(f, [f <= 0, f > 0],
                           [1e16, lambda x: alpha * epsilon(x, M)**beta])


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

