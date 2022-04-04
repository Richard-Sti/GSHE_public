from scipy import constants as c
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
    return alpha * epsilon(f, M)**beta

