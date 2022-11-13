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
Matching to the LIGO Lorentz-invariance constrains.
"""


from scipy.special import hyp2f1
from scipy import constants as consts


def D0_from_redshift(z, H0=67.9, Omega_matter=0.3065, Omega_Lambda=0.6935):
    r"""
    Calculate :math:`D_0` defined as in [1], Eq. 5. This is an effective
    distance measure.

    Paramaters
    ----------
    z : float or 1-dimensional array
        Redshift.
    H0 : float, optional
        Present time Hubble constant in units of
        :math:`\mathrm{km} \mathrm{s}^-1 \mathrm{Mpc}^{-1}`. By default 67.9.
    Omega_matter : float, optional
        Matter density parameter. By default 0.3065.
    Omega_Lambda : float. By default 0.6935.
        Dark energy density parameter.

    Returns
    -------
    D0 : float or 1-dimensional array
        Effective distance in :math:`\mathrm{Mpc}`.

    References
    ----------
    [1] Tests of General Relativity with the Binary Black Hole Signals from
    the LIGO-Virgo Catalog GWTC-1; LVK Collaboration
    """
    # Matter to Lambda ratio
    m2L = Omega_matter / Omega_Lambda

    # Define the hypergeometric function with some vals plugged in
    def _hyp2f1(x):
        return hyp2f1(-1/3, 1/2, 2/3, x)

    out = (1 + z) * _hyp2f1(-m2L) - _hyp2f1(-(1 + z)**3 * m2L)
    out *= (consts.c * 1e-3) / (H0 * Omega_Lambda**0.5)
    return out


def A0_from_sample(lambda_eff, z, Dlum, H0=67.9, Omega_matter=0.3065,
                   Omega_Lambda=0.6935):
    r"""
    Calculate :math:`A_0` following Section VII. of [1]. Note the typographic
    error in Eq. 4 of [1] as mentioned in Section VI. of [2].

    Paramaters
    ----------
    lambda_eff : float
        The effective wavelength used in sampling in :math:`\mathrm{m}`.
    z : float
        Redshift.
    Dlum : float
        Luminosity distance in :math:`\mathrm{Mpc}`.
    H0 : float, optional
        Present time Hubble constant in units of
        :math:`\mathrm{km} \mathrm{s}^-1 \mathrm{Mpc}^{-1}`. By default 67.9.
    Omega_matter : float, optional
        Matter density parameter. By default 0.3065.
    Omega_Lambda : float. By default 0.6935.
        Dark energy density parameter.

    Returns
    -------
    A0 : float
        The :math:`A_0` in units of :math:`\mathrm{peV}^2`.
    """
    D0val = D0_from_redshift(z, H0, Omega_matter, Omega_Lambda)  # In Mpc
    # Assuming `lambda_eff` comes in meters this should now be in joules^2
    A0 = (consts.h * consts.c / lambda_eff)**2 * (1 + z) * Dlum / D0val
    # Go from joules^2 to eV^2
    A0 /= consts.e**2
    # Go from eV^2 to peV^2
    A0 *= (1e12)**2
    return A0
