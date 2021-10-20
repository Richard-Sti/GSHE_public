# MIT License
#
# Copyright (c) 2021 Richard Stiskalek
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import sympy
from einsteinpy.symbolic import (MetricTensor, GenericVector)


def make_schwarzschild(return_orthonormal_tetrad=False, r_s=None, c=None):
    """
    Creates a symbolic spherical-coordinates Schwarzschild metric implemented
    in EinsteinPy.

    Arguments
    ---------
    return_orthonormal_tetrad : bool, optional
        Whether to return the orthonormal tetrad coordinates. By default False.
    r_s : float or sympy.Symbol, optional.
        Schwarzschild radius value to be substituted into the metric. By
        default substitutes :math:`r_s`.
    c : float or sympy.Symbol, optional.
        Speed of light value to be substituted into the metric. By default
        substitutes :math:`c`.

    Returns
    -------
    metric : `einsteinpy.symbolic.metric.MetricTensor`
        The desired metric tensor (`ll` configuration).
    tetrad : list of `einstenpy.symbolic.GenericVector`
        The orthonormal tetrad vectors `e0`, `e1`, ...
    """

    # Coordinate system
    t, theta, phi = sympy.symbols('t theta phi', real=True)
    r = sympy.Symbol('r', real=True, positive=True)
    coords = (t, r, theta, phi)
    # Metric variables
    if r_s is None:
        r_s = sympy.Symbol('r_s', real=True, positive=True)
    if c is None:
        c = sympy.Symbol('c', real=True, positive=True)
    # Metric
    R = 1 - r_s / r
    metric = sympy.diag(- R * c**2, 1 / R, r**2,
                        r**2 * sympy.sin(theta)**2).tolist()
    metric = MetricTensor(metric, coords, name='Schwarzschild')
    # Add the determinant by hand
    metric.det = - c**2 * r**4 * sympy.sin(theta)**2
    metric.sqrt_neg_det = c * r**2 * sympy.sin(theta)
    if return_orthonormal_tetrad:
        e0 = GenericVector([1 / R**0.5, 0, 0, 0], coords, config='u',
                           parent_metric=metric, name='e0')
        e1 = GenericVector([0, R**0.5, 0, 0], coords, config='u',
                           parent_metric=metric, name='e1')
        e2 = GenericVector([0, 0, 1 / r, 0], coords, config='u',
                           parent_metric=metric, name='e2')
        e3 = GenericVector([0, 0, 0, 1 / (r * sympy.sin(theta))], coords,
                           config='u', parent_metric=metric, name='e3')
        return metric, [e0, e1, e2, e3]
    return metric


def make_kerr(return_orthonormal_tetrad=False, r_s=None, a=None, c=None):
    """
    Creates a symbolic metric in Boyer Lindquist coordinates, implemented
    in EinsteinPy.

    Arguments
    ---------
    return_orthonormal_tetrad : bool, optional
        Whether to return the orthonormal tetrad coordinates. By default False.
    r_s : float or sympy.Symbol, optional.
        Schwarzschild radius value to be substituted into the metric. By
        default substitutes :math:`r_s`.
    a : float or sympy.Symbol, optional.
        Spin factor value to be substituted into the metric. By
        default substitutes :math:`a`.
    c : float or sympy.Symbol, optional.
        Speed of light value to be substituted into the metric. By default
        substitutes :math:`c`.

    Returns
    -------
    metric : `einsteinpy.symbolic.metric.MetricTensor`
        The desired metric tensor (`ll` configuration).
    tetrad : list of `einstenpy.symbolic.GenericVector`
        The orthonormal tetrad vectors `e0`, `e1`, ...
    """

    # Coordinate system
    t, theta, phi = sympy.symbols('t theta phi', real=True)
    r = sympy.Symbol('r', real=True, positive=True)
    coords = (t, r, theta, phi)
    # Metric variables
    if r_s is None:
        r_s = sympy.Symbol('r_s', real=True, positive=True)
    if c is None:
        c = sympy.Symbol('c', real=True, positive=True)
    if a is None:
        a = sympy.Symbol('a', real=True, positive=True)
    # Metric
    Sigma = r**2 + a**2 * sympy.cos(theta)**2
    Delta = r**2 - r_s * r + a**2

    metric = sympy.diag(
        - (1 - r_s * r / Sigma) * c**2,
        Sigma / Delta,
        Sigma,
        ((r**2 + a**2 + (r_s * r * (a**2) * sympy.sin(theta)**2 / Sigma))
         * sympy.sin(theta)**2)).tolist()
    metric[0][3] = metric[3][0] = (-r_s * a * r * sympy.sin(theta)**2 / Sigma
                                   * c)

    metric = MetricTensor(metric, coords, name='Kerr')
    # Add the determinant by hand
    metric.det = - c**2 * Sigma**2 * sympy.sin(theta)**2
    metric.sqrt_neg_det = c * Sigma * sympy.sin(theta)
    if return_orthonormal_tetrad:
        e0 = GenericVector([(r**2 + a**2) / sympy.sqrt(Sigma * Delta), 0, 0,
                            a / sympy.sqrt(Sigma * Delta)], coords,
                           parent_metric=metric, config='u', name='e0')
        e1 = GenericVector([0, sympy.sqrt(Delta / Sigma), 0, 0], coords,
                           parent_metric=metric, config='u', name='e1')
        e2 = GenericVector([0, 0, sympy.sqrt(1 / Sigma), 0], coords,
                           parent_metric=metric, config='u', name='e2')
        e3 = GenericVector([a * sympy.sin(theta) / Sigma**0.5, 0, 0,
                            1 / Sigma**0.5 / sympy.sin(theta)], coords,
                           parent_metric=metric, config='u', name='e3')
        return metric, [e0, e1, e2, e3]
    return metric
