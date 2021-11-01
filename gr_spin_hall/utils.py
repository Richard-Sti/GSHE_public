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

import numpy


def spherical_to_cartesian(X, r_normalise=False):
    r"""
    The usual conversion of spherical to Cartesian coordinates in
    :math:`\mathbb{E}^3`.

    Arguments
    ---------
    X: numpy.ndarray or  a 3-tuple
        Array of shape `(Npoints, 3)` where the second axis is `r`, `theta`,
        and `phi` respectively. If a tuple then it must be `r`, `theta, `phi`.
    r_normalise: bool, optional
        If `True` will set :math:`r = 1`. By default `False`.

    Returns
    -------
    out: numpy.ndarray
        Array of shape `(Npoints, 3)` where the second axis is `x`, `y`,
        and `z` respectively. If input was a tuple the shape is `(3, )`.
    """
    if isinstance(X, (list, tuple)):
        r, theta, phi = X
    else:
        r, theta, phi = X[:, 0], X[:, 1], X[:, 2]

    if r_normalise:
        r = 1

    stheta, ctheta = numpy.sin(theta), numpy.cos(theta)
    sphi, cphi = numpy.sin(phi), numpy.cos(phi)
    x = r * cphi * stheta
    y = r * sphi * stheta
    z = r * ctheta
    if isinstance(X, (list, tuple)):
        return numpy.array([x, y, z])
    return numpy.vstack([x, y, z]).T
