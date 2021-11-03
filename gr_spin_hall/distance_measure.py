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
# from .chain import Chain
# from .utils import spherical_to_cartesian


def angdist(point1, point2):
    if not numpy.isclose(point1['r'], point2['r']):
        raise ValueError("Add erro")

    return numpy.arccos(
        numpy.cos(point1['theta']) * numpy.cos(point2['theta'])
        + numpy.sin(point1['theta']) * numpy.sin(point2['theta'])
        * numpy.cos(point1['phi'] - point2['phi']))


# def minkowski_geodesic_fit(integrator_linear, pars=['r', 'theta', 'phi']):
#     X = spherical_to_cartesian(integrator_linear.chain.get_data(pars))

#     dX = X[1:, :] - X[0, :]
#     dX /= numpy.linalg.norm(dX, axis=1).reshape(-1, 1)

#     v = numpy.mean(dX, axis=0)
#     std = numpy.var(dX, axis=0).mean()**0.5
#     # TODO: Check that std is less than some boundary
#     return v, std


# def cartesian_distance(integrator, integrator_linear, observer_coordinates,
#                        pars=['r', 'theta', 'phi']):
#     v, __ = minkowski_geodesic_fit(integrator_linear)
#     pars = ['r', 'theta', 'phi']
#     r_observer = spherical_to_cartesian([observer_coordinates[p]
#                                          for p in pars])
#     r0 = spherical_to_cartesian([integrator.current_coord(p) for p in pars])
#     return numpy.linalg.norm(numpy.dot(r_observer - r0, v) * v
#                              - r_observer + r0)
