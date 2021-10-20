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

from abc import ABC

from itertools import product

import sympy
from sympy.functions.special.tensor_functions import LeviCivita
from einsteinpy.symbolic import (MetricTensor, GenericVector,
                                 BaseRelativityTensor)

import numpy


class BaseCalculator(ABC):
    """
    A base calculator class. Containts helpful functions to unpack symbolic
    tensors.
    """

    _metric = None
    _inv_metric = None
    _t_vector = None

    @property
    def metric(self):
        """The metric tensor."""
        return self._metric

    @metric.setter
    def metric(self, g):
        """Sets the metric tensor."""
        if not isinstance(g, MetricTensor):
            raise ValueError("metric must be of "
                             "`einsteinpy.symbolic.metric.MetricTensor` type.")
        self._metric = g
        self._inv_metric = g.inv()

    @property
    def inv_metric(self):
        """The inverse metric."""
        return self._inv_metric

    @property
    def epsilon(self):
        r"""The :math:`\epsilon` symbol."""
        return sympy.Symbol('epsilon', real=True, positive=True)

    @property
    def t_vector(self):
        r"""The timelike vector :math:`t^\mu`."""
        return self._t_vector

    @t_vector.setter
    def t_vector(self, t):
        """Sets the timelike vector `self.t_vector`."""
        if not isinstance(t, GenericVector):
            raise ValueError("t must be of "
                             "`einsteinpy.symbolic.metric.GenericVector` "
                             "type.")
        self._t_vector = t

    @property
    def Ndim(self):
        """Metric dimensionality."""
        return self.metric.dims

    @property
    def coords(self):
        r"""A tuple of the spacetime coordinates :math:`x^\mu`."""
        return self.metric.symbols()

    @property
    def p_covector_symbols(self):
        r"""
        A tuple of the :math:`p_\mu` covector symbols. The lower indices are
        assigned according to `self.coords`.
        """
        return tuple(sympy.symbols(['p_{}'.format(p) for p in self.coords]))

    @property
    def metric_variables(self):
        """
        A tuple of the metric variables (i.e. any variables other than
        `self.coords`).
        """
        return tuple(self.metric.variables)

    @property
    def symbols(self):
        """
        A tuple of symbols arranged as
        `self.coords + self.p_covector_symbols + self.metric_variables`.
        """
        return self.coords + self.p_covector_symbols + self.metric_variables

    @property
    def x_vector(self):
        r"""The :math:`x^\mu` vector."""
        return GenericVector(self.coords, self.coords, config='u',
                             parent_metric=self.metric)

    @property
    def p_covector(self):
        r"""The :math:`p_\mu covector."""
        return GenericVector(self.p_covector_symbols, self.coords, config='l',
                             parent_metric=self.metric)


class SymbolicCalculator(BaseCalculator):
    """
    pass

    """

    def __init__(self, metric, t):
        self.metric = metric
        self.t_vector = t

    @property
    def partial_inverse_metric(self):
        r"""
        The partial derivatives of the inverse metric,

        .. math::

            \partial_\mu \tensor{g}{^\alpha ^\beta}.

        Returns
        -------
        out : a nested lists
            A nested list of partial derivatives indexed as
            `out[mu][alpha][beta]`, following the index structure from above.
        """
        out = numpy.zeros((self.Ndim, self.Ndim, self.Ndim),
                          dtype=int).tolist()
        inv = self.inv_metric

        for mu, alpha, beta in product(range(self.Ndim), repeat=3):
            out[mu][alpha][beta] = sympy.diff(inv[alpha, beta],
                                              self.coords[mu])
        return out

    @property
    def sigma_bivector(self):
        """
        Docs.
        """
        Sigma = numpy.zeros((self.Ndim, self.Ndim), dtype=int).tolist()

        p_dot_t = sum(self.p_covector[mu] * self.t_vector[mu]
                      for mu in range(self.Ndim))
        t_covector = self.t_vector.change_config('l')

        for alpha, beta in product(range(self.Ndim), repeat=2):
            for mu, nu in product(range(self.Ndim), repeat=2):
                Sigma[alpha][beta] += (LeviCivita(alpha, beta, mu, nu)
                                       * self.p_covector[mu] * t_covector[nu])
            Sigma[alpha][beta] = sympy.simplify(Sigma[alpha][beta] / p_dot_t
                                                / self.metric.sqrt_neg_det)

        return BaseRelativityTensor(Sigma, self.coords, config='uu',
                                    parent_metric=self.metric, name='Sigma')

    @property
    def xdot(self):
        r"""
        The derivative of the ray coordinate with respect to the affine
        parameter,

        .. math::

            \dot{x}^\mu = \tensor{g}{^\mu ^\nu} p_\nu.
        """
        invg = self.inv_metric
        p = self.p_covector

        xdot = [0] * self.Ndim
        for mu in range(self.Ndim):
            for nu in range(self.Ndim):
                xdot[mu] += invg[mu, nu] * p[nu]
        # Wrap this in a tensor-like expression
        return GenericVector(xdot, self.coords)

    @property
    def pdot(self):
        r"""
        The derivative of the :math:`p_\nu` covector with respect to the
        affine parameter,

        .. math::
            \dot{p}_\mu = - \frac{1}{2} \partial_\mu \tensor{g}{^\alpha ^\beta}
            p_{\alpha} p_{\beta}.
        """
        pdot = [0] * self.Ndim
        p = self.p_covector

        p_invg = self.p_covectorartial_invg
        for mu in range(self.Ndim):
            for alpha in range(self.Ndim):
                for beta in range(self.Ndim):
                    pdot[mu] += p_invg[mu][alpha][beta] * p[alpha] * p[beta]
            pdot[mu] *= -sympy.Rational(1, 2)
        # Wrap this in a tensor-like expression
        return GenericVector(pdot, self.coords)

    @property
    def p0(self):
        r"""
        Returns the zeroth coordinate of the :math:`p_\nu` covector,

        .. math::
            p_0 = 1 / \tensor{g}{^0 ^0}
              * \left(
                  - \tensor{g}{^0 ^i} p_i
                  + \sqrt{(\tensor{g}{^0 ^i} p_i)^2 - \tensor{g}{^0 ^0}
                  \tensor{g}{^i ^j} p_i p_j
              \right),

        assumig that :math:`p_\alpha p^\alpha = 0`. Note that this only
        containts the
        :math:`\epsilon^0` contribution, check again why that is.
        """

        invg = self.inv_metric
        N = self.Ndim
        p = self.p_covector

        part1 = - sum(invg[0, i] * p[i] for i in range(1, N))
        part2 = sympy.sqrt(
            sum(invg[0, i] * p[i] for i in range(1, N))**2
            - invg[0, 0] * sum(invg[i, j] * p[i] * p[j]
                               for j in range(1, N) for i in range(1, N)))

        return (part1 + part2) / self.inv_metric[0, 0]
