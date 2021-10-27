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
from scipy.integrate import RK45
import warnings


class TerminationConditions:
    r"""
    Class to check various integration boundary conditions, such as hitting
    the event horizon or the integration limits.

    TODO: how in Kerr boundary calculated

    The event horizon boundary is calculated as

    .. math::

        H = \frac{1}{2} * \left(R_s + \sqrt{R_s^2 - 4 * a^2}\right),

    where :math:`R_s, a` are the Schwarzschild radius and the Kerr spin
    parameter, respectively. The event horizon condition is evaluated as

    .. math::
        r <= (1 + \xi) * H,

    where :math:`\xi` is the tolerance `self.radius_tolerance`.


    Parameters
    ----------
    rs: float
        The Schwarzschild radius.
    a: float
        Kerr spin factor.
    tau_max: float
        Upper affine parameter integration limit.
    radius_tolerance: float, optional
        Tolerance of the event horizon boundary. See its definition for more
        details.
    """
    _r_boundary = None
    _tau_max = None
    _radius_tolerance = None

    def __init__(self, rs, a, tau_max, radius_tolerance=0.01):
        # Parse the inputs
        self._set_radius_boundary(rs, a)
        self.tau_max = tau_max
        self.radius_tolerance = radius_tolerance

        # Termination functions. Key must be the relevant parameter and
        # the function's only argument must be the parameter value.
        self._termination_funcs = {"r": self.radius_termination,
                                   "tau": self.tau_termination}
        self._termination_params = list(self._bound_funcs.keys())

    @property
    def r_boundary(self):
        """
        The event horizon boundary. See :py:class:`TerminationConditions`
        docstrings for more details.
        """
        return self._r_boundary

    def _set_radius_boundary(self, rs, a):
        """
        Sets the event horizon boundary. See :py:class:`TerminationConditions`
        docstrings for more details.

        Parameters
        ----------
        rs: float
            The Schwarzschild radius.
        a: float
            The Kerr spin parameter.
        """
        if not rs > 0:
            raise ValueError("`rs` > 0. Currently `rs` = {:.5f}.".format(rs))
        if a > 0 and rs >= 2 * a:
            raise ValueError("`rs` >= 2 * `a`. Currently ""`rs` = {:.5f}, "
                             "a = {:.5f}.".format(rs, a))
        self._r_boundary = 0.5 * (rs + (rs**2 - 4 * a**2)**0.5)

    @property
    def tau_max(self):
        """The integration affine parameter upper limit."""
        return self._tau_max

    @tau_max.setter
    def tau_max(self, tau_max):
        """
        Sets the integration affine parameter upper limit. Should be higher
        than some lower limit but is not being checked.
        """
        self._tau_max = float(tau_max)

    @property
    def radius_tolerance(self):
        """
        The event horizon radius tolerance.
        See :py:class:`TerminationConditions` docstrings for more details.
        """
        return self._radius_tolerance

    @radius_tolerance.setter
    def radius_tolerance(self, tolerance):
        """
        Sets the event horizon radius tolerance. See
        :py:class:`TerminationConditions` docstrings for more details. Ensures
        it is positive.
        """
        if not tolerance >= 0:
            raise ValueError("Expected `radius_tolerance >= 0`, "
                             "currently {:.5f}".format(tolerance))
        self._radius_tolerance = tolerance

    def radius_termination(self, r):
        """
        Checks whether the integration reached the event horizon. See
        :py:class:`TerminationConditions` docstrings for more details.

        Returns
        -------
        to_terminate: bool
            Whether the termination condition is satisfied.
        """
        if r <= (1 + self.radius_tolerance) * self.r_boundary:
            warnings.warn("Horizon hit, terminating. Current `r` = {:.5f}, "
                          "`horizon` = {:.5f}.".format(r, self.r_boundary))
            return True
        return False

    def tau_termination(self, tau):
        """
        Checks whether the integration upper limit `self.tau_max` was reached.

        Returns
        -------
        to_terminate: bool
            Whether the termination condition is satisfied.
        """
        if tau >= self.tau_max:
            warnings.warn("Maximum integration `tau` reached. Current "
                          "`tau` = {:.5f}, `tau_max` = {:.5f}."
                          .format(tau, self.tau_max))
            return True
        return False

    def __getitem__(self, param):
        """Returns the appropriate termination function for `param`."""
        if param not in self._termination_params:
            raise KeyError("Unknown boundary type: `{}`. Allowed: `{}`"
                           .format(bound, self._termination_params))
        return self._termination_funcs[param]


class Solver:
    """
    TODO:
        - Add some comments
        - Add Schwarzschild radius termination condition
        - Check whether the ray is evolving linearly
        - Check the time condition
    """
    _params = None

    def __init__(self, fun, params, solver_kwargs, terminator):
        self._solver = RK45(fun, **solver_kwargs)
        self.params = params

        self._chain = numpy.hstack([self.solver.y,
                                    self.solver.t]).reshape(1, -1)
        self.term = terminator

    @property
    def params(self):
        """
        Parameters denoting the affine parameter and the differentiated
        functions.

        Returns
        -------
        params: list (of str)
            Parameter names.
        """
        return self._params

    @params.setter
    def params(self, params):
        """Sets the parameters."""
        if len(params) != self.solver.y.size:
            raise ValueError("The number of `params` does not match the "
                             "number the vector length of the initial "
                             "positions.")
        self._params = [str(par) for par in params] + ['tau']

    @property
    def solver(self):
        """
        The ODE solver.

        Returns
        -------
        solver: :py:class:`scipy.integrate.RK45`
            The ODE solver.
        """
        return self._solver

    @property
    def Nsteps(self):
        """
        The number of integration steps.

        Returns
        -------
        Nsteps: int
            Number of steps.
        """
        return self._chain.shape[0] - 1

    @property
    def chain(self):
        # Output this one as a structured numpy array
        return numpy.core.records.fromarrays(
            self._chain.T, names=self.params, formats=[float]*len(self.params))

    @property
    def current_positions(self):
        return {par: self._chain[-1, i] for i, par in enumerate(self.params)}

    def current_coord(self, param):
        if param == 'tau':
            return self.solver.t
        return self.solver.y[self.params.index(param)]

    @property
    def to_terminate(self):
        return any(self.term[p](self.current_coord(p)) for p in ['r', 'tau'])

    def run_steps(self, N):
        chain = numpy.full((N, self.solver.y.shape[0] + 1), numpy.nan)
        # Check the termination conditions.
        if self.to_terminate:
            warnings.warn("Solver already terminated.")
            return

        # Run for N steps
        for i in range(N):
            if self.to_terminate:
                chain = chain[:i, :]
                break

            # Advance the solver
            self.solver.step()
            # Store to the chain
            chain[i, :-1] = self.solver.y
            chain[i, -1] = self.solver.t

        self._chain = numpy.vstack([self._chain, chain])

    def run_termination(self, batch_size):
        while True:
            self.run_steps(batch_size)
            if self.to_terminate:
                break
