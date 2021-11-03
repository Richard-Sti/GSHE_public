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
from scipy.optimize import minimize
import warnings

from .chain import Chain


FAIL_TERM_FLAG = 0
SUCCESS_TERM_FLAG = 1
CONT_FLAG = 2


class TerminationConditions:
    r"""
    Class to check various integration boundary conditions, such as hitting
    the event horizon or the integration limits.

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
    _radius_max = None

    def __init__(self, rs, a, tau_max, radius_max=None, radius_tolerance=0.01):
        # Parse the inputs
        self._set_radius_boundary(rs, a)
        self.tau_max = tau_max
        self.radius_tolerance = radius_tolerance
        self.radius_max = radius_max

        # Termination functions. Key must be the relevant parameter and
        # the function's only argument must be the parameter value.
        self._termination_funcs = {"r": self.radius_termination,
                                   "tau": self.tau_termination}
        self._termination_params = list(self._termination_funcs.keys())

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
        if not (a >= 0 and rs >= 2 * a):
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
    def radius_max(self):
        """The maximum geodesic integration radius."""
        return self._radius_max

    @radius_max.setter
    def radius_max(self, radius_max):
        """
        Sets the maximum geodesic integration radius. Checks that it is higher
        than `self.rs`.
        """
        if radius_max is None:
            self._radius_max = numpy.infty
        if not radius_max > self.r_boundary:
            raise ValueError("`radius_max` must be larger than `self.rs`.")
        self._radius_max = float(radius_max)

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
        status_flag: int
            Status flag.
        """
        if r <= (1 + self.radius_tolerance) * self.r_boundary:
            warnings.warn("Horizon hit, terminating. Current `r` = {:.5f}, "
                          "`horizon` = {:.5f}.".format(r, self.r_boundary))
            return FAIL_TERM_FLAG
        if r >= self.radius_max:
            warnings.warn("Maximum `r` reached, terminating. Current "
                          "`r` = {:.5f} `radius_max` = {:.5f}."
                          .format(r, self.radius_max))
            return SUCCESS_TERM_FLAG
        return CONT_FLAG

    def tau_termination(self, tau):
        """
        Checks whether the integration upper limit `self.tau_max` was reached.

        Returns
        -------
        status_flag: int
            Status flag.
        """
        if tau >= self.tau_max:
            warnings.warn("Maximum integration `tau` reached. Current "
                          "`tau` = {:.5f}, `tau_max` = {:.5f}."
                          .format(tau, self.tau_max))
            return SUCCESS_TERM_FLAG
        return CONT_FLAG

    def __getitem__(self, param):
        """
        Returns the appropriate termination function for `param`.
        """
        if param not in self._termination_params:
            raise KeyError("Unknown boundary type: `{}`. Allowed: `{}`"
                           .format(param, self._termination_params))
        return self._termination_funcs[param]


class Integrator:
    """
    A RK45 integrator, checks for termination conditions, logs data, and
    terminates when the geodesic becomes a straight line.

    Parameters
    ----------
    func: `py:func`
        The set of first order ordinary differential equations. `func` must
        return a list of derivatives whose order corresponds to `params`.
    params: list (of str)
        Parameters that are being integrated.
    initial_positions: dict
        A dictionary of initial positions whose keys must agree with `params`.
    rs: float
        The Schwarzschild radius.
    a: float
        The Kerr spin factor. If `func` corresponds to the Schwarzschild
        solution instead of Kerr then it is ignored.
    tau_min: float
        Affine parameter lower integration limit.
    tau_max: float
        Affine parameter upper integration limit.
    radius_tolerance: float, optional
        Tolerance of the event horizon boundary.
        See `py:class:TerminationConditions` for more details.
    integrator_kwargs: dict
        Additional arguments to be passed into `scipy.integrate.RK45` such
        as the tolerance.
    """

    def __init__(self, model, params, initial_positions, rs, a, tau_min=0,
                 tau_max=1000, radius_max=None, affine_param='tau',
                 radius_tolerance=0.01, integrator_kwargs=None):
        # Object with termination conditions
        self.term = TerminationConditions(rs=rs, a=a, tau_max=tau_max,
                                          radius_max=radius_max,
                                          radius_tolerance=radius_tolerance)
        # Initialise the chain
        self._chain = Chain(initial_positions=initial_positions,
                            tau_min=tau_min, params=params,
                            affine_param=affine_param)
        # Initialise the solver
        self._solver = RK45(model, t0=tau_min, y0=self.chain.data[0, :-1],
                            t_bound=tau_max, **integrator_kwargs)

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
    def params(self):
        """
        Parameters denoting the the differentiated functions and the affine
        parameter.

        Returns
        -------
        params: list (of str)
            Parameter names.
        """
        return self.chain.params

    @property
    def affine_param(self):
        r"""
        The affine parameter :math:`\tau`.

        Returns
        -------
        affine_param: str
            The affine parameter.
        """
        return self.chain.affine_param

    @property
    def chain(self):
        """
        The chain used to log the outputs.

        Returns
        -------
        chain: `:py:class:gr_spin_hall.Chain`.
        """
        return self._chain

    @property
    def Nsteps(self):
        """
        The number of integration steps.

        Returns
        -------
        Nsteps: int
            Number of steps.
        """
        return self.chain.data.shape[0] - 1

    @property
    def current_coords(self):
        """
        Current coordinates of the integrator corresponding to `self.params`.

        Returns
        -------
        current_position: dict
            Dictionary with the current position.
        """
        out = {par: self.solver.y[i] for i, par in enumerate(self.params[:-1])}
        out.update({self.affine_param: self.solver.t})
        return out

    def current_coord(self, param):
        """
        Current value of parameter `param` of the integrator.
        Arguments
        ---------
        param: str
            Parameter whose current position to return.
        Returns
        -------
        value: float
            The current value of the parameter.
        """
        if param == self.affine_param:
            return self.solver.t
        return self.solver.y[self.params.index(param)]

    @property
    def termination_flag(self):
        """
        Checks the termination conditions and returns the termination flag.

        Returns
        -------
        flag: int
            Termination flag.
        """
        flags = [self.term[p](self.current_coord(p))
                 for p in self.term._termination_params]
        if FAIL_TERM_FLAG in flags:
            return FAIL_TERM_FLAG
        elif SUCCESS_TERM_FLAG in flags:
            return SUCCESS_TERM_FLAG
        else:
            return CONT_FLAG

    def coords_at_value(self, value, param, minimizer_kwargs=None):
        """
        Returns the `self.params` evaluated at `param = value`.

        Arguments
        ---------
        value: float
            Value of `param` at which to get `self.params`.
        param: str
            Parameter corresponding to `value`. Must be from `self.params`.
        minimizer_kwargs: **kwargs
            Optional kwargs passed into `scipy.optimize.minimize`.

        Returns
        -------
        result: dict
            Dictionary with the requested values of `self.params.`
        """

        # Check param is a valid parameter and specifically if its tau
        if param not in self.params:
            raise ValueError("Param `{}` not in `self.param = {}`"
                             .format(param, self.params))
        result = {}
        # Interp returns x^\mu and p_i as a function of tau
        interp = self.solver.dense_output()  # Returns a 1D arrray
        if param == self.affine_param:
            out = interp(value)
            result.update({self.affine_param: value})
        else:
            if minimizer_kwargs is None:
                minimizer_kwargs = {}

            index = self.params.index(param)
            squared_diff = lambda tau: (interp(tau)[index] - value)**2

            # Solves at which tau the difference is minimized
            x0 = self.current_coord(self.affine_param)  # initial tau
            sol = minimize(squared_diff, x0, **minimizer_kwargs)

            if not sol['success']:
                print(sol)
                raise ValueError("Minimizer did not converge. May be due to "
                                 "strange requested value.")
            # Check that the resulting array has size 1
            if sol['x'].size > 1:
                raise ValueError("Multiple solutions detected. Check inputs.")
            tau_min = sol['x'][0]
            result.update({self.affine_param: tau_min})
            # Cast this result into a dictionary
            out = interp(tau_min)
        result.update({p: out[i] for i, p in enumerate(self.params[:-1])})
        return result

    def run_steps(self, N):
        """
        Run the solver for `N` steps.

        Arguments
        ---------
        N: int
            Number of integration steps.

        Returns
        -------
        flag: int
            Termination flag.
        """
        # Force an integer value
        if not isinstance(N, int):
            N = int(N)
        if not N > 0:
            raise ValueError("Required `N` > 0. Currently `N` = {}.".format(N))
        chain = numpy.full((N, len(self.params)), numpy.nan)
        # Check if the solver already meets the termination conditions
        flag = self.termination_flag
        if flag in [FAIL_TERM_FLAG, SUCCESS_TERM_FLAG]:
            return flag

        # Run for N steps
        for i in range(N):
            # Advance the solver
            self.solver.step()
            # Store to the chain
            chain[i, :-1] = self.solver.y
            chain[i, -1] = self.solver.t  # tau always goes at the last pos.

            flag = self.termination_flag
            if flag in [FAIL_TERM_FLAG, SUCCESS_TERM_FLAG]:
                chain = chain[:i + 1, :]
                break

        # Stack the working chain to the big chain
        self.chain.append(chain)
        return flag

    def run_termination(self, batch_size):
        """
        Run the solver until termination (i.e. maximum `tau` or the radius
        boundary is hit).

        Arguments
        ---------
        batch_size: int
            Number of integration steps to perform in one batch.

        Returns
        -------
        flag: int
            Termination flag.
        """
        # Force an integer value
        if not isinstance(batch_size, int):
            batch_size = int(batch_size)
        # Check if the solver already meets the termination conditions
        flag = self.termination_flag
        if flag in [FAIL_TERM_FLAG, SUCCESS_TERM_FLAG]:
            return flag

        # Keep running until termination
        while True:
            flag = self.run_steps(batch_size)
            if flag in [FAIL_TERM_FLAG, SUCCESS_TERM_FLAG]:
                return flag
