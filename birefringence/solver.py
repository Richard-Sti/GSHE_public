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


class Solver:
    """
    TODO:
        - Add some comments
        - Add Schwarzschild radius termination condition
        - Check whether the ray is evolving linearly
    """

    def __init__(self, fun, parameters, solver_kwargs):
        self._solver = RK45(fun, **solver_kwargs)
        self.parameters = parameters

        self._Nsteps = 0

        self._chain = self.solver.y.reshape(1, -1)

    @property
    def solver(self):
        return self._solver

    @property
    def Nsteps(self):
        return self._Nsteps

    @property
    def chain(self):
        # Output this one as a structured numpy array
        return self._chain

    def run(self, N):
        # Ok so the t_bound condition gets evaluated prior to stepping
        chain = numpy.full((N, self.solver.y.shape[0]), numpy.nan)

        for i in range(N):
            self.solver.step()
            chain[i, :] = self.solver.y
            self._Nsteps += 1

        self._chain = numpy.vstack([self._chain, chain])
