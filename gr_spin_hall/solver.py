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
from .translate import LambdifyVector
import sympy


def get_ray_direction(lmbda_pi0, p_covector, position, system_params,
                      direction):
    pi0_coords = [str(p) for p in lmbda_pi0.coords]
    pi0_params = [str(p) for p in lmbda_pi0.system_params]

    x0 = [position[sym] for sym in pi0_coords]
    params = system_params.copy()
    params.update(direction)
    kwargs = {p: params[p] for p in pi0_params}
    out = lmbda_pi0(x0, **kwargs)
    return {str(p): out[i] for i, p in enumerate(p_covector)}


class Model:
    _system_params = None

    def __init__(self, xdot, pdot, x_vector, p_covector, system_params):

        self.system_params = system_params

        pars = [sympy.Symbol(p) for p in system_params.keys()]
#        pars = list(system_params.keys())

        self.xdot = LambdifyVector(xdot, x_vector + p_covector, pars)
        self.pdot = LambdifyVector(pdot, x_vector + p_covector, pars)

    @property
    def system_params(self):
        return self._system_params

    @system_params.setter
    def system_params(self, system_params):
        self._system_params = system_params

    def update_system_params(self, **kwargs):
        self.system_params.update(kwargs)

    def __call__(self, tau, x):
        xdot = self.xdot(x, **self.system_params)
        pdot = self.pdot(x, **self.system_params)
        return xdot + pdot
