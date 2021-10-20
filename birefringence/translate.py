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
import sympy
from sympy.parsing.mathematica import mathematica
from functools import lru_cache


def parse_mathematica_vector(fpath, coords, additional_translations=None,
                             subs=None):
    vector = []
    with open(fpath, "r") as file:
        for i, line in enumerate(file):
            expr = line
            if i > 0:
                raise ValueError("Unexpected input (more than 1 line).")
        # Mathematica vector is split like this
        # The brackets denote the first and last position.
        for i, sym in enumerate(expr.split(', ')):
            if sym.startswith('{'):
                sym = sym[1:]
            if sym.endswith('}'):
                sym = sym[:-1]
            vector.append(mathematica(sym, additional_translations))

    vector = sympy.Matrix(vector)

    if subs is not None:
        vector = vector.subs(subs)
    return vector


class LambdifyVector:
    """
    A class to lambdify a `sympy.matrices.dense.MutableDenseMatrix` vector.
    Ensures the correct ordering of input `x` in `self.__call__`.
    """
    _symbols = None

    def __init__(self, vector, symbols, subs_symbols=None):
        if not isinstance(vector, sympy.matrices.dense.MutableDenseMatrix):
            raise ValueError("`vector` must be "
                             "`sympy.matrices.dense.MutableDenseMatrix`.")
        self.symbols = symbols
        # Optionally substitute
        if subs_symbols is not None:
            vector = vector.subs(subs_symbols)
        # Unpack the remaining arguments
        self._args = list(vector.free_symbols)
        # Memoise dictionary
        modules = {'sin': self._sin, 'cos': self._cos, 'tan': self._tan}
        self._lmbdas = [sympy.utilities.lambdify(self.args, f, modules=modules)
                        for f in vector]
        # Set the ordering
        self._ordering = [None] * len(self.args)
        for i, arg in enumerate(self.args):
            try:
                self._ordering[i] = symbols.index(arg)
            except ValueError:
                raise ValueError("{} not found in `symbols` {}"
                                 .format(arg, symbols))

    @property
    def args(self):
        """Lambdified vector arguments."""
        return self._args

    @property
    def symbols(self):
        """Input symbols."""
        return self._symbols

    @symbols.setter
    def symbols(self, symbols):
        """Sets symbols."""
        if not isinstance(symbols, (list, tuple)):
            raise ValueError("`symbols` must be a list. Currently {}"
                             .format(type(symbols)))
        self._symbols = symbols

    @property
    def ordering(self):
        """Ordering of `self.args` in `self.symbols`."""
        return self._ordering

    @staticmethod
    @lru_cache
    def _sin(x):
        """Memoised `numpy.sin`."""
        return numpy.sin(x)

    @staticmethod
    @lru_cache
    def _cos(x):
        """Memoised `numpy.cos`."""
        return numpy.cos(x)

    @staticmethod
    @lru_cache
    def _tan(x):
        """Memoised `numpy.tan`."""
        return numpy.tan(x)

    def __call__(self, x):
        """
        Evaluates the vector at `x`.

        Arguments
        ---------
        x : list of numpy.ndarray
            Values at which to evaluate the vector. Must be order according to
            `self.symbols`.
        Returns
        -------
        out : list
            Vector evaluated at `x`.
        """
        if not isinstance(x, numpy.ndarray):
            x = numpy.asarray(x)
        return [lmbda(*x[self.ordering]) for lmbda in self._lmbdas]
