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

# Default mathemathica required translations
TRANSLATIONS = {'Csc[x]': '1 / sin(x)',
                'Cot[x]': '1 / tan(x)'}


def parse_mathematica_vector(fpath, coords, additional_translations=None,
                             subs=None):
    r"""
    Parses a vector expression from Mathematica and translates it to Sympy.
    The vector must look like :math:`\left\{x_0, x_1, \ldots\right)}`, i.e.
    be wrapped in curly brackets and separated with ', '. Note the space
    following the comma.

    See documentation for more details on how to export from Mathematica.


    Arguments
    ---------
    fpath: str
        Filepath to the saved '.txt' mathematica expression.
    coords: list (of :py:class:`sympy.Symbol`)
        Spacetime coordinates, `len(coords)` must match the vector length.
    additional_translations: dict
        Dictionary of translations of Mathematica functions to Sympy. For
        example: `{'Csc[x]': '1 / sin(x)'}`. Used for functions which are not
        natively supported in SymPy.
    subs: dict
        Dictionary of symbolic substitutions into the final expression, can
        look like `{'alpha': alpha}`, where `alpha` is a
        `:py:class:sympy.Symbol` type.

    Returns
    -------
    vector: sympy.Matrix
        The translated symbolic vector.
    """
    if additional_translations is None:
        additional_translations = TRANSLATIONS
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
    r"""
    A class to lambdify a `sympy.Matrix` vector. Enforces the correct ordering
    of input `x` in `self.__call__`.


    Arguments
    ---------
    vector: sympy.Matrix
        The symbolic expression which is to be lambdified.
    coords: list (of sympy.Symbol)
        Phase space coordinates.
    system_params: list (of sympy.Symbol)
        System parameters to be substituted into the vector. Typically
        the Schwarzschild radius etc..
    """
    _coords = None
    _system_params = None

    def __init__(self, vector, coords, system_params):
        # Parse the inputs
        if not isinstance(vector, sympy.Matrix):
            raise ValueError("`vector` must be `sympy.Matrix`. Currently `{}`"
                             .format(type(vector)))
        self.coords = coords
        self.system_params = list(system_params)

        self._args = list(vector.free_symbols)
        # In args we only want the phase space coords
        for param in self.system_params:
            self._args.remove(param)
        # Check that args have only phase space coordinates
        for arg in self.args:
            if arg not in self.coords:
                raise KeyError("`{}` not a phase space coordinate".format(arg))
        # Memoise dictionary
        modules = {'sin': self._sin, 'cos': self._cos, 'tan': self._tan}
        # Lambdify
        pars = self.args + self.system_params
        self._lmbdas = [sympy.utilities.lambdify(pars, f, modules=modules)
                        for f in vector]
        # Get the ordering
        self._ordering = self._get_ordering()

    @property
    def args(self):
        """
        Phase space coordinate arguments ordered as expected by `self._lmbdas`.

        Returns
        -------
        args: list (of `sympy.Symbol`)
            Phase space coordinates.
        """
        return self._args

    @property
    def coords(self):
        """
        Phase space coordinates ordered as expected to be input in `x` in
        `self.call`.

        Returns
        -------
        coords: list (of `sympy.Symbol`)
            Phase space coordinates.
        """
        return self._coords

    @coords.setter
    def coords(self, coords):
        """
        Sets `self.coords`. Makes sure it is a list.

        Arguments
        ---------
        coords: list (of `sympy.Symbol`)
            The expected inputs.
        """
        if not isinstance(coords, (list, tuple)):
            raise ValueError("`coords` must be a list. Currently {}"
                             .format(type(coords)))
        self._coords = coords

    @property
    def system_params(self):
        """
        System parameters expected to be passed in `self.call` as **kwargs.

        Returns
        -------
        coords: list (of `sympy.Symbol`)
            System parameters.
        """
        return self._system_params

    @system_params.setter
    def system_params(self, system_params):
        """
        Sets `self.system_params`. Makes sure it is a list.

        Arguments
        ---------
        system_params: list (of `sympy.Symbol`)
            The **kwargs keys in `self.call`.
        """
        if not isinstance(system_params, (list, tuple)):
            raise ValueError("`coords` must be a list. Currently {}"
                             .format(type(system_params)))
        self._system_params = list(system_params)

    @property
    def ordering(self):
        """
        Ordering of `self.args` in `self.coords`.

        Returns
        -------
        ordering: list (of ints)
            The ordering.
        """
        return self._ordering

    def _get_ordering(self):
        """
        Gets `self.ordering`. Checks `args` are present in `self.coords.`

        Returns
        -------
        ordering: list (of ints)
            The ordering of `self.args` in `self.coords`.
        """
        ordering = [None] * len(self.args)
        for i, arg in enumerate(self.args):
            try:
                ordering[i] = self.coords.index(arg)
            except ValueError:
                raise ValueError("{} not found in `coords` {}"
                                 .format(arg, self.coords))
        return ordering

    @staticmethod
    @lru_cache
    def _sin(x):
        """
        Memoised `numpy.sin`.

        Arguments
        ---------
        x: float or numpy.ndarray (of floats)
            Function input.

        Returns
        -------
        y: float or numpy.ndarray (of floats)
            Function output.
        """
        return numpy.sin(x)

    @staticmethod
    @lru_cache
    def _cos(x):
        """
        Memoised `numpy.cos`.

        Arguments
        ---------
        x: float or numpy.ndarray (of floats)
            Function input.

        Returns
        -------
        y: float or numpy.ndarray (of floats)
            Function output.
        """
        return numpy.cos(x)

    @staticmethod
    @lru_cache
    def _tan(x):
        """
        Memoised `numpy.tan`.

        Arguments
        ---------
        x: float or numpy.ndarray (of floats)
            Function input.

        Returns
        -------
        y: float or numpy.ndarray (of floats)
            Function output.
        """
        return numpy.tan(x)

    def __call__(self, x, **kwargs):
        r"""
        Evaluates the vector at `x` with system parameters **kwargs.

        Arguments
        ---------
        x: list of numpy.ndarray
            Vector :math:`x^\mu` and covectors :math:`p_i` at which to evaluate
            the vector. Must be order according to `self.coords`.
        **kwargs:
            Additional values corresponding to `self.fix_values`. Typically
            the Schwarzschild radius, polarisation etc..
        Returns
        -------
        out : list
            Vector evaluated at `x`.
        """
        if not isinstance(x, numpy.ndarray):
            x = numpy.asarray(x)
        return [lmbda(*x[self.ordering], **kwargs) for lmbda in self._lmbdas]
