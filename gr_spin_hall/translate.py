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
    A class to lambdify a `sympy.Matrix` vector. Enforces the correct ordering
    of input `x` in `self.__call__`.


    Arguments
    ---------
    vector: sympy.Matrix
        The symbolic expression which is to be lambdified.
    symbols: list (of sympy.Symbol)
        Symbols expected to be passed into the numerical expression during
        runtime.
    subs: dict, optinonal
        Substitution dictionary. Key must be the original symbol in
        `vector`, value the substitude symbol. By default no substitution.
    """
    _symbols = None

    def __init__(self, vector, symbols, subs=None):
        # Parse the inputs
        if not isinstance(vector, sympy.Matrix):
            raise ValueError("`vector` must be `sympy.Matrix`. Currently `{}`"
                             .format(type(vector)))
        self.symbols = symbols
        # Optionally substitute
        vector = self.substitute_symbols(vector, subs)
        # Unpack the remaining arguments after subtitution
        self._args = list(vector.free_symbols)
        # Memoise dictionary
        modules = {'sin': self._sin, 'cos': self._cos, 'tan': self._tan}
        # Lambdify
        self._lmbdas = [sympy.utilities.lambdify(self.args, f, modules=modules)
                        for f in vector]
        # Get the ordering
        self._ordering = self._get_ordering()

    @property
    def args(self):
        """
        Lambdified vector arguments ordered as accepted by the
        `sympy.utilities.lambdify` lambdified expressions `self.lmbdas`.

        Returns
        -------
        args: list (of `sympy.Symbol`)
            The lambdified expression inputs ordered as expected.
        """
        return self._args

    @property
    def symbols(self):
        """
        Symbols which are expected to be inputted in `x` in `self.call`.

        Returns
        -------
        symbols: list (of `sympy.Symbol`)
            The expected inputs.
        """
        return self._symbols

    @symbols.setter
    def symbols(self, symbols):
        """
        Sets `self.symbols`. Makes sure it is a list.

        Arguments
        ---------
        symbols: list (of `sympy.Symbol`)
            The expected inputs.
        """
        if not isinstance(symbols, (list, tuple)):
            raise ValueError("`symbols` must be a list. Currently {}"
                             .format(type(symbols)))
        self._symbols = symbols

    @property
    def ordering(self):
        """
        Ordering of `self.args` in `self.symbols`.

        Returns
        -------
        ordering: list (of ints)
            The ordering.
        """
        return self._ordering

    def _get_ordering(self):
        """
        Gets `self.ordering`. Checks `args` are present in `self.symbols.`

        Returns
        -------
        ordering: list (of ints)
            The ordering of `self.args` in `self.symbols`.
        """
        ordering = [None] * len(self.args)
        for i, arg in enumerate(self.args):
            try:
                ordering[i] = self.symbols.index(arg)
            except ValueError:
                raise ValueError("{} not found in `symbols` {}"
                                 .format(arg, self.symbols))
        return ordering

    @staticmethod
    def substitute_symbols(vector, subs):
        """
        Substitutes `subs` into vector. `subs`' keys must be present in
        `vector.free_symbols`.

        Arguments
        ---------
        vector: sympy.Matrix
            Original vector.
        subs: dict
            Substitution dictionary. Key must be the original symbol in
            `vector`, value the substitude symbol.

        Returns
        -------
        substituted_vector: sympy.Matrix
            Substituted vector.
        """
        if subs is None:
            return vector

        # Check all subs are present in the vector's symbols
        for key in subs.keys():
            if not isinstance(key, sympy.Symbol):
                raise ValueError("`{}` must be a `sympy.Symbol`. "
                                 "Currently `{}`".format(key, type(key)))
            if key not in vector.free_symbols:
                raise ValueError("`{}` not in vector's symbols `{}`."
                                 .format(key, vector.free_symbols))
        return vector.subs(subs)

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
