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


class Chain:
    r"""
    A chain class for data storage and retrieval of the stepped positions.

    Arguments
    ---------
    initial_positions: dict
        A dictionary of initial positions with keys matching `params`.
    tau_min: float
        The lower integration limit on :math:`\tau`.
    params: list (of str)
        List of parameters that are being integrated.
    affine_param: str, optional
        The affine parameter.
    """
    _chain = None
    _params = None
    _affine_param = None

    def __init__(self, initial_positions, tau_min, params,
                 affine_param='tau'):
        # Parse the inputs
        self.affine_param = affine_param
        self.params = params
        self._chain = self._create_chain(initial_positions, tau_min)

    @property
    def affine_param(self):
        r"""
        The affine parameter :math:`\tau`.

        Returns
        -------
        affine_param: str
            The affine parameter.
        """
        return self._affine_param

    @affine_param.setter
    def affine_param(self, affine_param):
        r"""Sets the affine parameter :math:`tau`."""
        if not isinstance(affine_param, str):
            raise ValueError("`affine_param` must be a `str`. Currently `{}`."
                             .format(type(affine_param)))
        self._affine_param = affine_param

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
        return self._params

    @params.setter
    def params(self, params):
        """Sets the parameters `self.params`."""
        self._params = [str(par) for par in params] + [self.affine_param]

    def _create_chain(self, initial_positions, tau_min):
        r"""
        Initialises the chain with `initial_positions` and puts `tau_min` at
        the last position.

        Arguments
        ---------
        initial_positions: dict
            Dictionary of initial positions. Dictionary keys must match
            `self.params`.
        tau_min: float
            Starting value of the affine parameter :math:`\tau`

        Returns
        -------
        chain: numpy.ndarray
            The initial chain of shape `(1, len(self.params))`.
        """
        if len(initial_positions) != len(self.params) - 1:
            raise ValueError("The number of initial positions `{}` does not "
                             "match the number of parameters `{}` (excluding "
                             "the affine par.)"
                             .format(len(initial_positions), len(self.params)))
        # Check we have the right initial positions
        x0 = [None] * (len(self.params) - 1)
        for i, key in enumerate(self.params):
            if key == self.affine_param:
                continue
            elif key not in initial_positions.keys():
                raise KeyError("Initial parameter `{}` not present in "
                               "`self.params: {}`".format(key, self.params))
            x0[i] = initial_positions[key]
        return numpy.hstack([x0, tau_min]).reshape(1, -1)

    @property
    def data(self):
        """
        The chain including the stepped positions.

        Returns
        -------
        chain: numpy.ndarray
            The chain as an array of shape `(self.Nsteps+1, len(self.params)).`
        """
        return self._chain

    def get_data(self, params=None, from_step=None):
        """
        Select a slice of the `self.data` array of stepped positions.

        Arguments
        ---------
        params: str of list (of str), optional
            Parameters to be returned. By default all parameters returned.
        from_step: int, optional
            From which step onwards is the data to be returned. By default all
            stepped positions are returned.

        Returns
        -------
        out: numpy.ndarray
            A sliced array of stepped positions. The second axis is ordered
            according to `params`.
        """
        # First check the from step input
        if from_step is None:
            from_step = 0
        if not isinstance(from_step, int):
            raise ValueError("`from_step` must be an int. Currently `{}`."
                             .format(type(from_step)))
        # If no parameters specifically selected.
        if params is None:
            return self._chain[from_step:, :]
        # Ensure we have a list if only a single point
        if isinstance(params, str):
            params = [params]

        if not all(isinstance(p, str) and (p in self.params) for p in params):
            raise ValueError("Unknown parameter `{}`. Must be from `{}`"
                             .format(params, self.params))
        return self.data[from_step:, [self.params.index(p) for p in params]]

    @property
    def structured_data(self):

        """
        The data stored in the chain as a structured array.

        Returns
        -------
        chain: numpy.recarray
            The chain with stepped positions.
        """
        return numpy.core.records.fromarrays(
            self._chain.T, names=self.params, formats=[float]*len(self.params))

    def append(self, chain):
        """
        Appends array `chain` to `self._chain`.

        Arguments
        ---------
        chain: numpy.ndarray
            Array with newly integrated positions of shape
            `(..., len(self.params))`.

        Returns
        -------
        None
        """

        if self._chain.data.shape[1] != chain.shape[1]:
            raise ValueError("Second axis of `chain` must have length "
                             "`len(self.params)`. Currently `{}`."
                             .format(chain.shape[1]))
        self._chain = numpy.vstack([self._chain, chain])
