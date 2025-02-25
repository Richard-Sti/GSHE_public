# Copyright (C) 2022 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
Tools to build the shadow plots.
"""
import numpy
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, KDTree


def ang2shadow_ingoing(X):
    r"""
    Convert the :math:`psi` and :math:`rho` to :math:`k_2` and :math:`k_3` for
    ingoing rays.

    Arguments
    ---------
    X : numpy.ndarray
        Array of :math:`psi` and :math:`rho` of shape (-1 ,2).

    Returns
    -------
    out : numpy.ndarray
        Array of :math:`k_2` and :math:`k_3` of shape (-1 ,2).
    """
    if not (X.ndim == 2 and X.shape[1] == 2):
        raise TypeError("`X` shape must be (-1, 2).")

    psi = X[:, 0]
    rho = X[:, 1]
    return numpy.vstack([numpy.sin(psi) * numpy.sin(rho), numpy.cos(psi)]).T


def read_signed_beta(data):
    """
    Read the absolute value of the beta and sign it according to the the GSHE
    delay.

    Parameters
    ----------
    data : dict
        Data dictionary whose keys must include `betas`, `Xgshe` and `Xgeo`.

    Returns
    -------
    betas : 1-dimensional array
    """
    # Take the highest epsilon time of arrival
    sign = numpy.sign(data["Xgshe"][:, -1, 2] - data["Xgeo"][:, 2])
    return sign * data["betas"][:, 0]


def get_upsilon_src(ks, betas, betalims, dk, magnification=None,
                    beta_sign=None, min_magnification=None):
    r"""
    Calculate :math:`\Upsilon_{\rm src}` as a function of
    :math:`\beta_{\rm lim}`. Sets :math:`\beta` that are NaN to 0.

    Arguments
    ---------
    ks : 2-dimensional array
        Array of :math:`k_2` and :math:`k_3` of shape (-1 ,2).
    betas : 2-dimensional array
        Array of :math:`\beta` as a function of `ks`.
    betalims : 1-dimensional array
        Array of :math:`\beta_{\rm lim}`.
    dk : float
        The :math:`k_2` and :math:`k_3` spacing, expected to be the same for
        both.
    magnification : 2-dimensional array, optional
        Array of magnification as a function of `ks`. By default `None`
    beta_sign : int, optional
        The sign of the :math:`\beta` to be integrated. If `None` both signs
        are considered.
    min_magnification : float, optional
        The minimum absolute value of magnification of a point to be summed.
        By default `None`, i.e. no threshold.

    Returns
    -------
    upsilon : 1-dimensional array
    """
    if not (ks.ndim == 2 and ks.shape[1] == 2):
        raise TypeError("`ks` array shape must be (-1, 2).")
    if not betas.ndim == 1:
        raise TypeError("`betas` must be a 1-dimensional array.")
    if not betalims.ndim == 1:
        raise TypeError("`betalims` must be a 1-dimensional array.")
    assert beta_sign in [-1, 1, None]

    betas = numpy.copy(betas)       # Copy since will be rewriting this array
    betas[numpy.isnan(betas)] = 0.  # Set NaN to 0. Will sum beta > 0 anyway.

    radius2 = numpy.sum(ks**2, axis=1)    # Squared distance from the origin
    mask = radius2 <= 1                   # Mask for the unit sphere
    area = numpy.sqrt(1 / (1 - radius2))  # Area element in spherical coords

    # If integrating both take the absolute value
    if beta_sign is None:
        betas = numpy.abs(betas)
    # If integrating only positive set negative values to zero
    if beta_sign == 1:
        betas[betas < 0] = 0
    # If integrating only negative set positive values to zero and take abs
    if beta_sign == -1:
        betas[betas > 0] = 0
        betas = numpy.abs(betas)
    # Take threshold on magnification. Again set betas below it to 0
    if min_magnification is not None:
        betas[magnification < min_magnification] = 0

    ups = numpy.zeros_like(betalims)
    for i, lim in enumerate(betalims):
        H = (betas > lim).astype(int)
        ups[i] = numpy.sum((H * area)[mask]) * dk**2

    ups /= (2 * numpy.pi)  # Normalise to 2 numpy.pi since only a half sphere
    return ups


def get_upsilon_obs(ks, betas, betalims, dk, magnification, nloops,
                    min_magnification=None, nloop_max=None,
                    magnification_sign=None, beta_sign=None):
    r"""
    Calculate :math:`\Upsilon_{\rm obs}` as a function of
    :math:`\beta_{\rm lim}`. Sets :math:`\beta` that are NaN to 0.

    Arguments
    ---------
    ks : 2-dimensional array
        Array of :math:`k_2` and :math:`k_3` of shape (-1 ,2).
    betas : 2-dimensional array
        Array of :math:`\beta` as a function of `ks`.
    betalims : 1-dimensional array
        Array of :math:`\beta_{\rm lim}`.
    dk : float
        The :math:`k_2` and :math:`k_3` spacing, expected to be the same for
        both.
    magnification : 2-dimensional array
        Array of magnification as a function of `ks`.
    nloops : 2-dimensional array
        Array of the number of loops as a function of `ks`.
    min_magnification : float, optional
        The minimum absolute value of magnification of a point to be summed.
        By default `None`, i.e. no threshold.
    nloop_max : int, optional
        Maximum (inclusive) number of loops to be summed when calculating
        :math:`\Upsilon_{\rm src}`. By default `None`, i.e. no restriction.
    magnification_sign : int, optional
        The sign of magnification whose :math:`\beta` are to be integrated. If
        `None` both signs are considered.

    Returns
    -------
    upsilon : 1-dimensional array
    """
    if not (ks.ndim == 2 and ks.shape[1] == 2):
        raise TypeError("`ks` array shape must be (-1, 2).")
    if not betas.ndim == 1:
        raise TypeError("`betas` must be a 1-dimensional array.")
    if not betalims.ndim == 1:
        raise TypeError("`betalims` must be a 1-dimensional array.")
    assert magnification_sign in [-1, 1, None]
    assert beta_sign in [-1, 1, None]

    # Copy since will be rewriting these arrays
    betas = numpy.copy(betas)
    magnification = numpy.copy(magnification)
    # Set NaN beta to 0. Will sum beta > 0.
    betas[numpy.isnan(betas)] = 0.
    # Set betas with loops more than threshold to 0
    if nloop_max is not None:
        betas[nloops > nloop_max] = 0.
    # Set NaN magnification to infinity so that the area element is 0
    magnification[numpy.isnan(magnification)] = numpy.infty

    # If integrating both take the absolute value
    if beta_sign is None:
        betas = numpy.abs(betas)
    # If integrating only positive set negative values to zero
    if beta_sign == 1:
        betas[betas < 0] = 0
    # If integrating only negative set positive values to zero and take abs
    if beta_sign == -1:
        betas[betas > 0] = 0
        betas = numpy.abs(betas)

    # If summing only pos. magnification set neg. magnification betas to zero
    if magnification_sign == 1:
        betas[magnification < 0] = 0
    # If summing only neg. magnification set pos. magnification betas to zero
    if magnification_sign == -1:
        betas[magnification > 0] = 0
    # Abs. value magnification threshold. Again set betas below it to 0
    if min_magnification is not None:
        betas[numpy.abs(magnification) < min_magnification] = 0

    radius2 = numpy.sum(ks**2, axis=1)    # Squared distance from the origin
    mask = radius2 <= 1                   # Mask for the unit sphere
    # Area element in spherical coords on the far sphere
    area = numpy.sqrt(1 / (1 - radius2)) / numpy.abs(magnification)

    ups = numpy.zeros_like(betalims)  # Preallocate and compute
    for i, lim in enumerate(betalims):
        H = (betas > lim).astype(int)
        ups[i] = numpy.sum((H * area)[mask]) * dk**2

    # OLD: Normalisation: for `nloop = 0`, the full sphere gets covered 3/4
    # times. The half sphere away from the BH + the trajectories that are
    # retrolensed, but don't compute as a full loop.
    # NEW: we now only normalise it with respect to 4numpy.pi!
    ups /= 4 * numpy.pi
    return ups


def build_shadowhull(grid, vals, N=100, dgamma=0.01):
    r"""
    Build a hull demarking the BH shadow boundary. Iterates at :math:`\gamma`
    in the :math:`k_2, k_3` plane and at each :math:`\gamma` finds the
    the innermost point from the `vals` array that is not NaN and builds the
    convex hulls out of these points.

    Arguments
    ---------
    grid : numpy.ndarray
        The :math:`k_2, k_3` array of shape (-1, 2).
    vals : numpy.ndarray
        The grid values of shape (-1, ).
    N : int, optional
        The number of :math:`\gamma` angles. By default 100.
    dgamma: float, optional
        Angular radius around which to search the innermost point.
        By default 0.01.

    Returns
    -------
    hull: scipy.spatial.qhull.ConvexHull
    """
    if not (grid.ndim == 2 and grid.shape[1] == 2):
        raise TypeError("`grid` array shape must be (-1, 2).")
    if not vals.ndim == 1:
        raise TypeError("`vals` must be a 1-dimensional array.")
    # Gamma pivots
    pivs = numpy.linspace(0, 2 * numpy.pi, N)
    radius2 = numpy.sum(grid**2, axis=1)
    ang = numpy.arctan2(grid[:, 1], grid[:, 0])

    R = numpy.zeros_like(pivs)
    m0 = ~numpy.isnan(vals)

    for i in range(N):
        mask = m0 & (numpy.arccos(numpy.cos(ang - pivs[i])) < dgamma)
        R[i] = numpy.min(radius2[mask])**0.5

    X = numpy.vstack([R * numpy.cos(pivs), R * numpy.sin(pivs)]).T
    return ConvexHull(X)


def inhull(X, hull, tol=1e-12):
    """
    Check if a point `X` is in `hull``.

    Arguments
    ---------
    X : numpy.ndarray
        The point of shape (-1, ).
    hull : scipy.spatial.qhull.ConvexHull
        The convex hull.
    tol : float, optional
        The tolerance. By default 1e-12.

    Returns
    -------
    is_inhull : bool
        Whether the point is in the hull.
    """
    if X.ndim != 1:
        raise TypeError("`X` must be 1-dimensional.")
    if not isinstance(hull, ConvexHull):
        raise TypeError("`hull` must be `scipy.spatial.qhull.ConvexHull.")
    return all((numpy.dot(eq[:-1], X) + eq[-1] <= tol)
               for eq in hull.equations)


def fillshadow(grid, vals, hull, tree_kwargs={}):
    """
    Fill NaN values outside of the BH shadow boundary with nearest neighbours'
    values. Returns a copy of `vals`.

    Arguments
    ---------
    grid : numpy.ndarray
        The :math:`k_2, k_3` array of shape (-1, 2).
    vals : numpy.ndarray
        The grid values of shape (-1, ).
    hull : scipy.spatial.qhull.ConvexHull
        The convex hull.
    tree_kwargs : dict
        Kwargs for `scipy.spatial.kdtree.KDTree`

    Returns
    -------
    vals : numpy.ndarray
        The nearest neighbour-interpolated values.
    """
    vals = numpy.copy(vals)

    # Points to be filled within R = 1, are NaN and outside the BH shadow
    nansmask = numpy.isnan(vals)
    fillmask = numpy.sum(grid**2, axis=1) <= 1
    for i in range(grid.shape[0]):
        if fillmask[i]:
            fillmask[i] &= ~inhull(grid[i, :], hull) & nansmask[i]

    # Nearest neighbour filling using the KDTree
    notnansmask = ~nansmask
    tree = KDTree(grid[notnansmask, :], **tree_kwargs)
    for i in numpy.where(fillmask)[0]:
        (__, __), (__, j) = tree.query(grid[i, :], k=2)
        vals[i] = vals[notnansmask][j]
    return vals


def smoothshadow(grid, vals, N, method="linear"):
    r"""
    Render the BH shadow on a :math:`N \times N` dimensional grid.

    Arguments
    ---------
    grid : numpy.ndarray
        The :math:`k_2, k_3` array of shape (-1, 2).
    vals : numpy.ndarray
        The grid values of shape (-1, ).
    N : int
        The grid size.
    method : str, optional
        The interpolation method of `scipy.interpolate.griddata`.
        By default "linear".

    Returns
    -------
    X, Y, Z : numpy.ndarray(s)
        Arrays of shape (N, N)
    """
    k2 = numpy.linspace(-1, 1, N)
    k3 = numpy.linspace(-1, 1, N)

    X, Y = numpy.meshgrid(k2, k3)
    highgrid = numpy.vstack([x.reshape(-1, ) for x in (X, Y)]).T
    return X, Y, griddata(grid, vals, highgrid, method=method,
                          rescale=True).reshape(X.shape)
