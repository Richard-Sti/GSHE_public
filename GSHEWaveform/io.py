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
Functions to read Julia outputs.
"""
from os.path import isfile, join

import numpy


def read_shooting(runID, fpath, betathreshold=numpy.infty, verbose=False):
    """
    Read in data and remove points with std(beta) / mean(beta) > betathreshold.

    Arguments
    ---------
    runID : str
        The run ID.
    fpath : str
        The data folder.
    betathreshold : float
        The beta standard deviation to mean threshold.
    verbose : bool, optional
        Verbosity flag to print the description.

    Returns
    -------
    data : dict
        The data dictionary.
    """
    folder = join(fpath, "run_{}".format(runID))
    if verbose:
        with open(join(folder, "Description.txt")) as f:
            print(f.read())

    data = {"Xgeo": numpy.load(join(folder, "Xgeos.npy")),
            "Xgshe": numpy.load(join(folder, "Xgshes.npy")),
            "eps": numpy.load(join(folder, "Epsilons.npy")),
            "alphas": numpy.load(join(folder, "alphas.npy")),
            "betas": numpy.load(join(folder, "betas.npy")),
            "xs": numpy.load(join(folder, "dir1.npy")),
            "ys": numpy.load(join(folder, "dir2.npy"))}

    if isfile(join(folder, "Xgeos_mu.npy")):
        data.update({"Xgeo": numpy.load(join(folder, "Xgeos_mu.npy"))})

    if isfile(join(folder, "Xgshes_mu.npy")):
        data.update({"Xgshe": numpy.load(join(folder, "Xgshes_mu.npy"))})

    # Calculate the grid
    grid = numpy.vstack([x.reshape(-1, )
                         for x in numpy.meshgrid(data["xs"], data["ys"])]).T
    data.update({"grid": grid})

    notnans = ~numpy.isnan(data["betas"][:, 0])
    # Check betas
    betamask = (data["betas"][:, 1] / data["betas"][:, 0]) < betathreshold
    beta_delmask = ~betamask & notnans
    print("Eliminating {} point due to beta with average {:.4f} loops."
          .format(numpy.sum(beta_delmask),
                  numpy.mean(data["Xgshe"][beta_delmask, -1, 5])))
    # Set bad points to NaN
    for p in ["alphas", "betas"]:
        data[p][beta_delmask, ...] = numpy.nan
    return data
