import numpy
from os.path import join


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
