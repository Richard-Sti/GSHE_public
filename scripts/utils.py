import numpy
import h5py
from os.path import join
from glob import glob
try:
    import GSHEWaveform
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import GSHEWaveform


def get_liv_fnames(loaddir):
    files = glob(join(loaddir, "liv_*"))
    files = [f.split("/")[-1] for f in files]
    files = [f for f in files if "Aminus" in f]

    names = [f.split("liv_")[1].split("_")[0] for f in files]
    return files, names


def A0_from_liv_samples(filename):
    """
    Read in the Lorentz-violation file and read off `A0`.
    """
    f = h5py.File(filename)

    isfound = False
    for key in f.keys():
        if "alpha" in key:
            isfound = True
            break

    if not isfound:
        return None

    samples = numpy.copy(f[key]["posterior_samples"]) if isfound else None
    f.close()
    return GSHEWaveform.A0_from_sample(
        10**samples["log10lambda_eff"], samples["redshift"],
        samples["luminosity_distance"])
