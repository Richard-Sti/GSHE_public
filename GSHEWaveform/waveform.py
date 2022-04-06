from copy import deepcopy

import numpy
from pycbc.waveform.utils import apply_fd_time_shift, fd_to_td

from .utils import mixing, linear_to_circular, circular_to_linear




def gshe_to_circular(hcirc_tilde, time_delay):

    freqs = hcirc_tilde.sample_frequencies.data

    dt = time_delay(freqs)

    hcirc_tilde_gshe = deepcopy(hcirc_tilde)

    hcirc_tilde_gshe.data *= mixing(freqs, dt)

    return hcirc_tilde_gshe


def gshe_to_linear(hptilde, hctilde, right_time_delay, left_time_delay):
    
    hrtilde, hltilde = linear_to_circular(hptilde, hctilde)

    hrtilde_gshe = gshe_to_circular(hrtilde, right_time_delay)
    hltilde_gshe = gshe_to_circular(hltilde, left_time_delay)

    return circular_to_linear(hrtilde_gshe, hltilde_gshe)

    
def fd_to_td_wshift(htilde, left_window=None, right_window=None, tshift=-1):
    htilde = apply_fd_time_shift(htilde, tshift)
    h = fd_to_td(htilde, left_window=left_window, right_window=right_window)
    
    h.start_time -= tshift

    return h

