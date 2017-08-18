from Filtering.eksmoothing import *

import numpy as np


def compute_parameters_sameni(beat):
    N = len(beat)
    phase = compute_phase(beat=beat, bins=N)
    values = beat_fitter(beat, phase)
    return values


def compute_phase(beat, bins):
    phase = np.linspace(-np.pi, np.pi, bins)
    return phase


def sameni_evolution(beats):
    return map(compute_parameters_sameni, beats)
