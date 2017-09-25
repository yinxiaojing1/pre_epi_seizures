from Filtering.eksmoothing import *

import numpy as np


def sameni_evolution(signal_arguments, sampling_rate):
    seizure_list = signal_arguments['feature_group_to_process']
    parameters_list = [map(compute_parameters_sameni, beat_list) for beat_list in seizure_list]
    mdata = ['']*len(parameters_list)
    return parameters_list, mdata


def compute_parameters_sameni(beat):
    # print 'ne swizure'
    phase = np.linspace(-np.pi, np.pi, 1000, endpoint=True)
    beat = beat_fitter(beat, phase)
    # print beat
    return beat

