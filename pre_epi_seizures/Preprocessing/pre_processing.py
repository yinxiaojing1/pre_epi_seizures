
from pre_epi_seizures.storage_utils.storage_utils_hdf5 import  load_signal, save_signal

from filtering import baseline_removal, create_filtered_dataset

from segmentation import create_rpeak_dataset, create_heart_beat_dataset

from formatter_logging import logger

import matplotlib.pyplot as plt
import numpy as np
import logging
import sys

_logger = logging.getLogger(__name__)


def plot_models(models, names, colors):
    plt.figure()
    for ii, (model, name) in enumerate(zip(models, names), 1):
        plt.subplot(len(models), 1, ii)
        plt.title(name)
        for sig, color in zip(model.T, colors):
            _logger.debug(sig)
            plt.plot(sig, color=color)
            # if scatter not None:
            #     plt.scatter(scat, color=scatter_color)

    plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
    plt.show()


def main(arg):

    logger.debug("Starting Gaussian Fit...")
    logger.debug("vlfsn")
    path = '~/Desktop/phisionet_dataset.h5'
    name = 'sz_'+str(arg)
    group = 'medianFIR'

    baseline_removal(path, name, 'raw')
    create_rpeak_dataset(path, name, group)


    X = load_signal(path, name, group)
    X_raw = load_signal(path, name, 'raw')
    rpeaks = load_signal(path,'rpeaks_'+name, group)
    _logger.debug(rpeaks)
    _logger.debug(np.shape(rpeaks['signal']))

    heart_beat, rpeak = create_heart_beat_dataset(path=path,
                                                  name=name,
                                                  group=group,
                                                  save_dfile=None)

    _logger.debug(np.asmatrix(rpeaks['signal'].T))
    models = [X['signal'].T[0:40*200], X_raw['signal'][0:40*200]]
    names = ['Filtered', 'Raw' ]
    colors = ['red']
    plot_models(models, names, colors)

main(6)