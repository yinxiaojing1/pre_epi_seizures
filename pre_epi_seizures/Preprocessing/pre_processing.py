from biosppy import storage as st_hdf5

from Filtering import medianFIR, filterIR5to20, filter_signal, gaussian_fit

from biosppy.signals import ecg

import matplotlib.pyplot as plt
import numpy as np
import logging
import sys

_logger = logging.getLogger(__name__)


def baseline_removal(path, name, group):
    create_filtered_dataset(path=path, name=name, group=group, filtmethod='medianFIR', save_dfile=path)


def create_filtered_dataset(path, name, group, filtmethod='medianFIR', save_dfile=None, multicolumn=False, **kwargs):
    """
    Load dataset from a hdf5 file, filter with filtmethod save it.
    Parameters:
    -----------
    dfile: str
        Path pointing to csv file containing 1 record per column.
    filtmethod: str (default: 'medianFIR')
        Name of the filter function defined in the global scope. For instance,
        'medianFIR' or 'filterIR5to20'.
    save_dfile: str (default: None)
        Path pointing to new csv file. If None, replace 'raw' with 'filtmethod'.
        If 'raw' does not exist in dfile name, append '_filtmethod'.
    multicolumn: bool (default: False)
        If the csv file to be loaded has multi-index columns. See
        read_dataset_csv and save_dataset_csv functions.
    kwargs: dict
        Additional arguments to filtmethod function (e.g. fs).
    """
    if save_dfile is None:
        save_dfile = dfile.replace('raw', filtmethod) if 'raw' in dfile else dfile.split('.')[0] + '_{}.csv'.format(filtmethod)
    X = load_signal(path=path, name=name, group=group) # 1 record per row
    X_filt = globals()[filtmethod](X['signal'].T, **kwargs)
    save_signal(signal=X_filt, mdata=X['mdata'], path=path, name=name, group=filtmethod)


def plot_models(models, names, colors):
    plt.figure()
    for ii, (model, name) in enumerate(zip(models, names), 1):
        plt.subplot(len(models), 1, ii)
        plt.title(name)
        for sig, color in zip(model.T, colors):
            plt.plot(sig, color=color)

    plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
    plt.show()


def load_signal(path, name, group):
    opened_file = st_hdf5.HDF(path, 'r+')

    try:
        signal = opened_file.get_signal(name=name, group=group)
    except Exception as e:
        _logger.debug(e)

    opened_file.close()
    return signal


def save_signal(signal, mdata, path, name, group):
    opened_file = st_hdf5.HDF(path, 'r+')

    try:
        signal = opened_file.add_signal(signal=signal, mdata=mdata, name=name, group=group)
    except Exception as e:
        _logger.debug(e)

    opened_file.close()


def fetch_phisionet_path(nb_sz):
    return '~/Desktop/phisionet_dataset.h5'


def create_rpeak_dataset(path, name, group, save_dfile=None):
    X = load_signal(path=path, name=name, group=group) # 1 record per row
    signal_to_filter = X['signal'][0,:].T
    rpeaks = ecg.hamilton_segmenter(signal=signal_to_filter, sampling_rate=X['mdata']['fs'])
    save_signal(signal=rpeaks['rpeaks'], mdata='', path=path, name='rpeaks_'+name, group=group)




def setup_logging(loglevel = 'INFO'):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def main(*args):
    setup_logging('DEBUG')
    _logger.debug("Starting Gaussian Fit...")
    path = '~/Desktop/phisionet_dataset.h5'
    

# main(1, 2, 3, 4, 5, 6, 7)