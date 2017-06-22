from biosppy import storage as st_hdf5
import preprocessing
from Filtering import medianFIR, filterIR5to20, filter_signal, gaussian_fit
from rpeak_detector import rpeak_detector

import matplotlib.pyplot as plt
import numpy as np
import logging
import sys

_logger = logging.getLogger(__name__)


def baseline_removal(path, name, group):
    setup_logging('DEBUG')
    nb_sz = 1
    name = 'sz_'+str(nb_sz)
    path = fetch_phisionet_path(nb_sz)
    group = 'raw'
    # models = [ecg_raw, X.T]
    # names = ['Raw ECG', 'Filtered']
    # colors = ['Red']
    # plot_models(models, names, colors)
    _logger.debug('Creating fitered dataset')
    create_filtered_dataset(dfile=path, name=name, group=group, filtmethod='medianFIR', save_dfile=path)


def create_filtered_dataset(dfile, name, group, filtmethod='medianFIR', save_dfile=None, multicolumn=False, **kwargs):
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
    X = fetch_signal(path=dfile, name=name, group=group) # 1 record per row
    X_filt = globals()[filtmethod](X['signal'].T, **kwargs)
    save_signal(signal=X_filt, mdata=X['mdata'], path=save_dfile, name=name, group=filtmethod)


def plot_models(models, names, colors):
    plt.figure()
    for ii, (model, name) in enumerate(zip(models, names), 1):
        plt.subplot(len(models), 1, ii)
        plt.title(name)
        for sig, color in zip(model.T, colors):
            plt.plot(sig, color=color)

    plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
    plt.show()


def fetch_signal(path, name, group):
    opened_file = st_hdf5.HDF(path, 'r+')
    signal = opened_file.get_signal(name=name, group=group)
    opened_file.close()
    return signal


def save_signal(signal, mdata, path, name, group):
    opened_file = st_hdf5.HDF(path, 'r+')
    signal = opened_file.add_signal(signal=signal, mdata=mdata, name=name, group=group)
    opened_file.close()


def fetch_phisionet_path(nb_sz):
    return '~/Desktop/phisionet_dataset.h5'


def dataset_segmentation(dfile, name, group, save_dfile=None, rpeaks_list=None, rdetector='UNSW_RPeakDetector', lim=[-100, 250], **kwargs):
    """
    Load dataset from csv file, apply ECG segmentation using fixed-sized window around detected R peaks,
    save the resulting segments into a csv (multi-index dataframe).
    The R peaks are detected using rdetector function.
    Parameters:
    -----------
    dfile: str
        Path pointing to csv file containing 1 record per column.
    save_dfile: str (default: None)
        Path pointing to new csv file. If None, append '_segments' to dfile name.
    rpeaks_list: list of arrays (default: None)
        R peak locations (1 array per signal). If None, computes the rpeaks using rdetector.
    rdetector: str (default: UNSW_RPeakDetector)
        Name of the R peak detector function defined in the global scope. For instance,
        'UNSW_RPeakDetector'.
    lim: list of int (default: [-100, 250])
        Lower and upper bound w.r.t detected R peaks [number of samples].
    kwargs: dict
        Additional arguments to rdetector function (e.g. fs).
    """
    signal = fetch_signal(path=dfile, name=name, group=group)
    X = signal['signal']
    y = [1]

    # rpeaks_list = globals()[rdetector](X, **kwargs) if rpeaks_list is None else rpeaks_list
    rpeak_list = rpeak_detector(signal=X,
                                sampling_rate=signal['fs'], method='christov')
    X_new, y_new = [], []
    lb, ub = lim
    ssize = ub-lb
    for xx, yy, Rpeaks in zip(X, y, rpeaks_list):
        if len(Rpeaks)==0:
            continue
        xx_segments = np.vstack([xx[lb+rpeak:ub+rpeak] for rpeak in Rpeaks if len(xx[lb+rpeak:ub+rpeak])==ssize])
        y_new.append([yy]*len(xx_segments))
        X_new.append(xx_segments)
    y_new = np.array(list(itertools.chain.from_iterable(y_new)))
    if len(y_new) == 0:
        raise ValueError('Could not perform segmentation on any record.')
    if save_dfile is None:
        save_dfile = dfile.split('.')[0] + '_segments.csv'
    save_dataset_csv(X_new, y_new, save_dfile)



def setup_logging(loglevel = 'INFO'):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def main(arg):
    setup_logging('DEBUG')
    _logger.debug("Starting crazy calculations...")
    path = '~/Desktop/phisionet_dataset.h5'

    # try:
    #     baseline_removal(path=path, name='sz_'+str(arg),
    #                      group = 'raw')
    # except Exception as e:
    #     _logger.debug(e)

    X = fetch_signal(path = path, name='sz_'+str(arg),
                     group = 'raw')

    rpeak = rpeak_detector(signal=X['signal'].T, sampling_rate=X['mdata']['fs'], method='christov')

    gaussian_fit(X['signal'], rpeak)
    _logger.debug(header)
    _logger.debug(signals['signal'])


main(2)