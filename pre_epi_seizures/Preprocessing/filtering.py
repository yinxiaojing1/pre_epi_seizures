from pre_epi_seizures.logging_utils.formatter_logging\
    import logger as _logger

from pre_epi_seizures.storage_utils.storage_utils_hdf5\
    import save_signal, load_signal, delete_signal

from pre_epi_seizures.storage_utils.data_handlers import *

from Filtering import medianFIR, filterIR5to20,\
    filter_signal, gaussian_fit

from Filtering.eksmoothing import EKSmoothing

import numpy as np
import time

def baseline_removal(signal_arguments, sampling_rate):

    signal_list = signal_arguments['feature_group_to_process']
    _logger.info('Removing the baseline ...')
    feature = 'medianFIR'
    ti = time.time()
    feature_signal_list = [create_filtered_dataset(signal, filtmethod='medianFIR',
            sampling_rate=sampling_rate) for signal in signal_list]
    tf = time.time() - ti
    print tf, 
    print 'seconds'
    stop
    return feature_signal_list, feature



def noise_removal(path, name, group, sampling_rate):
    _logger.info('Removing noise ...')
    create_filtered_dataset(path=path, name=name, group=group, filtmethod='FIR_lowpass_40hz')

def eks_smoothing(path, name, group, sampling_rate):
    try:
        # make_new_rpeaks
        rpeaks_signal_structure = load_signal(path_to_load ,zip(group_list, rpeaks_names))
    except Exception as e:
        print e
        _logger.debug(e)
        create_rpeak_dataset(path_to_load, zip(group_list_baseline_removal, name_list), sampling_rate)
        rpeaks_signal_structure = load_signal(path_to_load ,zip(group_list, rpeaks_names))

def create_filtered_dataset(signal, filtmethod,
                            sampling_rate):
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
        Path pointing to new csv file. 
        If None, replace 'raw' with 'filtmethod'.
        If 'raw' does not exist in dfile name, append '_filtmethod'.
    multicolumn: bool (default: False)
        If the csv file to be loaded has multi-index columns. See
        read_dataset_csv and save_dataset_csv functions.
    kwargs: dict
        Additional arguments to filtmethod function (e.g. fs).
    """
    print signal
    print len(signal)
    if filtmethod == 'medianFIR':
        signal = np.asmatrix(signal)
        X_filt = globals()[filtmethod](signal, fs=500)
        _logger.debug(X_filt)

    elif filtmethod == 'FIR_lowpass_40hz':
        X_filt = globals()['filter_signal'](signal, ftype='FIR', band='lowpass',
                  order=10, frequency=40,
                  sampling_rate=sampling_rate)
        _logger.debug(X_filt)

    print X_filt
    return X_filt
