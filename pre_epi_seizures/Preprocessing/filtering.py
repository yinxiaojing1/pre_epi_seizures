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
    # signal = np.asmatrix(signal_list)
    # signal = np.asarray(signal_list)
    feature_mdata = [{'fs': sampling_rate}] * len(signal_list)
    print feature_mdata

    feature_signal_list = [create_filtered_dataset(signal, filtmethod='medianFIR',
            sampling_rate=sampling_rate) for signal in signal_list]

    return feature_signal_list, feature_mdata



def noise_removal(path, name, group, sampling_rate):
    _logger.info('Removing noise ...')
    create_filtered_dataset(path=path, name=name, group=group, filtmethod='FIR_lowpass_40hz')


def eks_smoothing(signal_arguments, sampling_rate):
   signal_list = signal_arguments['feature_group_to_process']
   rpeak_list = signal_arguments['rpeak_group_to_process']
   print rpeak_list
   filtered = EKSmoothing(signal_list, rpeak_list, fs=sampling_rate)
   mdata_list = [{'fs': sampling_rate}]*len(filtered)
   return filtered, mdata_list


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
    if filtmethod == 'medianFIR':
        print np.shape(signal)
        print type(signal)
        print signal
        ti = time.time()
        X_filt = globals()[filtmethod](signal, fs=1000)
        tf = time.time() - ti
        print tf, 
        print 'seconds'
        _logger.debug(X_filt)

    elif filtmethod == 'FIR_lowpass_40hz':
        X_filt = globals()['filter_signal'](signal, ftype='FIR', band='lowpass',
                  order=10, frequency=40,
                  sampling_rate=sampling_rate)
        _logger.debug(X_filt)




    print X_filt
    return X_filt
