from pre_epi_seizures.logging_utils.formatter_logging\
    import logger as _logger

from pre_epi_seizures.storage_utils.storage_utils_hdf5\
    import save_signal, load_signal, delete_signal

from pre_epi_seizures.storage_utils.data_handlers import *

# from pre_processing import \
#         input_default_params

from Filtering import medianFIR, filterIR5to20,\
    filter_signal, gaussian_fit

from Filtering.eksmoothing import EKSmoothing

import numpy as np
import time




def baseline_removal(signal_arguments, window_params, add_params, win_param_to_process, param_to_process):
    signal_list = signal_arguments['feature_group_to_process']
    _logger.info('Removing the baseline ...')
    feature = 'medianFIR'
    # signal = np.asmatrix(signal_list)
    # signal = np.asarray(signal_list)

    # default_win_params = dict()
    # default_add_params = dict()
    # # -------------------------------
    # # Default win params (comment to override)
    # default_win_params['win'] = 0.001
    # default_win_params['init'] = 0
    # default_win_params['finish'] = 4200
    # default_win_params['samplerate'] = 1000

    # # -------------------------------
    # # Default add params (comment to override)
    # default_add_params['filt'] = 'medianFIR'
    # default_add_params['init'] = 0
    # default_add_params['finish'] = 4200

    # default_win_params = input_default_params(win_params,
    #                         win=0.001,
    #                         init=0,
    #                         finish=4200,
    #                         samplerate=1000)

    # default_add_params = input_default_params(win_params,
    #                         filt='medianFIR')


    # ------------------------------ # TEMPORARY
    try:
        # Compute feature_array_list
        sampling_rate = window_params['samplerate']
        filtmethod = add_params['filt']
        init = window_params['init']
        finish = window_params['finish']
        feature_signal_list = [np.asarray(create_filtered_dataset(signal, filtmethod='medianFIR',
                               sampling_rate=sampling_rate)) for signal in signal_list]
        # No resampling is made --- Change needed if resampling different
        #(check HRV code - template for resampling signals, very good)

        # Compute time domain
        feature_window_list = [np.linspace(init,
                                           finish,
                                           (finish - init) * sampling_rate)]
        feature_window_list = feature_window_list * len(feature_signal_list)

        # Get feature names
        mdata_list = [{'feature_legend': ['baseline_removal']}] * len(feature_signal_list)
    
    except Exception as e:
        print e
        feature_signal_list = [[]]
        feature_window_list = [[]]
        mdata_list = [{'feature_legend': ['baseline_removal']}] * len(feature_signal_list)
        
        

    # Return the objects
    return feature_signal_list, mdata_list, feature_window_list



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
        X_filt = globals()[filtmethod](signal, fs=1000)
        
    elif filtmethod == 'FIR_lowpass_40hz':
        X_filt = globals()['filter_signal'](signal, ftype='FIR', band='lowpass',
                  order=10, frequency=40,
                  sampling_rate=sampling_rate)
        _logger.debug(X_filt)

    return X_filt


def visual_inspection(raw_signal_list,
                      filtered_signal_list,
                      begin_sec, end_sec):
    import matplotlib.pylot as plt
    
    for raw_signal, filtered_signal in zip(raw_signal_list,
                                           filtered_signal_list):
        plt.figure(figsize=(20, 20))
        plt.plot(raw_signal.T)
        plt.plot(filterd_signal.T)
        plt.xlim(begin_sec * 1000, end_sec * 1000)
        plt.legend(['raw', 'filtered'])
        plt.show()
        
