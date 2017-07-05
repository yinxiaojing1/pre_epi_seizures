from pre_epi_seizures.storage_utils.storage_utils_hdf5 import  load_signal, save_signal

from pre_epi_seizures.logging_utils.formatter_logging import logger as _logger

from filtering import baseline_removal, create_filtered_dataset

from segmentation import create_rpeak_dataset, create_heart_beat_dataset

from Filtering.gaussian_fit import get_phase, mean_extraction


import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import logging
import sys
import copy
import memory_profiler


def plot_models(times, models, names, colors):
    plt.figure()
    for ii, (time, model, name) in enumerate(zip(times, models, names), 1):
        plt.subplot(len(models), 1, ii)
        plt.title(name)
        for ts, sig, color in zip(time.T, model.T, colors):
            plt.plot(ts, sig, color=color)
            # if scatter not None:
            #     plt.scatter(scat, color=scatter_color)

    plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
    plt.show()


def shape_array(array):
    return np.array([array]).T


# @profile
def main(arg):

    # _logger.debug("Starting Gaussian Fit...")
    # _logger.debug("vlfsn")

# #*****************phisionet data*******************************
#     path = '~/Desktop/phisionet_dataset.h5'
#     name = ['sz_'+str(arg)]
#     group = ['medianFIR', 'raw']

#     # # baseline_removal(path, name, 'raw')
#     # # create_rpeak_dataset(path, name, group)
#     X = load_signal(path, name, group)

#     # path = '~/Desktop/dummy.h5'
#     # # rpeaks = load_signal(path,'rpeaks_'+name, group)

#     _logger.debug(X[0]['signal'])
#     _logger.debug(X[0]['mdata'])

    # heart_beat, rpeak = create_heart_beat_dataset(path=path,
    #                                               name=name,
    #                                               group=group,
    #                                               save_dfile=None)

    # _logger.debug(np.asmatrix(rpeaks['signal'].T))
    # models = [X['signal'][40*200:60*200], X_raw['signal'][40*200:60*200]]
    # names = ['Filtered', 'Raw' ]
    # colors = ['red']
    # plot_models(models, names, colors)

#****************************************************************
    # path = '~/Desktop/HSM_data.h5'
    # name = ['FA77748S']
    # group = ['/PATIENT1/crysis']

    # raw = load_signal(path, name, group)

    # begin_seizure_seconds = raw[0]['mdata']['crysis_time_seconds']
    # begin_seizure_sample = int(1000*begin_seizure_seconds[0])

    # sampling_rate_hertz = 1000

    # signal = raw[0]['signal']
    # no_seizure_ecg_raw = signal[0:begin_seizure_sample,0]

    # ecg_10min_5min_raw = signal[begin_seizure_sample
    #                              -sampling_rate_hertz*10*60:
    #                              begin_seizure_sample
    #                              + sampling_rate_hertz*5*60]


    path_to_save = '~/Desktop/seizures_datasets_new.h5'
    # mdata_list = [raw[0]['mdata']]
    # signal_list = [ecg_10min_5min_raw]
    name_list = ['10_15', 'rpeaks_10_15']
    group_list= ['raw', 'medianFIR']

    # save_signal(path_to_save, signal_list,
                # mdata_list, name_list, group_list)
    # ecg_10min_5min_raw = X[0]['signal']
    # ecg_10min_5min_medianFIR = X[1]['signal']


    # create_rpeak_dataset(path_to_save, name, group)
    # _logger.debug(begin_seizure_sample)

    X = load_signal(path_to_save, name_list, group_list)
    

    signal = X[2]['signal'].T
    rpeaks = X[3]['signal'].T

    _logger.debug(signal[0,rpeaks])
    _logger.debug(X[2]['mdata'])

    # phase = get_phase(signal[0,:], rpeaks[0,:])
    # _logger.debug(phase)
    # # mean_extraction(signal, phase)

    # baseline_removal(path_to_save, name_list, group_list)
    # create_rpeak_dataset(path_to_save, name_list[0], group_list[1])

    # decimated = sp.signal.decimate(raw[0]['signal'], 2)
    # stop
    

    # baseline_removal(path_to_save, ['10_15'], ['raw'])

    Fs = X[2]['mdata']['sample_rate']

    N = len(signal[0,:])
    T = (N - 1) / Fs

    time = np.linspace(0, T, N, endpoint=False)
    _logger.debug(len(time))
    _logger.debug(N)

    time = np.array([time])

    _logger.debug('time %s', time)
    _logger.debug('time %s', signal)

    start = 105*1000
    end = 110*1000
    times = [time.T]
    models = [signal.T]
    names = ['raw', 'medianFIR', 'rpeaks']
    colors = ['red']
    plot_models(times, models, names, colors)

main(3)