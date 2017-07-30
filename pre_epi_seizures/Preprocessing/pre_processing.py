from pre_epi_seizures.storage_utils.storage_utils_hdf5 import \
    load_signal, save_signal

from pre_epi_seizures.storage_utils.data_handlers import *

from pre_epi_seizures.logging_utils.formatter_logging import logger as _logger

from filtering import baseline_removal, create_filtered_dataset

from segmentation import create_rpeak_dataset, create_heart_beat_dataset,\
    compute_beats

from Filtering.gaussian_fit import get_phase, mean_extraction,\
    beat_fitter, ecg_model

from Filtering.filter_signal import filter_signal

from biosppy.signals import ecg

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import logging
import sys
import copy
import memory_profiler


def plot_single_model(time, model, names, color, start, end):
    print 'here0'
    plt.figure()
    print len(model)
    for i in xrange(0, len(model)):
        print model[i,:]
        plt.subplot(len(model), 1, i+1)
        plt.plot(model[i,:], color=color)
        # plt.axvline(x=30*60)
        plt.xlim([start, end])
        plt.ylim([-500, 500])

    plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)

def plot_single_model_fft(time, model, names, color, start, end):
    plt.figure()
    print len(model)
    for i in xrange(0, len(model)):
        plt.subplot(len(model), 1, i+1)
        sig = model[i, start:end]
        yf = sp.fft(sig)
        P = 1.0 / 1000
        xf = np.linspace(0, 1.0 / (2.0 * P), len(sig) // 2)
        plt.plot(
            xf, 2.0 / len(sig) * np.abs(yf[0:len(sig) // 2]), color=color)
        plt.xlim([0, 100])
        plt.ylim([0, 100])


    plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)



def plot_models(times, models, names, colors, start, end):
    plt.figure()
    for ii, (time, model, name) in enumerate(zip(times, models, names), 1):
        plt.subplot(len(models), 1, ii)
        plt.title(name)
        for ts, sig, color in zip(time.T, model.T, colors):
            plt.plot(ts, sig, color=color)
            plt.xlim([start, end])
            plt.ylim([-500, 500])
            # if scatter not None:
            #     plt.scatter(scat, color=scatter_color)

    plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)



def plot_models_scatter(time_scatter_models, scatter_models,
                        time_models, models, names, colors, start, end):
    plt.figure()
    for ii, (time_scatter_model, scatter_model, time_model, model, name)\
        in enumerate(zip(time_scatter_models, scatter_models, 
                         time_models, models, names), 1):
        plt.subplot(len(models), 1, ii)
        plt.title(name)
        for ts_scatter, sig_scatter, ts, sig, color\
            in zip(time_scatter_model.T, scatter_model.T, time_model.T, model.T, colors):
            plt.plot(ts, sig, color=color)
            plt.scatter(ts_scatter, sig_scatter, color='g')
            plt.xlim([start,end])
            plt.ylim([-500,500])
            #     plt.scatter(scat, color=scatter_color)

    plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)


def fft_plot(times, models, names, colors):
    plt.figure()
    for ii, (time, model, name) in enumerate(zip(times, models, names), 1):
        plt.subplot(len(models), 1, ii)
        plt.title(name)
        for ts, sig, color in zip(time.T, model.T, colors):
            yf = sp.fft(sig)
            P = 1.0 / 1000
            xf = np.linspace(0, 1.0 / (2.0 * P), len(sig) // 2)
            plt.plot(
                xf, 2.0 / len(sig) * np.abs(yf[0:len(sig) // 2]), color=color)
            plt.xlim([0, 100])
            plt.ylim([0, 5])

            # if scatter not None:
            #     plt.scatter(scat, color=scatter_color)

    plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)


# def plot_one_model(time, model, names, color)


def shape_array(array):
    return np.array([array]).T


def main():

    #signal
    time_before_seizure = 30
    time_after_seizure = 10
    path_to_load = '~/Desktop/seizure_datasets.h5'
    name_list = [str(time_before_seizure*60) + '_' + str(time_after_seizure*60)]
    group_list = ['raw']

    # Raw signal 
    signal_structure_raw = load_signal(path_to_load,zip(group_list, name_list))
    # one_signal_structure_raw = get_one_signal_structure(signal_structure, zip(group_list, name_list)[0])
    # records_raw = get_multiple_records(one_signal_structure_raw)


    # Baseline removal
    group_list_baseline_removal = ['medianFIR']
    try:
        signal_structure_baseline_removal = load_signal(path_to_load,zip(group_list_baseline_removal, name_list))
    except Exception as e:
        _logger.debug(e)
        baseline_removal(path_to_load, name_list[0], group_list[0])
        signal_structure_baseline_removal = load_signal(path_to_load,zip(group_list_baseline_removal, name_list))



    # create_rpeak_dataset(path_to_load, zip(group_list_baseline_removal, name_list))

    f_loss = lambda label:'rpeaks_'+label

    labels = map(str, range(0,6))

    print labels
    rpeaks_names = map(f_loss, labels)
    print rpeaks_names

    group_list = group_list_baseline_removal * len(labels)
    print group_list

    rpeaks = load_signal(path_to_load ,zip(group_list, rpeaks_names))

    print rpeaks

    stop

    one_signal_structure = get_one_signal_structure(signal_structure_baseline_removal, zip(group_list_baseline_removal, name_list)[0])
    records = get_multiple_records(one_signal_structure)

    signal = records[1,:]

    signal = filter_signal(signal=signal, ftype='FIR', band='lowpass',
                  order=50, frequency=40,
                  sampling_rate=1000)

    rpeaks = ecg.hamilton_segmenter(signal=signal,
                                    sampling_rate=1000)
    print rpeaks
    phase = get_phase(signal, rpeaks['rpeaks'])

    beats = compute_beats(signal, rpeaks['rpeaks'])
    # plt.plot(beats[5])
    # plt.show()

    print phase
    values = beat_fitter(beats[0],phase[1000:2000])

    print values
    stop
    # create_rpeak_dataset(path_to_load, zip(group_list_baseline_removal, name_list))

    # Plot the dataset
    # records_baseline_removal = records
    # Fs = 1000
    # N = len(records_baseline_removal[0, :])
    # T = (N - 1) / Fs
    # time = np.linspace(0, T, N, endpoint=False)
    # names = ['1', '1', '1', '2', '2', '5']
    # color = 'red'
    # start = 0
    # end = 1000
    # plot_single_model(time, beats, names, color, start, end)
    # # plot_single_model_fft(time, np.array([signal]), names, color, start, end)
    # plt.show()
    # stop

    values = beat_fitter(signal[start:end], phase[start:end])
    model = ecg_model(values, phase[start:end])
    print values
    print 'The model is ... '
    print model
    # plt.figure()
    # plt.plot(phase)
    # plt.show()

    start = 0
    end = len(model)
    plot_single_model(time, np.array([model]), names, color, start, end)
    plt.show()

main()