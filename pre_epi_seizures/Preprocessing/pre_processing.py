from pre_epi_seizures.storage_utils.storage_utils_hdf5 import \
    load_signal, save_signal

from pre_epi_seizures.storage_utils.data_handlers import *

from pre_epi_seizures.logging_utils.formatter_logging import logger as _logger

from filtering import baseline_removal, noise_removal, create_filtered_dataset

from segmentation import create_rpeak_dataset, create_heart_beat_dataset,\
    compute_beats

from Filtering.gaussian_fit import get_phase, mean_extraction,\
    beat_fitter, ecg_model

from Filtering.filter_signal import filter_signal

from Filtering.eksmoothing import EKSmoothing

from resampling import resample_rpeaks

from visual_inspection import visual_inspection

from biosppy.signals import ecg

import time
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
    # path_to_load = '~/Desktop/phisionet_seizures.h5'
    sampling_rate = 1000
    path_to_load = '~/Desktop/seizure_datasets.h5'
    name_list = [str(time_before_seizure*60) + '_' + str(time_after_seizure*60)]
    group_list_raw = ['raw']

    # Raw signal 
    signal_structure_raw = load_signal(path_to_load,zip(group_list_raw, name_list))
    # one_signal_structure_raw = get_one_signal_structure(signal_structure, zip(group_list, name_list)[0])
    # records_raw = get_multiple_records(one_signal_structure_raw)

    # stop # ========================================================

    # Baseline removal
    group_list_baseline_removal = ['medianFIR']
    try:
        signal_structure_baseline_removal = load_signal(path_to_load,zip(group_list_baseline_removal, name_list))
    except Exception as e:
        _logger.debug(e)
        baseline_removal(path_to_load, name_list[0], group_list_raw[0], sampling_rate)
        signal_structure_baseline_removal = load_signal(path_to_load,zip(group_list_baseline_removal, name_list))

    # stop # =========================================================

    # Noise removal 
    group_list_noise_removal = ['FIR_lowpass_40hz']
    try:
        signal_structure_noise_removal = load_signal(path_to_load,zip(group_list_noise_removal, name_list))
    except Exception as e:
        print e
        _logger.debug(e)
        noise_removal(path_to_load, name_list[0], group_list_raw[0], sampling_rate)
        signal_structure_noise_removal = load_signal(path_to_load,zip(group_list_noise_removal, name_list))

    # stop # =========================================================

    # Rpeak detection
    # create_rpeak_dataset(path_to_load, zip(group_list_noise_removal, name_list))
    names = lambda label: name_list[0] + '_rpeaks_'+label
    labels = map(str, range(0,5))
    rpeaks_names = map(names, labels)
    group_list = group_list_baseline_removal * len(labels)

    try:
        make_new_rpeaks
        rpeaks_signal_structure = load_signal(path_to_load ,zip(group_list, rpeaks_names))
    except Exception as e:
        print e
        _logger.debug(e)
        create_rpeak_dataset(path_to_load, zip(group_list_noise_removal, name_list), sampling_rate)
        rpeaks_signal_structure = load_signal(path_to_load ,zip(group_list, rpeaks_names))

    # stop # =========================================================

    # Visual inspection of rpeak detection
    patient_nr = 4

    one_signal_structure = get_one_signal_structure(signal_structure_baseline_removal, zip(group_list_baseline_removal, name_list)[0])
    records = get_multiple_records(one_signal_structure)
    mdata = get_mdata_dict(one_signal_structure)

    # stop
    one_signal_structure = get_one_signal_structure(rpeaks_signal_structure, zip(group_list, rpeaks_names)[patient_nr])
    rpeaks = [get_multiple_records(get_one_signal_structure(rpeaks_signal_structure, group_name)) for group_name in zip(group_list, rpeaks_names)]


    Fs = sampling_rate
    N = len(records[0,:])
    T = (N - 1) / Fs
    t = np.linspace(0, T, N, endpoint=False)
    print time

    rpeaks_resample = resample_rpeaks(np.diff(rpeaks[patient_nr]), rpeaks[patient_nr], t)
    # rpeaks = rpeaks[patient_nr]
    # signal = records[patient_nr,:]

    # visual_inspection(signal, rpeaks, rpeaks_resample, time_before_seizure,
    #             0, time[-1], sampling_rate)

    # signal_to_filter = signal
    # rpeaks = ecg.hamilton_segmenter(signal=signal_to_filter,
    #                                 sampling_rate=sampling_rate)
    tmp = time.time()
    filtered = EKSmoothing(records, rpeaks,
        fs=sampling_rate, bins=250, verbose=False, 
        oset=False, savefolder=None)
    s = time.time() - tmp

    path_to_load = '~/Desktop/seizure_datasets.h5'
    name_list = [str(time_before_seizure*60) + '_' + str(time_after_seizure*60)]
    group_list_esksmooth = ['esksmooth']

    save_signal(path=path, signal_list=filtered,
                mdata_list=[mdata], name_list=name_list, group_list=group_list_esksmooth)

    print 'seconds elapsed:' 
    print s

    # print filtered[0]
    # plt.subplot(2,1,1)
    # plt.plot(signal_to_filter)
    # plt.subplot(2,1,2)
    # plt.plot(filtered[0])
    # plt.show()

    stop


    # print rpeaks
    # phase = get_phase(signal, rpeaks)

    # mnx, sdx, mnphase = mean_extraction(signal[0:5*60*1000], phase[0:5*60*1000], bins = 1000)

    # print mnx

    # print phase
    # values = beat_fitter(mnx,mnphase)

    # print values
    # model = ecg_model(values, mnphase)

    # filtered = EKSmoothing(signal, [rpeaks['rpeaks']], fs=1000., bins=250, verbose=True, oset=False, savefolder=None)

    plt.figure()
    # plt.plot(mnx)
    plt.plot(filtered)


    # beats = compute_beats(signal, rpeaks)
    # plt.figure()
    # plt.plot(beats[beat_nr])

    # print phase
    # values = beat_fitter(beats[beat_nr],phase[beat_nr*1000:(beat_nr+1)*1000])

    # print values
    # model = ecg_model(values, phase[beat_nr*1000:(beat_nr+1)*1000])

    # plt.plot(model)
    # plt.show()

main()