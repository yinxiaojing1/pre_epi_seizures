from pre_epi_seizures.storage_utils.storage_utils_hdf5 import \
    load_signal, save_signal, delete_signal, list_group_signals

from pre_epi_seizures.storage_utils.data_handlers import *

from pre_epi_seizures.logging_utils.formatter_logging import logger as _logger

from filtering import baseline_removal, noise_removal, create_filtered_dataset

from segmentation import create_rpeak_dataset, create_heart_beat_dataset,\
    compute_beats, find_rpeaks, detect_rpeaks

from resampling import resample_rpeaks, interpolate_signal

from visual_inspection import visual_inspection

from morphology import *
# from Filtering.gaussian_fit import get_phase, mean_extraction,\
#     beat_fitter, ecg_model

from Filtering.filter_signal import filter_signal

from Filtering.eksmoothing import *


from biosppy.signals import ecg

import time
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import logging
import sys
import copy
import memory_profiler
import functools

def compress(path_to_load, group_name_list):
    print 'before loading' 
    print group_name_list

    # Memory loop (one signal at a time)
    for i, group_name in enumerate(group_name_list):
    signal_structure = load_signal(path_to_load, [group_name])
    one_signal_structure = get_one_signal_structure(signal_structure, group_name)
    record = get_multiple_records(one_signal_structure)


def load_feature(path_to_load, group_to_extract, feature_name, compute_flag=False, even_compress=True):

    group_to_load = [group_to_extract+feature_name]
    print list_group_signals(path_to_load, group_to_load)
    stop
    try:
        load_signal(path_to_load, )
    print list_group_signals(path_to_load, group_to_extract[0])
    stop
    group_file = [group_to_extract+feature_name]
    try:
        if compute_flag:
            make_new_computation
        signal_structure = load_signal(path_to_load,zip(group_file, ['features']))

    except Exception as e:
        _logger.debug(e)
        baseline_removal(path_to_load, name_list[0], group_list_to_filter[0], sampling_rate)
        signal_structure_baseline_removal = load_signal(path_to_load,zip(group_list_baseline_removal, name_list))

    one_signal_structure = get_one_signal_structure(signal_structure_baseline_removal, zip(group_list_baseline_removal, name_list)[0])
    records = get_multiple_records(one_signal_structure)
    mdata = get_mdata_dict(one_signal_structure)
    return records, mdata


def load_baseline_removal(path_to_load, group_list_to_filter, name_list, sampling_rate, compute_flag = False):
    group_list_baseline_removal = ['medianFIR']
    try:
        if compute_flag:
            make_new_computation
        signal_structure_baseline_removal = load_signal(path_to_load,zip(group_list_baseline_removal, name_list))
    except Exception as e:
        _logger.debug(e)
        baseline_removal(path_to_load, name_list[0], group_list_to_filter[0], sampling_rate)
        signal_structure_baseline_removal = load_signal(path_to_load,zip(group_list_baseline_removal, name_list))

    one_signal_structure = get_one_signal_structure(signal_structure_baseline_removal, zip(group_list_baseline_removal, name_list)[0])
    records = get_multiple_records(one_signal_structure)
    mdata = get_mdata_dict(one_signal_structure)
    return records, mdata


def load_kalman(path_to_load, group_list_to_filter, name_list, sampling_rate, compute_flag = False):
    group_list_esksmooth = ['esksmooth']
    try:
        if compute_flag:
            make_new_computation
        signal_structure_baseline_removal = load_signal(path_to_load,zip(group_list_esksmooth, name_list))
    except Exception as e:
        _logger.debug(e)
        pass

    one_signal_structure = get_one_signal_structure(signal_structure_baseline_removal, zip(group_list_esksmooth, name_list)[0])
    records = get_multiple_records(one_signal_structure)
    mdata = get_mdata_dict(one_signal_structure)
    print 'records'
    print records
    return records, mdata


def load_noise_removal(path_to_load, group_list_to_filter, name_list, sampling_rate, compute_flag = False):
    group_list_noise_removal = ['FIR_lowpass_40hz']
    try:
        if compute_flag:
            make_new_computation
        signal_structure_noise_removal = load_signal(path_to_load,zip(group_list_noise_removal, name_list))
    except Exception as e:
        print e
        _logger.debug(e)
        noise_removal(path_to_load, name_list[0], group_list_to_filter[0], sampling_rate)
        signal_structure_noise_removal = load_signal(path_to_load,zip(group_list_noise_removal, name_list))

    one_signal_structure = get_one_signal_structure(signal_structure_noise_removal, zip(group_list_noise_removal, name_list)[0])
    records = get_multiple_records(one_signal_structure)
    mdata = get_mdata_dict(one_signal_structure)
    return records, mdata


def load_rpeaks(path_to_load, group_list_to_filter, name_list, sampling_rate, compute_flag = False):
    names = lambda label: name_list[0] + '_rpeaks_'+label
    labels = map(str, range(0,5))
    rpeaks_names = map(names, labels)
    group_list = group_list_to_filter * len(labels)

    try:
        if compute_flag:
            make_new_computation
        signal_structure_rpeaks = load_signal(path_to_load ,zip(group_list, rpeaks_names))
    except Exception as e:
        print e
        _logger.debug(e)
        create_rpeak_dataset(path_to_load, zip(group_list_to_filter, name_list), sampling_rate)
        signal_structure_rpeaks = load_signal(path_to_load ,zip(group_list, rpeaks_names))

    rpeaks = [get_multiple_records(get_one_signal_structure(signal_structure_rpeaks, group_name)) for group_name in zip(group_list, rpeaks_names)]
    return rpeaks


def plot_single_model(time, model, color, start, end):
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
    # path_to_load = '~/Desktop/phisionet_seizures_new.h5'
    sampling_rate = 1000
    path_to_load = '~/Desktop/seizure_datasets_new.h5'
    name_list = [str(time_before_seizure*60) + '_' + str(time_after_seizure*60)]
    group_list_raw = ['raw']
    group_list_baseline_removal = ['medianFIR']
    group_list_noise_removal = ['FIR_lowpass_40hz']
    group_list_esksmooth = ['esksmooth']

    group_list = [str(
    time_before_seizure*60) + '_' + str(time_after_seizure*60) + '/raw']
    group_name_list = list_group_signals(path_to_load, group_list[0])['signals']
    # compress(path_to_load, group_name_list)

    load_extract_feature(path_to_load, group_list, 'baseline_removal', compute_flag = False)
    # print list_group_signals(path_to_load, group_list[0])
    stop
    # Raw signal 
    # signal_structure_raw = load_signal(path_to_load,zip(group_list_raw, name_list))
    # one_signal_structure_raw = get_one_signal_structure(signal_structure_raw, zip(group_list_raw, name_list)[0])
    # records_raw = get_multiple_records(one_signal_structure_raw)

    # stop # ========================================================

    # # Baseline removal
    # records_baseline_removal, mdata_baseline_removal = load_baseline_removal(path_to_load, group_list_raw, name_list, sampling_rate, compute_flag = False)

    # # stop # =========================================================

    # # Noise removal 
    # records_noise_removal, mdata_noise_removal = load_noise_removal(path_to_load, group_list_baseline_removal, name_list, sampling_rate, compute_flag = False)

    # # stop # =========================================================

    # # Rpeak detection
    # rpeaks_noise_removal = load_rpeaks(path_to_load, group_list_noise_removal, name_list, sampling_rate, compute_flag = False)

    # # Noise removal kalman
    # records_kalman, mdata_kalman = load_kalman(path_to_load, group_list_baseline_removal, name_list, sampling_rate, compute_flag = False)

    # # stop # =========================================================
    # # Allocate time array
    # Fs = sampling_rate
    # N = len(records_noise_removal[0,:])
    # T = (N - 1) / Fs
    # t = np.linspace(0, T, N, endpoint=False)


    # # Visual inspection of rpeak detection
    # start = time_before_seizure*60 - 10
    # end = time_before_seizure*60 + 10
    # seizure_nr = 3

    # #rpeaks
    # rpeaks = rpeaks_noise_removal[seizure_nr]
    # rpeaks_resample = resample_rpeaks(np.diff(rpeaks), rpeaks, t)

    # #signal
    # signal_baseline_removal = records_baseline_removal[seizure_nr,:]
    # signal_noise_removal = records_noise_removal[seizure_nr,:]
    # signal_kalman = records_kalman[seizure_nr,:]


    # #plot
    # plt.subplot(3,1,1)
    # plt.xlim([start*250, end*250])
    # plt.plot(signal_kalman)
    # plt.subplot(3,1,2)
    # plt.plot(t, signal_noise_removal)
    # plt.xlim([start, end])
    # plt.subplot(3,1,3)
    # plt.plot(t, signal_baseline_removal)
    # plt.xlim([start, end])
    # plt.show()
    # # # stop

    # # visual_inspection(signal_noise_removal, rpeaks, rpeaks_resample, t, time_before_seizure,
    # #             start, end, sampling_rate)

    # # Visual inspection of rpeak detection --Kalman
    start = 1300
    end = 1400
    seizure_nr = 1

    # Fs = 250
    # N = len(records_kalman[0,:])
    # T = (N - 1) / Fs
    # t_down = np.linspace(0, T, N, endpoint=False)
    # factor = 4

    # signal_kalman_inter = interpolate_signal(signal_kalman, factor)

    # print np.shape(signal_kalman)
    # print np.shape(t)


    # #rpeaks
    # rpeaks = map(functools.partial(detect_rpeaks,
    #             sampling_rate=sampling_rate), [signal_kalman_inter])[0]
    # print rpeaks

    # rpeaks_resample = resample_rpeaks(np.diff(rpeaks), rpeaks, t)


    # visual_inspection(signal_kalman_inter, rpeaks, rpeaks_resample, t, time_before_seizure,
    #             start, end, 250)

    # beats = compute_beats(signal_kalman_inter, rpeaks)
    # tmp = time.time()
    # values = sameni_evolution(beats)
    # s = time.time() - tmp

    # values = np.asarray(values)
    # print s, 
    # print ' seconds'


    name_list = ['sameni_parameters_nr' + str(seizure_nr)]
    mdata_list = ['']
    signal_structure_baseline_removal = load_signal(path_to_load, zip(group_list_esksmooth, name_list))

    one_signal_structure = get_one_signal_structure(signal_structure_baseline_removal, zip(group_list_esksmooth, name_list)[0])
    records = get_multiple_records(one_signal_structure)
    mdata = get_mdata_dict(one_signal_structure)

    print records
    # stop
    color = 'g'

    # plt.plot(records.T[,:])
    # plt.show()

    # plot_single_model(time, records.T, color, start, end)
    # plt.show()




main()

# g1