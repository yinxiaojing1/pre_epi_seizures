from pre_epi_seizures.storage_utils.storage_utils_hdf5 import \
    load_signal, save_signal

from pre_epi_seizures.storage_utils.data_handlers import *

from pre_epi_seizures.logging_utils.formatter_logging import logger as _logger

from pre_epi_seizures.Preprocessing.pre_processing import \
    load_baseline_removal, load_noise_removal, load_rpeaks, load_kalman

from pre_epi_seizures.Preprocessing.visual_inspection import \
    visual_inspection


from heart_rate_variability import inst_heart_rate


from morphology import compute_baseline_model

from segmentation import compute_fixed_beats

from pca import compute_PC, trace_evol_PC

from resampling import resample_rpeaks, interpolate_signal, interpolate

# # from filtering import baseline_removal, create_filtered_dataset

# # from segmentation import create_rpeak_dataset, create_heart_beat_dataset,\
# #     compute_beats

# from Filtering.gaussian_fit import get_phase, mean_extraction,\
#     beat_fitter, ecg_model

# from Filtering.filter_signal import filter_signal

from biosppy.signals import ecg

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import logging
import sys
import copy
import memory_profiler

def main():

    #signal
    time_before_seizure = 30
    time_after_seizure = 10
    # path_to_load = '~/Desktop/phisionet_seizures.h5'
    sampling_rate = 1000
    path_to_load = '~/Desktop/seizure_datasets.h5'
    name_list = [str(time_before_seizure*60) + '_' + str(time_after_seizure*60)]
    group_list_raw = ['raw']
    group_list_baseline_removal = ['medianFIR']
    group_list_noise_removal = ['FIR_lowpass_40hz']
    group_list_esksmooth = ['esksmooth']

    # Raw signal 
    signal_structure_raw = load_signal(path_to_load,zip(group_list_raw, name_list))
    # one_signal_structure_raw = get_one_signal_structure(signal_structure, zip(group_list, name_list)[0])
    # records_raw = get_multiple_records(one_signal_structure_raw)

    # stop # ========================================================

    # Baseline removal
    records_baseline_removal, mdata_baseline_removal = load_baseline_removal(path_to_load, group_list_raw, name_list, sampling_rate, compute_flag = False)

    # stop # =========================================================

    # Noise removal 
    records_noise_removal, mdata_noise_removal = load_noise_removal(path_to_load, group_list_baseline_removal, name_list, sampling_rate, compute_flag = False)


    # Noise removal kalman

    records_kalman, mdata_kalman = load_kalman(path_to_load, group_list_baseline_removal, name_list, sampling_rate, compute_flag = False)
 
    # stop # =========================================================

    # Rpeak detection
    # rpeaks_noise_removal = load_rpeaks(path_to_load, group_list_noise_removal, name_list, sampling_rate, compute_flag = False)


    # stop # =========================================================
    # Allocate time array
    Fs = 250
    N = len(records_kalman[0,:])
    T = (N - 1) / Fs
    t = np.linspace(0, T, N, endpoint=False)
    factor = 2

    # Visual inspection of rpeak detection
    start = 280
    end = 300
    seizure_nr = 1

    if seizure_nr < 3:
        Fs = 1000
        N = len(records_baseline_removal[0,:])
        T = (N - 1) / Fs
        t_new = np.linspace(0, T, N, endpoint=False)
        signal_inter = records_baseline_removal[seizure_nr]
        rpeaks_RAM = ecg.hamilton_segmenter(signal=signal_inter,
                                    sampling_rate=sampling_rate)['rpeaks']
    else:
        signal = records_kalman[seizure_nr]
        signal_inter, t_new = interpolate(t, signal, 1000)
        rpeaks_RAM = ecg.hamilton_segmenter(signal=signal_inter,
                                    sampling_rate=sampling_rate)['rpeaks']

    # stop
    y = np.diff(rpeaks_RAM)

    # visual_inspection(signal_inter, rpeaks_RAM, y, t, time_before_seizure,
    #             start, end, sampling_rate)

    t_rpeaks = t_new[rpeaks_RAM[1:]]
    y_new, t_n = interpolate(t_rpeaks, y, 2)
    print (len(t)-1)/250
    print (len(t_n)-1)/2
    time_before_seizure = time_before_seizure*60
    print time_before_seizure
    #rpeaks
    # rpeaks = rpeaks_noise_removal[seizure_nr]
    # rpeaks_resample = resample_rpeaks(np.diff(rpeaks), rpeaks, t)
    # rpeaks = find_rpeaks(rpeaks, start * sampling_rate,
    #     end * sampling_rate)

    #signal
    # signal_baseline_removal = records_baseline_removal[seizure_nr,:]
    # signal_noise_removal = records_noise_removal[seizure_nr,:]

    #plot
    plt.subplot(2,1,1)
    plt.plot(t_new, signal_inter)
    plt.plot(t_new[rpeaks_RAM], signal_inter[rpeaks_RAM], 'o')
    # plt.axvline(x=time_before_seizure*60, color = 'g')
    plt.subplot(2,1,2)
    plt.plot(t_n, y_new)
    plt.axvline(x=time_before_seizure, color='g')
    plt.plot()
    # plt.xlim([start, end])
    # plt.subplot(2,1,2)
    # plt.plot(t, signal_baseline_removal)
    # plt.xlim([start, end])
    plt.show()
    # # stop

    beats = compute_fixed_beats(signal_inter, rpeaks_RAM)
    print np.shape(beats[0:5])
    pca = compute_PC(beats[0:5])

    pca = np.dot(pca, beats[0:5])
    print np.shape(pca)

    evol = trace_evol_PC(beats)
    print np.shape(evol) 
    evol = evol.T

    t_evol = t_new[rpeaks_RAM[6:-1]]
    print len(t_evol)


    ev = [interpolate(t_evol, eigen, 2) for eigen in evol]

    plt.subplot(6,1,1)
    plt.plot(ev[4][1], 1*1000*60/y_new[1:len(ev[4][1])+1])
    plt.axvline(x=time_before_seizure, color='g')
    plt.legend(['HRV', 'Seizure onset'])
    plt.ylabel('bpm')
    plt.subplot(6,1,2)
    plt.plot(ev[4][1], ev[4][0])
    plt.axvline(x=time_before_seizure, color='g')
    plt.legend(['Eigen-Value 1', 'Seizure onset'])
    plt.subplot(6,1,3)
    plt.plot(ev[3][1], ev[3][0])
    plt.axvline(x=time_before_seizure, color='g')
    plt.legend(['Eigen-Value 2', 'Seizure onset'])
    plt.subplot(6,1,4)
    plt.plot(ev[2][1], ev[2][0])
    plt.axvline(x=time_before_seizure, color='g')
    plt.legend(['Eigen-Value 3', 'Seizure onset'])
    plt.subplot(6,1,5)
    plt.plot(ev[1][1], ev[1][0])
    plt.axvline(x=time_before_seizure, color='g')
    plt.legend(['Eigen-Value 4', 'Seizure onset'])
    plt.subplot(6,1,6)
    plt.plot(ev[0][1], ev[0][0])
    plt.axvline(x=time_before_seizure, color='g')
    plt.legend(['Eigen-Value 5', 'Seizure onset'])
    plt.xlabel('t (s)')
    plt.subplots_adjust(0.09, 0.1, 0.94, 0.94, 0.26, 0.46)

    plt.show()
# #     # print(pca.explained_variance_ratio_)
#     # print(pca.components_) 

#     hist = np.histogram(pca[:,4].T, bins=100, range=None, normed=False, weights=None, density=None)
#     # stop
#     # model = compute_baseline_model(signal, rpeaks, start, end)

#     # print len(model)

#     plt.hist(pca[:,0], bins=1000)
#     plt.ylim(0,40)
#     plt.show()

#     stop
#     models = [compute_baseline_model(record, rpeak, start, end) for record,rpeak in zip(records, rpeaks)]

#     print np.shape(models)
main()