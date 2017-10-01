from pre_epi_seizures.storage_utils.storage_utils_hdf5 import \
    load_signal, save_signal, delete_signal, list_group_signals

from pre_epi_seizures.storage_utils.data_handlers import *

from pre_epi_seizures.logging_utils.formatter_logging import logger as _logger

from filtering import baseline_removal, noise_removal,\
    create_filtered_dataset, eks_smoothing

from segmentation import *

from hrv import *

from rqa import *

from pca import *

from resampling import resample_rpeaks, interpolate_signal,\
    interpolation, decimation

from morphology import *
# from Filtering.gaussian_fit import get_phase, mean_extraction,\
#     beat_fitter, ecg_model

from Filtering.filter_signal import filter_signal

from Filtering.eksmoothing import *

from biosppy.signals import ecg
from biosppy.clustering import centroid_templates, kmeans

from memory_profiler import profile


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


def extract_feature(feature, arg_list):
    return locals[feature](arg_list)


def get_names(group_name_list):
    print group_name_list
    return [group_name[1] for group_name in group_name_list]


def input_default_params(input_params_dict, **default_params_dict):
    final_win_params = dict()

    for k in default_params_dict.keys():
        try:
            final_win_params[k] = input_params_dict[k]
        except Exception as e:
            final_win_params[k] = default_params_dict[k]

    return final_win_params

def get_final_params(win_input_params,
                   feature_group_to_process, begin_str, end_str):
    final_win_params = dict()
    win_params_from_str = get_params_from_str(
                            feature_group_to_process, 
                            begin_str,
                            end_str)

    for k in win_params_from_str:
        try:
            final_win_params[k] = win_input_params[k]
        except Exception as e:
            final_win_params[k] = win_params_from_str[k]

    return final_win_params


def get_params_from_str(feature_group_to_process, begin_str, end_str):
    params_from_str = dict()
    # print feature_group_to_process
    window_str_info = feature_group_to_process[
                        feature_group_to_process.index(begin_str)+1:
                        feature_group_to_process.index(end_str)+6]  
    a = window_str_info.split('_')
    a = a[1:-1]
    for k in a:
        try:
            params_from_str[k.split(':')[0]] = float(k.split(':')[1])
        except Exception as e:
            print e
            params_from_str[k.split(':')[0]] = k.split(':')[1]

    win_params_from_str = params_from_str
    return win_params_from_str


def get_str_from_params(params, begin_str, end_str):
    str_params = begin_str
    for k in params.keys():
        str_params += '_' + k + ':' + str(params[k])
    str_params += end_str
    return str_params
# def get_str_from_params(params, begin_str, end_str):
#     for 


# def windows_str(params):
#     str_params += '_[window:' + str(params['window'])
#     str_params += '-begin:' + str(params['begin'])
#     str_params += '-endwindow]'

#     return str_params

def load_feature(path_to_load, feature_to_load, files='just_new_data', sampling_rate=1000, **feature_groups_required):

    feature_group_to_process = feature_groups_required['feature_group_to_process']

    feature_group_extracted = [feature_group_to_process + '/' + feature_to_load]

    feature_groups_to_process = {k:feature_groups_required[k] for k in feature_groups_required.keys() if 'process' in k}

    # feature_group_aux = {k:feature_groups_required[k] for k in feature_groups_required.keys() if 'group' in k and 'process' not in k}

    # auxiliary_inputs = {k:feature_groups_required[k] for k in feature_groups_required.keys() if 'group' not in k and 'process' not in k}

    win_inputs = {k[4:]:feature_groups_required[k] for k in feature_groups_required.keys() if 'group' not in k and 'window' in k}
    param_inputs = {k[6:]:feature_groups_required[k] for k in feature_groups_required.keys() if 'group' not in k and 'param' in k}

    # print window_inputs
    # print param_inputs

    # # stop
    win_final = get_final_params(win_inputs,
                             feature_group_to_process,
                             '$beginwin', 
                             'endwin$')

    param_final = get_final_params(param_inputs,
                             feature_group_to_process, 
                             '$beginparam',
                             'endparam$')


    print feature_groups_to_process
    if files=='all_new':
        names_to_save = get_names(list_group_signals(path_to_load, feature_group_to_process)['signals'])
        for k in feature_groups_to_process.keys():
            feature_groups_to_process[k] = list_group_signals(path_to_load, feature_groups_to_process[k])['signals']
        names_to_save = get_names(list_group_signals(path_to_load, feature_group_to_process)['signals'])
        print feature_groups_to_process

    if files=='just_new_data':
        names_already_processed = get_names(list_group_signals(path_to_load, feature_group_extracted[0])['signals'])
        names_to_save = get_names(list_group_signals(path_to_load, feature_group_to_process)['signals'])
        names_to_save = [name for name in names_to_save if name not in names_already_processed]
        # feature_group_to_process = [(feature_group_to_process, name) for name in names_to_save]
        # # stop # Ver melhor
        # print feature_group_to_process
        # # stop

        # print feature_groups_required
        # stop
        # print feature_groups_required
        # # stop
        for k in feature_groups_to_process.keys():
            feature_groups_to_process[k] = [(feature_groups_to_process[k], name) for name in names_to_save]
        # print feature_groups_required
        
        # print feature_groups_to_process
        # stop

        #*****************IMPORTANT CHANGE***************************
        # names_to_save = get_names(list_group_signals(path_to_load, feature_group_to_process)['signals'])

    if files=='existent':
        feature_group_name_extracted = list_group_signals(path_to_load, feature_group_extracted[0])['signals']
        print feature_group_name_extracted
        try:
            signal_structure = load_signal(path_to_load, feature_group_name_extracted)
            extracted_features = [get_multiple_records(get_one_signal_structure(signal_structure, group_name)) for group_name in feature_group_name_extracted]
            mdata = [get_mdata_dict(get_one_signal_structure(signal_structure, group_name)) for group_name in feature_group_name_extracted]
            return extracted_features, mdata
        except Exception as e:
            _logger.debug(e)

    # Load all signals into memory --INTENSIVE BE CAREFUL
    # for k in feature_groups_to_process.keys():
    #     signal_structure = load_signal(path_to_load, feature_groups_to_process[k])
    #     feature_groups_to_process[k] = [get_multiple_records(get_one_signal_structure(signal_structure, group_name)) for group_name in feature_groups_to_process[k]]

    for i, name in enumerate(names_to_save):
        dict_to_process = {}
        for k in feature_groups_to_process.keys():
            group_name = feature_groups_to_process[k][i]
            # print group_name
            signal_structure = load_signal(path_to_load, [group_name])
            # print signal_structure
            # stop
            feature_signal = [get_multiple_records(get_one_signal_structure(signal_structure, group_name))]
            # print feature_signal
            # stop
            dict_to_process[k] = feature_signal

        # MEMORY INTENSIVE - BE CAREFUUL *********************
        # dict_to_process = {}
        # for k in feature_groups_required.keys():
        #     dict_to_process[k] = [feature_groups_required[k][i]]
        #*****************************************************

        # print dict_to_process
        return_structure = globals()[feature_to_load](dict_to_process,
                                                      sampling_rate,
                                                      win_final, param_final)
        extracted_features = return_structure[0]
        win_params = return_structure[1]
        params = return_structure[2]
        # print params
        # print win_params
        win_str = get_str_from_params(win_params, '_$beginwin',
                                      'endwin$_')
        param_str = get_str_from_params(params, '_$beginparam',
                                        'endparam$_')

        feature_group_extracted = [feature_group_extracted[0] + win_str + param_str]
        # print feature_group_extracted
        # stop
        # print extracted_features
        # print mdata_features
        delete_signal(path_to_load, [name], feature_group_extracted)
        save_signal(path_to_load, extracted_features,[''] * len(extracted_features), [name], feature_group_extracted)

    # # Memory Intensive
    # extracted_features, mdata_features = globals()[feature_to_load](feature_groups_required, sampling_rate)

    # delete_signal(path_to_load, names_to_save, feature_group_extracted)
    # save_signal(path_to_load, extracted_features, mdata_features, names_to_save, feature_group_extracted)




# @profile
def main():

    #signal
    sampling_rate = 1000
    time_before_seizure = 50 * 60
    time_after_seizure = 20 * 60
    path_to_load = '/Volumes/ASSD/pre_epi_seizures/h5_files/processing_datasets/seizure_datasets_new.h5'
    dataset_name = str(
    time_before_seizure) + '_' + str(time_after_seizure)

    raw_name = ['raw' + '_$beginwin_samplerate:1000' 
                   + '_win:0.001'
                   + '_init:0_finish:' + str(time_after_seizure + time_before_seizure)
                   + '_endwin$_'
                   + '_$beginparams_param:None'
                   + '_endparam$_'
                   ]
    raw_name = raw_name[0]


    baseline_removal_name = 'baseline_removal'
    raw_dataset_name = dataset_name + '/' + raw_name
    baseline_removal_dataset_name = raw_dataset_name + '/' + baseline_removal_name
    decimated_dataset_name = baseline_removal_dataset_name + '/' + 'decimation'
    eks_dataset_name = decimated_dataset_name + '/' + 'eks_smoothing'
    interpolated_dataset_name = eks_dataset_name + '/' + 'interpolation'
    # group_name_list = list_group_signals(path_to_load, group_list[0])['signals']
    # compress(path_to_load, group_name_list)

    # Load existing features --------------------------------------
    # raw = load_feature(path_to_load, raw_name, files='existent', feature_group_to_process=dataset_name)[0]


    # Extract the Features -----------------------------------------
    rpeaks = load_feature(path_to_load, 'rpeak_detection', files='all_new', feature_group_to_process=raw_dataset_name,
                          param_method='hamilton')

    # baseline_removal = load_feature(path_to_load, baseline_removal_name, files='all_new', feature_group_to_process=raw_dataset_name)

    # decimated = load_feature(path_to_load, 'decimation', files='existent', feature_group_to_process=baseline_removal_dataset_name)
    # rpeaks = load_feature(path_to_load, 'rpeak_detection', files='all_new', feature_group_to_process=baseline_removal_dataset_name)

    # hrv = load_feature(path_to_load, 'hrv_computation', files='all_new', feature_group_to_process=baseline_removal_dataset_name, rpeak_group_to_process=baseline_removal_dataset_name + '/' + 'rpeak_detection', window=10)

    # hrv_time_domain = load_feature(path_to_load,
                                   # 'hrv_time_domain_features',
                                   # files='existent',
                                   # feature_group_to_process=baseline_removal_dataset_name + '/' + 'hrv_computation_w:10',
                                   # window=10)


    # QRS_fixed = load_feature(path_to_load, 'QRS_fixed_segmentation', files='all_new', feature_group_to_process=baseline_removal_dataset_name, rpeak_group_to_process=baseline_removal_dataset_name + '/' + 'rpeak_detection')

    # eks = load_feature(path_to_load, 'eks_smoothing', files='existent', feature_group_to_process=baseline_removal_dataset_name + '/' + 'decimation', rpeak_group_to_process=baseline_removal_dataset_name + '/' + 'decimation' + '/' + 'rpeak_detection')[0]
    # # stop
    # # time_array_to_interpolate = np.linspace(0, 40*60 - 1.0/500, 40*60*500)
    # # print time_array_to_interpolate
    # interpolated = load_feature(path_to_load, 'interpolation', sampling_rate=500, files='existent', feature_group_to_process=eks_dataset_name)[0]
    # rpeaks = load_feature(path_to_load, 'rpeak_detection', files='existent', feature_group_to_process=interpolated_dataset_name)[0]
    # hrv = load_feature(path_to_load, 'hrv_computation',
    #                    files='all_new',
    #                    feature_group_to_process=interpolated_dataset_name,
    #                    rpeak_group_to_process=interpolated_dataset_name + '/' + 'rpeak_detection',
                       
                       
    # # beat = load_feature(path_to_load, 'beat_phase_segmentation', files='existent', feature_group_to_process=interpolated_dataset_name, rpeak_group_to_process=interpolated_dataset_name + '/' + 'rpeak_detection')[0]
    # pca = load_feature(path_to_load, 'pca_beat_amp_computation',
    #                    files='all_new',
    #                    feature_group_to_process=baseline_removal_dataset_name + '/' + 'QRS_fixed_segmentation',
    #                    window=baseline_removal_dataset_name + '/rpeaks',
    #                    begin=5)[0]

    # # sameni = load_feature(path_to_load, 'sameni_evolution', files='all_new', feature_group_to_process=interpolated_dataset_name + '/' + 'beat_phase_segmentation')[0]
    # # rqa = load_feature(path_to_load, 'rqa_computation', files='all_new', feature_group_to_process=interpolated_dataset_name + '/' + 'QRS_fixed_segmentation')[0]
    # # stop
    # # print rqa
    # # stop

    # # ploting
    # # print sameni
    # # stop
    # start = 10*60
    # end = start + 10 
    # sz_nr = 0
    # signal = raw
    stop
    print signal
    # stop
    # signal_t = hrv
    # n = np.linspace(0, (len(signal[sz_nr])-1)/1000, len(signal[sz_nr]))
    # n_t = np.linspace(0, (len(signal_t[sz_nr])-1)/1000, len(signal_t[sz_nr]))

    plt.figure()
    # plt.subplot(1,2,1)
    # plt.title('interpolated ECG')
    plt.plot(signal[sz_nr])
    # plt.plot(n[rpeaks[sz_nr]], signal[sz_nr][rpeaks[sz_nr]], 'o', color='g')
    plt.xlim([start*sampling_rate, end*sampling_rate])
    # plt.xlabel('time[s]')
    # plt.subplot(1,2,2)
    # plt.title('Detrended and Denoised ECG')
    # plt.plot(signal_t[sz_nr])
    # # plt.xlim([start, end])
    # plt.xlabel('time[s]')
    plt.savefig(raw_name + '.png')
    stop
    # stop
    # #phase 
    # signal = signal[sz_nr]

    # features = ecg.ecg(signal=signal, sampling_rate=1000.0, show=True)

    # # print len(idx_up

    # beats = zip(idx_up[0:-1], idx_up[1:])

    # data = [signal[i:f] for i,f in beats]
    # print len(data)
    # print len(rpeaks)


    # sample = len(data) - 1
    # plt.plot(data[sample]*0.05)
    # plt.plot(phase[beats[sample][0]:beats[sample][1]])
    # # plt.xlim([start*1000, end*1000])
    # plt.show()

    # fig_phase = plt.figure()t
    # phase = get_phase(interpolated[sz_nr], rpeaks[sz_nr])
    # print phase[0]


    # Sameni parameters Evaluation 
    # bins = len(x)
    #     rloc = int(bins/2) # r is assumed to be at the center
    #     thetai = np.zeros(5) # phase loc
    #     thetai[0] = phase[int(.2*bins)+np.argmax(x[int(.2*bins):int(.45*bins)])]
    #     idx = int(.44*bins) + np.argmin(x[int(.44*bins):int(.5*bins)])
    #     thetai[1] = phase[idx]
    #     thetai[2] = phase[rloc]
    #     thetai[3] = phase[2*rloc - idx]
    #     thetai[4] = phase[int(5/8*bins)+np.argmax(x[int(5/8*bins):int(7/8*bins)])]
    #     bi = np.array([.1, .05, .05, .05, .2]) # width
    #     ai = np.zeros(5) # amplitude
    #     ai[0] = np.abs(np.max(x[int(.2*bins):int(.45*bins)]))
    #     ai[1] = -np.abs(np.min(x))
    #     ai[2] = np.abs(np.max(x))
    #     ai[3] = -np.abs(np.min(x))
    #     ai[4] = np.abs(np.max(x[int(5/8*bins):int(7/8*bins)]))
    #     values0 = np.hstack((ai, bi, thetai))
    # parameters = sameni[sz_nr]
    # plt.figure()
    # plt.title('T-wave Amplitude/Width/Phase')
    # plt.subplot(3,1,1)
    # plt.plot(parameters[:, 4])
    # plt.subplot(3,1,2)
    # plt.plot(parameters[:, 9])
    # plt.subplot(3,1,3)
    # plt.plot(parameters[:, 14])
    # plt.show()
    # plt.title('R-peaks Amplitude/Width/Phase')
    # plt.subplot(3,1,1)
    # plt.plot(parameters[:, 2])
    # plt.subplot(3,1,2)
    # plt.plot(parameters[:, 7])
    # plt.subplot(3,1,3)
    # plt.plot(parameters[:, 12])
    # plt.show()
    # plt.title('P-wave Amplitude/Width/Phase')
    # plt.subplot(3,1,1)
    # plt.plot(parameters[:, 0])
    # plt.subplot(3,1,2)
    # plt.plot(parameters[:, 5])
    # plt.subplot(3,1,3)
    # plt.plot(parameters[:, 10])
    # plt.show()
    # plt.title('Q-wave Amplitude/Width/Phase')
    # plt.subplot(3,1,1)
    # plt.plot(parameters[:, 1])
    # plt.subplot(3,1,2)
    # plt.plot(parameters[:, 6])
    # plt.subplot(3,1,3)
    # plt.plot(parameters[:, 11])
    # plt.show()





    return


# def settings (a)

if __name__ == '__main__':
    main()

# g1