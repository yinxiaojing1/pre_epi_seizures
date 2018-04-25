from storage_utils.storage_utils_hdf5 import \
    load_signal, save_signal, delete_signal, list_group_signals

from storage_utils.data_handlers import *

from logging_utils.formatter_logging import logger as _logger

from Preprocessing.filtering import baseline_removal, noise_removal,\
    create_filtered_dataset, eks_smoothing

from Preprocessing.get_default_params import *

from Preprocessing.segmentation import *

from Preprocessing.hrv import *

from Preprocessing.rqa import *

from Preprocessing.pca import *

from Preprocessing.resampling import resample_rpeaks, interpolate_signal,\
    interpolation, decimation

from Preprocessing.morphology import *
# from Filtering.gaussian_fit import get_phase, mean_extraction,\
#     beat_fitter, ecg_model

from Preprocessing.Filtering.filter_signal import filter_signal

from Preprocessing.Filtering.eksmoothing import *

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
    # print 'before loading' 
    # print group_name_list

    # Memory loop (one signal at a time)
    for i, group_name in enumerate(group_name_list):
        signal_structure = load_signal(path_to_load, [group_name])
        one_signal_structure = get_one_signal_structure(signal_structure, group_name)
        record = get_multiple_records(one_signal_structure)


def extract_feature(feature, arg_list):
    return locals[feature](arg_list)


def get_names(group_name_list):
    #print group_name_list
    return [group_name[1] for group_name in group_name_list
            if 'window' not in group_name[1]]


def get_list_group_from_str(path_to_load, str):
    all_group = list_group_signals(path_to_load, '/b')
    print all_group
    stop


def input_default_params(input_params_dict, **default_params_dict):
    final_win_params = dict()

    for k in default_params_dict.keys():
        try:
            final_win_params[k] = input_params_dict[k]
        except Exception as e:
            final_win_params[k] = default_params_dict[k]

    return final_win_params


# Highly Ineficient
def get_feature_group_name_list(path_to_map, feature_name):
    with open(path_to_map, 'r') as inF:
        feature_group_name_list = [ feature_group_name[:feature_group_name.index('#')]
                                     + feature_group_name[feature_group_name.index('#') + 1:
                                        feature_group_name.index('!')]
                                    for feature_group_name in inF
                                    if '#' and feature_name in feature_group_name
                                   ]

    inF.close()
            # print feature_name
    return feature_group_name_list


def _get_feature_group_name_list(feature_group_name, feature_name):
    relevant = feature_group_name[:feature_group_name.index('#')] + feature_group_name[feature_group_name.index('#')+ 1:feature_group_name.index('!')]


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


def get_latest_feature_name(feature_group_name):
    feature_name = feature_group_name.split('/')[-1]
    return feature_name


def get_win_str_snippet(feature_group_name, begin_str, end_str):
    feature_group_to_process = get_latest_feature_name(feature_group_name)
    window_str_info = feature_group_to_process[
                        feature_group_to_process.index(begin_str)+1:
                        feature_group_to_process.index(end_str)+6] 
    a = window_str_info.split('__')
    # a = a[1:-1]
    return a


def get_params_from_str_last(feature_groups_to_process, begin_str, end_str):
    params_from_str = dict()

    a = get_win_str_snippet(feature_groups_to_process, begin_str, end_str)
    #print a
    # stop
    for k in a:
        try:
            params_from_str[k.split(':')[0]] = float(k.split(':')[1])
        except Exception as e:
            print e
            params_from_str[k.split(':')[0]] = k.split(':')[1]

    win_params_from_str = params_from_str

    return win_params_from_str


def get_params_from_str(feature_group_to_process, begin_str, end_str):
    params_from_str = dict()
    # print feature_group_to_process
    # stop

    window_str_info = feature_group_to_process[
                        feature_group_to_process.index(begin_str)+1:
                        feature_group_to_process.index(end_str)+6]  


    a = window_str_info.split('_')
    a = a[1:-1]
    # print 'hello'
    # print a 
    # stop
    for k in a:
        try:
            params_from_str[k.split(':')[0]] = float(k.split(':')[1])
        except Exception as e:
            #print e
            params_from_str[k.split(':')[0]] = k.split(':')[1]

    win_params_from_str = params_from_str
    return win_params_from_str


def get_str_from_params(params, begin_str, end_str):
    str_params = begin_str
    for k in params.keys():
        str_params += '_' + k + ':' + str(params[k]) + '_'
    str_params += end_str
    return str_params



# def get_str_from_params(params, begin_str, end_str):
#     for 


# def windows_str(params):
#     str_params += '_[window:' + str(params['window'])
#     str_params += '-begin:' + str(params['begin'])
#     str_params += '-endwindow]'

#     return str_params

def load_feature(path_to_load, path_to_map, feature_to_load,
                 files='just_new', sampling_rate=1000,
                 **feature_groups_required):

  
    feature_group_to_process = feature_groups_required['feature_group_to_process']

    feature_group_extracted = [feature_group_to_process + '/' + feature_to_load]

    feature_groups_to_process = {k:feature_groups_required[k] for k in feature_groups_required.keys() if 'process' in k}
    
    print feature_group_to_process
    print feature_group_extracted
    

    # feature_group_aux = {k:feature_groups_required[k] for k in feature_groups_required.keys() if 'group' in k and 'process' not in k}

    # auxiliary_inputs = {k:feature_groups_required[k] for k in feature_groups_required.keys() if 'group' not in k and 'process' not in k}


    # Input and default parameters from feature to extract


    win_inputs = {k[4:]:feature_groups_required[k] for k in feature_groups_required.keys() if 'group' not in k and 'window' in k}
    param_inputs = {k[6:]:feature_groups_required[k] for k in feature_groups_required.keys() if 'group' not in k and 'param' in k}

    win_final, param_final = get_input_and_default_params(win_inputs, param_inputs, feature_to_load)

    
    win_str = get_str_from_params(win_final, '_$beginwin',
                                      'endwin$_')
    param_str = get_str_from_params(param_final, '_$beginparam',
                                        'endparam$_')


    feature_group_to_save = [feature_group_extracted[0] + win_str + param_str]

    feature_groups_saved_list = get_feature_group_name_list(path_to_map, feature_to_load+ '#')


    # Default parameters from feature to process ---------------------------------------------------------------
    win_param_to_process = get_params_from_str(
                                 feature_group_to_process,
                                '$beginwin', 
                                 'endwin$')

    param_to_process = get_params_from_str(
                                   feature_group_to_process, 
                                   '$beginparam',
                                   'endparam$')


    if feature_group_to_save[0] not in feature_groups_saved_list:
        print 'Saving to txt'
        group_name = feature_group_to_save[0]
        txtname = path_to_load[:-3] + '_map.txt'

        with open(path_to_map, 'ab') as inF:
            inF.write(group_name[:group_name.index(feature_to_load) + len(feature_to_load)]
                            + '#'
                            + group_name[group_name.index(feature_to_load) + len(feature_to_load):]
                            + '!' + "\n")  # python will convert \n to os.linesep

        inF.close()

    if files=='just_new':
        names_already_processed = get_names(list_group_signals(path_to_load, feature_group_to_save[0])['signals'])
        names_to_save = get_names(list_group_signals(path_to_load, feature_group_to_process)['signals'])
        names_to_save = [name for name in names_to_save if name not in names_already_processed]
        for k in feature_groups_to_process.keys():
            feature_groups_to_process[k] = [(feature_groups_to_process[k], name) for name in names_to_save]
        # stop

    # print feature_groups_to_process
    if files=='all_new':
        names_to_save = get_names(list_group_signals(path_to_load, feature_group_to_process)['signals'])
        for k in feature_groups_to_process.keys():
            feature_groups_to_process[k] = list_group_signals(path_to_load, feature_groups_to_process[k])['signals']
        names_to_save = get_names(list_group_signals(path_to_load, feature_group_to_process)['signals'])
        # print feature_groups_to_process
        #*****************IMPORTANT CHANGE***************************
        # names_to_save = get_names(list_group_signals(path_to_load, feature_group_to_process)['signals'])

    if files=='existent':
        feature_group_name_extracted = list_group_signals(path_to_load, feature_group_extracted[0])['signals']
        # print feature_group_name_extracted
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
            signal_structure = load_signal(path_to_load, [group_name])
            feature_signal = [get_multiple_records(get_one_signal_structure(signal_structure, group_name))]
            dict_to_process[k] = feature_signal

        # MEMORY INTENSIVE - BE CAREFUUL *********************
        # dict_to_process = {}
        # for k in feature_groups_required.keys():
        #     dict_to_process[k] = [feature_groups_required[k][i]]
        #*****************************************************

        # print dict_to_process

        return_object = globals()[feature_to_load](dict_to_process,
                                                   win_final, param_final,
                                                   win_param_to_process,
                                                   param_to_process)

        extracted_features = return_object[0]
        mdata_list = return_object[1]
        window_list = return_object[2]

        delete_signal(path_to_load, [name], feature_group_to_save)
        delete_signal(path_to_load, ['window_' + name], feature_group_to_save)
        
        save_signal(path_to_load, extracted_features,
                    mdata_list,
                    [name], feature_group_to_save)
        save_signal(path_to_load, window_list,
                    [''] * len(window_list),
                    ['window_' + name], feature_group_to_save)


    # # Memory Intensive
    # extracted_features, mdata_features = globals()[feature_to_load](feature_groups_required, sampling_rate)

    # delete_signal(path_to_load, names_to_save, feature_group_extracted)
    # save_signal(path_to_load, extracted_features, mdata_features, names_to_save, feature_group_extracted)



# @profile

def main(disk, dataset_files_path, **kwargs):
    path_to_load = disk + 'h5_files/processing_datasets/seizure_datasets_new.h5'
    path_to_map = disk + 'h5_files/processing_datasets/seizure_datasets_new_map.txt'
    sz = (path_to_load, path_to_map)

    path_to_load = disk + 'h5_files/processing_datasets/baseline_datasets_new.h5'
    path_to_map= disk + 'h5_files/processing_datasets/baseline_datasets_new_map.txt'
    baseline = (path_to_load, path_to_map)

    order = [sz, baseline]

    for path_to_load, path_to_map in order:
        print ''
        print '********PATH: ' + str(path_to_load) + '******************'
        _main(path_to_load, path_to_map, **kwargs)
    
def _main(disk,
          dataset_files_path, **kwargs):
    

    # Input Stage
    path_to_load = disk + dataset_files_path + '.h5'
    path_to_map = disk + dataset_files_path + '_map.txt'

    # General Feature Extraction pipeline
    # # 1. Raw -----------------------------------------------------------------------------------
    raw_groups = get_feature_group_name_list(path_to_map,
                                             'raw#')

    # 2. Baseline removal and denoising------------------------------------------------------
    files = 'just_new'
    feature_name = 'baseline_removal'
    for raw_group in raw_groups:
        param_filt_variation = ['MedianFIR']
        for param_filt in param_filt_variation:
            load_feature(path_to_load, path_to_map, feature_name,
                         files=files,
                         feature_group_to_process=raw_group,
                         param_filt = param_filt)
   
    
    print 'Lets see what happened to the baseline_signal'
    STOP

    # 3. Segmentation---------------------------------------------------------------------------
    
    files = 'just_new'
    feature_name = 'rpeak_detection'
    group_to_process = get_feature_group_name_list(path_to_map,
                                             'baseline_removal#')
    
 
    for group in group_to_process:
        param_method_variation = ['hamilton']
        for param_method in param_method_variation:
            load_feature(path_to_load, path_to_map, feature_name,
                         files=files,
                         feature_group_to_process=group,
                         param_method = param_method)
            
              


    # 3.1 HRV computation-----------------------------------------------------------------------
    
    files='all_new'
    rpeaks_groups_to_process = get_feature_group_name_list(path_to_map,
                                             'rpeak_detection#')
    feature_name = 'hrv_computation'
    groups_to_process = get_feature_group_name_list(path_to_map,
                                             'baseline_removal#')
    rpeaks_groups_to_process = [feature_group_name
                                for feature_group_name in rpeaks_groups_to_process
                                if 'baseline_removal' in feature_group_name]
    resampling_rate_list = [1000]
    resampling_method_list = ['cubic_splines']
    
    for baseline_signal, rpeaks in zip(groups_to_process, rpeaks_groups_to_process):
        
        for resampling_rate, resampling_method in zip(resampling_rate_list,
                                                      resampling_method_list):
            load_feature(path_to_load, path_to_map, feature_name,
                         files=files,
                         feature_group_to_process=rpeaks,
                         resampling_rate=resampling_rate,
                         param_resampling_method=resampling_method
                         )

    
    # 3.2 HRV time features computation-----------------------------------------------------------------------
    groups_to_process = get_feature_group_name_list(path_to_map,
                                              'hrv_computation#')
    rpeaks_groups_to_process = get_feature_group_name_list(path_to_map,
                                             'rpeak_detection#')
    feature_name = 'hrv_time_features'
    for group in zip(groups_to_process, rpeaks_groups_to_process):
        win_win_variation = [2 * 60, 3 * 60]
        for win_win in win_win_variation:
            load_feature(path_to_load, path_to_map,
                         feature_name,
                         files=files,
                         feature_group_to_process=group[0], 
                         rpeak_group_to_process=group[1])


    files = 'just_new'

    rpeaks_groups_to_process = get_feature_group_name_list(path_to_map,
                                             'rpeak_detection#')
    # stop
    feature_name = 'QRS_fixed_segmentation'
    groups_to_process = get_feature_group_name_list(path_to_map,
                                             'baseline_removal#')
    rpeaks_groups_to_process = [feature_group_name
                                for feature_group_name in rpeaks_groups_to_process
                                if 'baseline_removal' in feature_group_name]
    sampling_rate=1000
    for groups in zip(groups_to_process, rpeaks_groups_to_process):
        win_samplerate_variation = [sampling_rate]
        for win_samplerate in win_samplerate_variation:
            # stop
            load_feature(path_to_load, path_to_map, feature_name,
                         files=files,
                         feature_group_to_process=groups[1],
                         rpeak_group_to_process=groups[0],
                         win_samplerate=win_samplerate)
            
     
    
    files = 'just_new'
            
            
    # phase-invariant beats
    #rpeaks_groups_to_process = get_feature_group_name_list(path_to_map,
                #                             'rpeak_detection#')
    #feature_name = 'beat_phase_segmentation'
    #groups_to_process = get_feature_group_name_list(path_to_map,
           #                                  'baseline_removal#')
    #rpeaks_groups_to_process = [feature_group_name
    #                            for feature_group_name in rpeaks_groups_to_process
     #                           if 'baseline_removal' in feature_group_name]
    #sampling_rate=1000
    #for groups in zip(groups_to_process, rpeaks_groups_to_process):
      #  win_samplerate_variation = [sampling_rate]
     #   for win_samplerate in win_samplerate_variation:
            # stop
       #     load_feature(path_to_load, path_to_map, feature_name,
       #                  files=files,
       #                  feature_group_to_process=groups[1],
       #                  rpeak_group_to_process=groups[0],
        #                 win_samplerate=win_samplerate)

    #stop

    # # # STOP

    #rpeaks_groups_to_process = get_feature_group_name_list(path_to_map,
                                           #  'rpeak_detection#')
    #feature_name = 'beat_phase_segmentation'
    #files = 'all_new'
    #groups_to_process = get_feature_group_name_list(path_to_map,
                                          #   'baseline_removal#')
    #rpeaks_groups_to_process = [feature_group_name
     #                           for feature_group_name in rpeaks_groups_to_process
      #                          if 'baseline_removal' in feature_group_name]

    #for groups in zip(groups_to_process, rpeaks_groups_to_process):
     #   win_samplerate_variation = [sampling_rate]
      #  for win_samplerate in win_samplerate_variation:
       #     # stop
        #    load_feature(path_to_load, path_to_map, feature_name,
         #                files=files,
          #               rpeak_group_to_process=groups[1],
           #              feature_group_to_process=groups[0],
            #             win_samplerate=win_samplerate)


    # STOP


    groups_to_process = get_feature_group_name_list(path_to_map,
                                             'QRS_fixed_segmentation#')
    print groups_to_process
    feature_name = 'pca_beat_amp_computation'
    for groups in zip(groups_to_process, rpeaks_groups_to_process):
        nr_comp_variation = [5]
        for nr_comp in nr_comp_variation:
            # stop
            load_feature(path_to_load, path_to_map, feature_name,
                         files=files,
                         feature_group_to_process=groups[1],
                         rpeak_group_to_process=groups[0],
                         nr_comp = nr_comp_variation)

            
    print 'PCA DONE!'
    
    
    
    print 'Computing P_fixed_segmentation'        
            
    rpeaks_groups_to_process = get_feature_group_name_list(path_to_map,
                                             'rpeak_detection#')
    # stop
    feature_name = 'P_fixed_segmentation'
    files = 'just_new'
    groups_to_process = get_feature_group_name_list(path_to_map,
                                             'baseline_removal#')
    rpeaks_groups_to_process = [feature_group_name
                                for feature_group_name in rpeaks_groups_to_process
                                if 'baseline_removal' in feature_group_name]

    for groups in zip(groups_to_process, rpeaks_groups_to_process):
        win_samplerate_variation = [sampling_rate]
        for win_samplerate in win_samplerate_variation:
            # stop
            load_feature(path_to_load, path_to_map, feature_name,
                         files=files,
                         feature_group_to_process=groups[1],
                         rpeak_group_to_process=groups[0],
                         win_samplerate=win_samplerate) 
            
    print 'Computing PCA'       
    groups_to_process = get_feature_group_name_list(path_to_map,
                                             'P_fixed_segmentation#')
    
    print groups_to_process
  
    feature_name = 'pca_beat_amp_computation'
    for groups in zip(groups_to_process, rpeaks_groups_to_process):
        nr_comp_variation = [5]
        for nr_comp in nr_comp_variation:
            # stop
            load_feature(path_to_load, path_to_map, feature_name,
                         files=files,
                         feature_group_to_process=groups[1],
                         rpeak_group_to_process=groups[0],
                         nr_comp = nr_comp_variation)

            
    print 'PCA DONE!'
            
            
    
    return


    groups_to_process = get_feature_group_name_list(path_to_map,
                                             'QRS_fixed_segmentation#')
    files = 'just_new'
    print groups_to_process
    feature_name = 'rqa_computation'
    for groups in zip(groups_to_process, rpeaks_groups_to_process):
        nr_comp_variation = [5]
        for nr_comp in nr_comp_variation:
            # stop
            load_feature(path_to_load, path_to_map, feature_name,
                         files=files,
                         rpeak_group_to_process=groups[1],
                         feature_group_to_process=groups[0],
                         nr_comp = nr_comp_variation)
            
                
    print 'RQA DONE!'
    return

    # # 3.2.1 F beat extraction
    # files = 'all_new'
    # for baseline_removal_group in baseline_removal_groups:
    #     param_filt_variation = ['hamilton']
    #     for param_filt in param_filt_variation:
    #         load_feature(path_to_load, baseline_removal_name,
    #                      files=files,
    #                      feature_group_to_process=raw_dataset_name,
    #                      param_filt = 'hamilton')


    # # 3.2.2 phasemapped-window segmentation
    # files = 'all_new'
    # for baseline_removal_group in baseline_removal_groups:
    #     param_filt_variation = ['hamilton']
    #     for param_filt in param_filt_variation:
    #         load_feature(path_to_load, baseline_removal_name,
    #                      files=files,
    #                      feature_group_to_process=raw_dataset_name,
    #                      param_filt = 'hamilton')

    # # 3.2.1 pha



    



    return


# def settings (a)

if __name__ == '__main__':
    main()

# g1