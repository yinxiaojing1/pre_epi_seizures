from logging_utils.formatter_logging import\
    logger as _logger

from storage_utils.storage_utils_hdf5 import\
    load_signal, save_signal, list_group_signals

from storage_utils.data_handlers import\
    get_record, get_sampling_frequency, get_seizure_times_seconds,\
    get_one_signal_structure, parse_sample_seizure

import numpy as np

from datetime import datetime

import os

# from __future__ importS print_function


def get_feature_group_name_list(path_to_map, feature_name):
    if os.path.isfile(path_to_map):
      with open(path_to_map, 'r+b') as inF:
          feature_group_name_list = [ feature_group_name[:feature_group_name.index('#')]
                                       + feature_group_name[feature_group_name.index('#') + 1:
                                          feature_group_name.index('!')]
                                      for feature_group_name in inF
                                      if '#' and feature_name in feature_group_name
                                     ]

      inF.close()
              # print feature_name
      return feature_group_name_list
    else: return []


def write_feature_to_map(path_to_map, feature_group):
    if feature_group\
      not in get_feature_group_name_list(path_to_map, feature_group):
        group_name = feature_group
        f = open(path_to_map,'w')
        #f.write(group_name[:group_name.index('raw') + 3]
         #        + '#'
         #        + group_name[group_name.index('raw') + 3:]
         #        + '!' + '\n')  # python will convert \n to os.linesep
        f.close()


def return_dates(list_group):
    all_date_times = [datetime.strptime(
                                name[1][len(name[1])-26:len(name[1])-1],
                                '%Y-%m-%d %H:%M:%S.%f')
                                for name in list_group['signals']]
    all_date_times = all_date_times
    return sorted(all_date_times)


def find_index_seizure_file(date_time_seizure,
                                all_date_times):
    index = all_date_times.index([all_date_time
                              for all_date_time in all_date_times
                              if all_date_time < date_time_seizure][-1])

    return index


def find_indexes_seizure_files(date_time_seizures, all_date_times):
    return [find_index_seizure_file(data_time_seizure, all_date_times)
            for data_time_seizure in date_time_seizures]


def get_nr_leads(list_group_all_files):
    # retrieve datetimes from all files
    all_date_times = return_dates(list_group_all_files)

    # compute the number of leads
    nr_leads = len(list_group_all_files['signals'])/len(list(set(all_date_times)))

    return nr_leads


def list_seizures_files_patient(path_to_load, patient_dict, patient_number):
    # list of all patients
    list_group = list_group_signals(path=path_to_load,
                                    group='PATIENT'+ str(patient_number))

    patients = patient_dict





    # retrieve datetimes from all files
    all_date_times = return_dates(list_group)

    # compute the number of leads
    nr_leads = len(list_group['signals'])/len(list(set(all_date_times)))

    # retrieve the datetimes of seizures
    date_seizures = patients['dates_of_seizure']
    time_seizures = patients['ictal_on_time']
    date_time_seizures = [datetime.combine(d_s, t_s)
                          for d_s,t_s
                          in zip(date_seizures, time_seizures)]
    seizures = range(len(date_time_seizures))

    print date_time_seizures
    
    
    # find indexes
    indexes = find_indexes_seizure_files(date_time_seizures, all_date_times)
    

    # fetch seizure filenames
    list_files_seizures = [(list_group['signals'][index + i], sz)
                           for index, sz in zip(indexes, seizures)
                           for i in xrange(nr_leads)]

    return list_files_seizures


def list_free_files_patient(path_to_load, patient_number):
    list_group = list_group_signals(path=path_to_load,
                                    group=fetch_group_free(
                                        patient_number))
    return list_group['signals']


# def list_all_files_patient(path_to_load, patient_number):
#     list_seizures = list_seizures_files_patient(path_to_load, patient_number)
#     list_free = list_free_files_patient(path_to_load, patient_number)
#     list_all = list_seizures + list_free
#     list_sorted = sorted([(item[1], item[0]) for item in list_all])
#     return [(item[1], item[0]) for item in list_sorted]

def list_all_files_patient(path_to_load, patient_number):
    list_group = list_group_signals(path=path_to_load,
                                    group='PATIENT'+ str(patient_number))
    return list_group

def get_record_dataset_seizure(path_to_load,
                               samples_before_seizure, samples_after_seizure,
                               sample_seizure, group_name_all_list,
                               group_name_seizure, total_record, nr_leads):

    diff_before_seizure = int(sample_seizure - samples_before_seizure)
    diff_after_seizure = int(sample_seizure + samples_after_seizure)
    samples_record = 2 * 3600 * 1000

    group_name_all_list = group_name_all_list['signals']


    if diff_before_seizure < 0:
        group_name_previous_file = group_name_all_list[
            group_name_all_list.index(group_name_seizure) - 1 * nr_leads]
        previous_record = get_record(
            one_signal_structure=get_one_signal_structure(
                signals_structure=load_signal(path=path_to_load,
                                              group_name_list=[group_name_previous_file]),
            group_name=group_name_previous_file))

        previous_record_to_append = previous_record[diff_before_seizure:]
        total_record = np.concatenate((previous_record_to_append,
                                       total_record))
        samples_added = abs(diff_before_seizure)
        diff_before_seizure = int(diff_before_seizure + samples_added)
        diff_after_seizure = int(diff_after_seizure + samples_added)
        samples_record = int(samples_record + samples_added)

    if diff_after_seizure > samples_record:
        group_name_next_file = group_name_all_list[
            group_name_all_list.index(group_name_seizure) + 1]

        print group_name_next_file
        next_record = get_record(
            one_signal_structure=get_one_signal_structure(load_signal(
                path=path_to_load,
                group_name_list=[group_name_next_file]),
            group_name=group_name_next_file))

        next_record_to_append = next_record[
            :diff_after_seizure - samples_record]

        total_record = np.concatenate((total_record, next_record_to_append))

    # print len(total_record[diff_before_seizure:diff_after_seizure])
    # stop
    return total_record[diff_before_seizure:diff_after_seizure]


def get_record_dataset_seizure_file(path_to_load,
                                    path_to_save,
                                    path_to_map,
                                    time_before_seizure,
                                    time_after_seizure,
                                    group_name_all_list,
                                    group_name_file_seizure,
                                    patient_dict,
                                    patient_nr,
                                    nr_leads):
    ''' Creates a record from a seizure file, based on the criteria
        explicited by time_before_seizure and time_after_seizure,
        '''
    try:
        # Input requirements
        # Get seizure number and corresponding file name
        seizure = group_name_file_seizure[1]
        group_name_file_seizure = group_name_file_seizure[0]
        
        print '' 
        print '------------'
        print 'creating raw dataset'
        print seizure
        print group_name_file_seizure

        

        # load and allocate to memory the seizure file
        signals_structure = load_signal(path_to_load, [group_name_file_seizure])
        one_signal_structure = get_one_signal_structure(signals_structure,
                                                        group_name_file_seizure) 
        total_record = get_record(one_signal_structure)
        
        print '' 

        print 'Raw Data from disk'
        print total_record
        print '-------------'


        # get seizure datetime
        seizure_time = datetime.combine(
                        patient_dict['dates_of_seizure'][seizure],
                        patient_dict['ictal_on_time'][seizure]
                        )

        # generate output file name
        name = group_name_file_seizure[1]
        date_time_file = datetime.strptime(
                                     name[len(name) - 26:len(name) - 1],
                                    '%Y-%m-%d %H:%M:%S.%f')

        # create seizure record (Logic)
        Fs = 1000
        time_seconds_seizure = (seizure_time - date_time_file).total_seconds()
        sample_seizure = int(Fs*time_seconds_seizure)
        samples_before_seizure = int(time_before_seizure * Fs)
        samples_after_seizure = int(time_after_seizure * Fs)

        dataset_seizures_file = [get_record_dataset_seizure(
                    path_to_load=path_to_load,
                    samples_before_seizure=samples_before_seizure,
                    samples_after_seizure=samples_after_seizure,
                    sample_seizure=sample_seizure,
                    group_name_all_list=group_name_all_list,
                    group_name_seizure=group_name_file_seizure,
                    total_record=total_record,
                    nr_leads=nr_leads)]

        # Output peocedures
        # Prep record for saving
        dataset_seizures_file = [(group_name_file_seizure[1] +
                                  '_' + str(seizure), signal)
                                 for signal in dataset_seizures_file]
        save_dataset(path_to_save,
                     path_to_map,
                     time_before_seizure,
                     time_after_seizure,
                     patient_nr,
                     [dataset_seizures_file])
        return dataset_seizures_file
    except Exception as e:
        
        print 'FAILURE' 
        print e
    # *** ATENTION. take notice [dataset_seizure_file].
    # This list construct is required, since save_dataset is 
    # prepared to save list of records (dataset), not a single
    # record


    # print dataset_seizures_file
    

def save_dataset(path_to_save,
                 path_to_map,
                 time_before_seizure,
                 time_after_seizure,
                 patient_number,
                 dataset_list_files):

    group_list = ['/' + str(time_before_seizure) + '_' + str(time_after_seizure) + '/raw'
                   + '_$beginwin_samplerate:1000' 
                   + '_win:0.001'
                   + '_init:0_finish:' + str(time_after_seizure + time_before_seizure)
                   + '_endwin$_'
                   + '_$beginparams_param:None'
                   + '_endparam$_'
                   ]


    #write_feature_to_map(path_to_map, group_list[0])
    for dataset_list_file in dataset_list_files:
        print dataset_list_file
        name_list = [str(patient_number) + '_' + seizure_record[0] for seizure_record in dataset_list_file]
        signal_list = [np.array([seizure_record[1]]) for seizure_record in dataset_list_file]
        mdata_list=[{'fs':1000}]*len(dataset_list_file)
        # stop
        save_signal(path_to_save, signal_list, mdata_list, name_list, group_list)


def create_seizure_dataset_patient(path_to_load, path_to_save,
                                   path_to_map,
                                   time_before_seizure,
                                   time_after_seizure, patient_number):
    _logger.debug('searching all files')

    # import datetime of seizure from selected patient
    from storage_utils.patients_data_new import patients
    patient_dict = patients[str(patient_number)]

    # retrieve the names of all available files
    group_name_all_list = list_all_files_patient(path_to_load=path_to_load,
                                                 patient_number=patient_number)

    print patient_dict
    print group_name_all_list
    print 'many files'
    
 
    _logger.debug(group_name_all_list)

    # get number of leads
    nr_leads = get_nr_leads(group_name_all_list)

    # retrieve the names of all the seizure files
    group_name_seizure_list = list_seizures_files_patient(
        path_to_load=path_to_load,
        patient_dict=patient_dict,
        patient_number=patient_number)
    _logger.debug(group_name_seizure_list)
    

    # Get all the seizure records into memory
    seizure_list = [get_record_dataset_seizure_file(
                    path_to_load=path_to_load,
                    path_to_save=path_to_save,
                    path_to_map=path_to_map,
                    time_before_seizure=time_before_seizure,
                    time_after_seizure=time_after_seizure,
                    group_name_all_list=group_name_all_list,
                    group_name_file_seizure=group_name,
                    patient_dict=patient_dict,
                    patient_nr=patient_number,
                    nr_leads=nr_leads)
                    for group_name in group_name_seizure_list]


    # Save seizure records to disk
    #save_dataset(path_to_save, path_to_map, time_before_seizure, time_after_seizure, patient_number, dataset_list)
    #_logger.error('the dataset_list is the following: %s', dataset_list)
    #return dataset_list


def create_seizure_dataset(path_to_load, path_to_save,
                           path_to_map,
                           time_before_seizure,
                           time_after_seizure, patient_list):

    # Run logic for each patient
    [create_seizure_dataset_patient(path_to_load,
                                    path_to_save,
                                    path_to_map,
                                    time_before_seizure,
                                    time_after_seizure,
                                    patient)
     for patient in patient_list]


def create_datasets_nk(disk,
                       time_before_seizure,
                       time_after_seizure,
                       patient_list):

    # required input files
    path_to_load = disk + ('h5_files/raw_fulldata/HSM_data.h5')
    path_to_map = disk + ('h5_files/processing_datasets/' +
                          'seizure_datasets_new_map.txt')

    # Output files
    path_to_save = disk + ('h5_files/processing_datasets' +
                           '/seizure_datasets_new.h5')

    # Run logic
    create_seizure_dataset(path_to_load, path_to_save,
                           path_to_map,
                           time_before_seizure,
                           time_after_seizure,
                           patient_list)
