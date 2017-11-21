from create_datasets_nk import * 

from datetime import timedelta

import os

def parse_string_from_name(filename):
    name = filename[1]

    # get name, lead and date from record
    name_len = len(name)
    name_lead_date = name.split('_')
    return name_lead_date


def get_file_datetime(parsed_string):
    name = parsed_string[2]

    # convert string date to datetime
    date_time_file = datetime.strptime(
                                 name,
                                '%Y-%m-%d %H:%M:%S.%f')
    return date_time_file





def fetch_group_raw_to_map():
    group_list = ['/raw'
                   + '_$beginwin_samplerate:1000' 
                   + '_win:0.001'
                   + '_init:0_finish:' + str(time_after_seizure + time_before_seizure)
                   + '_endwin$_'
                   + '_$beginparams_param:None'
                   + '_endparam$_'
                   ]

    return group_list[0]


def get_baseline_filenames(list_free_filenames_disk, list_seizure_filenames_disk):
    # parse filenames
    list_free_filenames = map(parse_string_from_name,
                            list_free_filenames_disk)
    list_seizure_filenames = map(parse_string_from_name,
                                 list_seizure_filenames_disk)
    # convert to datetime
    list_free_filenames_datetime = map(get_file_datetime,
                                         list_free_filenames)
    list_seizure_filenames_datetime = map(get_file_datetime,
                                         list_seizure_filenames)

    # get filenames from files at least 2 hours before any seizure --Assumption baseline
    datetime_1st_seizure = list_seizure_filenames_datetime[0]    #1st seizure date time
    init_files = [list_free_filenames_disk[i]
                  for i, file in enumerate(list_free_filenames_datetime)
                  if datetime_1st_seizure - file > timedelta(hours=4)]

    return init_files


def save_records_raw(path_to_save, baseline_filenames, baseline_data, patient_number):
    # allocate raw group of features
    group_list = ['/raw'
                   + '_$beginwin_samplerate:1000' 
                   + '_win:0.001'
                   + '_init:0_finish:' + str(time_after_seizure + time_before_seizure)
                   + '_endwin$_'
                   + '_$beginparams_param:None'
                   + '_endparam$_'
                   ]

    name_list = [str(patient_number) + '_' + filename[1] for filename in baseline_filenames]
    baseline_data = [record[1]['signal'][0:1000*60*50].T for record in baseline_data.items()]

    mdata_list = [''] * len(baseline_data)
    save_signal(path_to_save, baseline_data, mdata_list, name_list, group_list)


def _create_free_datasets(path_to_save, path_to_load, patient_number):

    # import data from selected patient
    from patients_data_new import patients
    patients = patients[str(patient_number)]

    # get all filenames from disk
    list_all_filenames_disk = list_all_files_patient(path_to_load,
                                              patient_number)['signals']

    print list_all_filenames_disk 
    # get seizure filenames from disk
    list_seizure_filenames_disk = list_seizures_files_patient(path_to_load,
                                             patients, patient_number)
    # stop
    list_seizure_filenames_disk = [seizure[0]
                              for seizure in list_seizure_filenames_disk]

    # get free from seizures filenames from disk
    list_free_filenames_disk = [filename
                           for filename in list_all_filenames_disk
                           if filename not in list_seizure_filenames_disk]

    baseline_filenames = get_baseline_filenames(list_free_filenames_disk,
                                             list_seizure_filenames_disk)
    # return baseline_filenames


    baseline_data = load_signal(path_to_load, baseline_filenames)

    print baseline_data


    save_records_raw(path_to_save, baseline_filenames, baseline_data, patient_number)


def create_free_datasets(path_to_map, path_to_save, path_to_load, *patients):
    [_create_free_datasets(path_to_save, path_to_load, patient)
            for patient in patients]
    feature_group = fetch_group_raw_to_map()
    write_feature_to_map(path_to_map, feature_group)












path_to_load = '/Volumes/ASSD/pre_epi_seizures/h5_files/raw_fulldata/HSM_data.h5'
path_to_save = '/Volumes/ASSD/pre_epi_seizures/h5_files/processing_datasets/baseline_datasets_new.h5'
path_to_map = '/Volumes/ASSD/pre_epi_seizures/h5_files/processing_datasets/baseline_datasets_new_map.txt'
# patient_number = 1

time_before_seizure = 50 * 60
time_after_seizure = 20 * 60

create_free_datasets(path_to_map, path_to_save, path_to_load, 3, 4)


stop

dataset = create_seizure_dataset(path_to_load, path_to_save,
                                 time_before_seizure,
                                 time_after_seizure,  3, 4)

_logger.debug('the dataset is the following: %s', dataset)
_logger.debug(np.shape(dataset))


