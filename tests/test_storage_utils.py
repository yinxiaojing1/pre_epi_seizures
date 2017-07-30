from pre_epi_seizures.storage_utils.storage_utils_hdf5 import load_signal 
from pre_epi_seizures.storage_utils.data_handlers import *
from pre_epi_seizures.storage_utils.create_datasets import *

import numpy as np

def test_load_signal():
    path_to_load = '~/Desktop/HSM_data.h5'
    patient_number = 1

    # Load all seizures files
    group_name_list = list_seizures_files(path_to_load=path_to_load,
                                     patient_number=patient_number)
    X = load_signal(path =path_to_load,
                    group_name_list=group_name_list)

    # test handlers for a single seizure file
    group_name = group_name_list[0]
    one_signal_structure = X[group_name]
    Fs = get_sampling_frequency(one_signal_structure)
    seizure_times = get_seizure_times_seconds(one_signal_structure)
    records = get_multiple_records(one_signal_structure)
    record = get_record(one_signal_structure)

    assert Fs==1000
    assert seizure_times == [4051.593]
    assert np.shape(records) == (2*Fs*3600,1)
    assert np.shape(record) == (2*Fs*3600,)




    # test handlers for all seizure files

def test_create_dataset():
