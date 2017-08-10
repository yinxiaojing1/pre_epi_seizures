import storage_utils_phisionet as st_ph
from seizure_onsets_variables import seizure_onsets_seconds
from storage_utils_hdf5 import save_signal


import numpy as np

def create_datasets_phisionet(time_before_seizure,
                              time_after_seizure,
                              seizure_onsets_seconds,
                              *args):

    records = [st_ph.load_header_signals_phisionet(arg) for arg in args]
    # print records[0][1]
    # stop
    mdata = {'seizure_nbs':args}

    dataset = [create_record(record[0]['fs'], record[1], time_before_seizure,
                  time_after_seizure, seizure_onset_seconds).flatten()
               for record, seizure_onset_seconds in zip(records, seizure_onsets_seconds)]

    dataset = [data for data in dataset if len(data)!=0]
    print dataset
    print len(dataset[1])
    return np.asarray(dataset), mdata


def create_record(sampling_frequency, seizure_record, time_before_seizure,
                  time_after_seizure, seizure_onset):
    try:
        seizure_record = np.asarray(seizure_record[:,0])
        samples_before_seizure = int(sampling_frequency * time_before_seizure)
        samples_after_seizure = int(sampling_frequency * time_after_seizure)
        seizure_onset_sample = int(sampling_frequency * seizure_onset)
        print seizure_record[seizure_onset_sample
                              + samples_after_seizure]
        print len(seizure_record[seizure_onset_sample
                              - samples_before_seizure:
                              seizure_onset_sample
                              + samples_after_seizure])

        return seizure_record[seizure_onset_sample
                              - samples_before_seizure:
                              seizure_onset_sample
                              + samples_after_seizure]
    except Exception as e:
        print e
        return np.array([])

time_before_seizure = 10
time_after_seizure = 5

record, mdata = create_datasets_phisionet(time_before_seizure*60,time_after_seizure*60,
                                seizure_onsets_seconds, 1, 2, 3, 4, 5, 6, 7)

path_to_save = '~/Desktop/phisionet_seizures.h5'
name_list = [str(time_before_seizure*60) + '_' + str(time_after_seizure*60)]
group_list_raw = ['raw']
mdata_list = [mdata]
signal_list = [record]

save_signal(path_to_save, signal_list, mdata_list, name_list, group_list_raw)

print record
print mdata


