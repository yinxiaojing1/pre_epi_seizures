def get_multiple_records(one_signal_structure):
    ''' retrieves the record from the signal structre'''

    # The structure is biosppy return_tuple
    return one_signal_structure['signal']


def get_list_group_signals(group, list_signals_structure):
    return [group_name[1] for group_name in list_signals_structure['signals']]


def get_record(one_signal_structure): 
    return get_multiple_records(one_signal_structure)[:,0]


def get_sampling_frequency(one_signal_structure):
    return one_signal_structure['mdata']['sample_rate']


def get_seizure_times_seconds(one_signal_structure):
    return one_signal_structure['mdata']['crysis_time_seconds']


def get_one_signal_structure(signals_structure, group_name):
    return signals_structure[group_name]


def get_multiple_records_group_name(signals_structure, group_name):
    one_signal_structure = get_one_signal_structure(signals_structure, group_name)
    records = get_multiple_records(one_signal_structure)
    return records


def get_mdata_dict(one_signal_structure):
    # print one_signal_structure
    return one_signal_structure['mdata']


def get_all_seizure_times_seconds(signals_structure):
    return {group_name:
             get_seizure_times_seconds(one_signal_structure)
             for group_name, one_signal_structure in signal_structure.iteritems()}


def get_filename_from_signals(signals_structure):
    return [group_name[1] for group_name in signals_structure]


def fetch_group_seizures(patient_number):
    return '/PATIENT' + str(patient_number) + '/crysis'


def fetch_group_free(patient_number):
    return '/PATIENT' + str(patient_number) + '/free'


def parse_sample_seizure(seizure_time_seconds):
    return 1000 * seizure_time_seconds
