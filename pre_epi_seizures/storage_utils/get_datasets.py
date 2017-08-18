from pre_epi_seizures.logging_utils.formatter_logging\
    import logger

from storage_utils_hdf5 import list_group_signals

from data_handlers import *


def retrieve_seizure_times_file(list_signals_file):
    indexes = [string_signal.find('_') for string_signal in list_signals_file]
    print indexes
    times_seizure = [(string_signal[0:index], string_signal[index+1:-1])
        for index, string_signal in zip(indexes, list_signals_file)]
    times_seizure = zip(*times_seizure)
    times_before_seizure = list(times_seizure[0])
    times_after_seizure = list(times_seizure[1])

    times_before_seizure = map(float,times_before_seizure)
    times_after_seizure = map(float,times_after_seizure)

    return times_before_seizure, times_after_seizure



def find_created_datasets(list_signals_file,
    time_before_seizure, time_after_seizure):
    retrieve_seizure_times_file(list_signals_file)
    times_before_seizure_file, times_after_seizure_file = retrieve_seizure_times_file(list_signals_file)

    if all(times_before_seizure_file > time_before_seizure\
            for time_before_seizure in times_before_seizure_file)\
        and\
       all(times_after_seizure_file > time_after_seizure\
            for time_after_seizure in times_after_seizure_file):

        


    # if all(times_before_seizure_file > time_before_seizure
    #     for time_before_seizure in times_before_seizure_file) and 
    #    all(times_after_seizure_file > time_after_seizure
    #     for time_after_seizure in times_after_seizure_file):
    #     print 'hello'
    stop
    return name in list_signals_file


def get_datasets(time_before_seizure, time_after_seizure,
                 dataset_name='HSM', group='raw'):

    print dataset_name
    if dataset_name == 'HSM':
        path = '~/Desktop/seizure_datasets.h5'
        print path
    elif dataset_name == 'phisionet':
        path = '~/Desktop/phisionet_datasets.h5'
    else:
        logger.debug('Invalid dataset name')
        stop

    list_signals_structure = list_group_signals(path, group)
    print list_signals_structure['signals'][0]


    list_signals_file = get_list_group_signals(group, list_signals_structure)

    find_created_datasets(list_signals_file,
        time_before_seizure, time_after_seizure)



time_before_seizure = 30
time_after_seizure = 10

get_datasets(time_before_seizure, time_after_seizure)