from pre_epi_seizures.logging_utils.formatter_logging\
import logger as _logger

from pre_epi_seizures.storage_utils.storage_utils_hdf5\
import save_signal, load_signal, delete_signal

from pre_epi_seizures.storage_utils.data_handlers import *

from biosppy.signals import ecg

import numpy as np

import matplotlib.pyplot as plt

def shape_array(array):
    return np.array([array]).T


def create_rpeak_dataset(path, group_name_list):
    _logger.info('Detecting the R-peaks')
    X = load_signal(path=path, group_name_list=group_name_list)

    for group_name in group_name_list:
        dataset = get_multiple_records(get_one_signal_structure(X, group_name_list[0]))
        labels = map(str, range(0,len(dataset)))

        rpeaks_name = 'rpeaks'
        print  dataset

        rpeaks = map(detect_rpeaks, dataset)
        print rpeaks

        name_list = map(create_rpeak_label, labels)
        print name_list

        mdata_list = [''] * len(labels)
        print mdata_list

        group_list = [group_name[0]] * len(labels)
        print group_list

        save_signal(path=path, signal_list=rpeaks,
                mdata_list=mdata_list, name_list=name_list, group_list=group_list)


def detect_rpeaks(record):
    rpeaks = ecg.hamilton_segmenter(signal=record,
                                    sampling_rate=1000)
    return rpeaks['rpeaks']


def create_rpeak_label(label):
    return 'rpeaks_' + label


# # def compute_rpeaks(signal, Fs, method='hamilton'):
# #     if method='hamilton':
# #         return ecg.hamilton_segmenter(signal=signal_to_filter[i,:], sampling_rate=Fs)


# # def interp_rpeaks(rpeaks, )

def compute_beats(signal, rpeaks):
    return [signal[rpeak - 400:rpeak + 600] for rpeak in rpeaks[1:-2]]



def create_heart_beat_dataset(path, name, group, save_dfile=None):
    X = load_signal(path=path, name=name, group=group) # 1 record per row
    rpeaks = load_signal(path=path, name='rpeaks', group=group)
    signal_to_segment = X['signal'][0,:].T
    heart_beat, rpeaks = ecg.extract_heartbeats(
                                 signal=signal_to_segment,
                                 rpeaks=rpeaks['signal'], sampling_rate=1000.0,
                                 before=0.2, after=0.4
                                 )
    return heart_beat, rpeaks


