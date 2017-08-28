from pre_epi_seizures.logging_utils.formatter_logging\
import logger as _logger

from pre_epi_seizures.storage_utils.storage_utils_hdf5\
import save_signal, load_signal, delete_signal

from pre_epi_seizures.storage_utils.data_handlers import *

from biosppy.signals import ecg

import numpy as np

import matplotlib.pyplot as plt

import functools


def hrv_computation(signal_arguments, sampling_rate):
    rpeaks_list = signal_arguments['feature_group_to_process']
    print rpeaks_list
    hrv_list = map(compute_hrv, rpeaks_list)
    print hrv_list[1]
    mdata = [''] * len(hrv_list)
    return hrv_list, mdata


def rpeak_detection(signal_arguments, sampling_rate):
    signal_list = signal_arguments['feature_group_to_process']
    rpeaks = map(functools.partial(detect_rpeaks,
                 sampling_rate=sampling_rate), signal_list)
    print rpeaks
    mdata = [''] * len(rpeaks)
    return rpeaks, mdata


def QRS_fixed_segmentation(signal_arguments, sampling_rate):
    signal_list = signal_arguments['feature_group_to_process']
    rpeaks_list = signal_arguments['rpeak_group_to_process']
    # print rpeaks_list

    beats = [compute_QRS(signal, rpeaks, sampling_rate) 
        for signal, rpeaks in zip(signal_list, rpeaks_list)]

    print 'done'
    mdata = [''] * len(rpeaks)
    return beats, mdata

def compute_hrv(rpeaks):
    print rpeaks
    return 1.0/np.diff(rpeaks)


def detect_rpeaks(record, sampling_rate=1000):
    rpeaks = ecg.hamilton_segmenter(signal=record,
                                    sampling_rate=sampling_rate)

    return rpeaks['rpeaks']


def find_rpeaks(rpeaks, start, end): 
    samples = np.arange(start, end, 1)
    goodvalues = samples
    ix = np.in1d(rpeaks.ravel(), goodvalues).reshape(rpeaks.shape)
    return rpeaks[np.where(ix)[0]]


def compute_QRS(signal, rpeaks, sampling_rate):
    return [signal[rpeak - int(0.4*sampling_rate):rpeak + int(0.6*sampling_rate)] for rpeak in rpeaks[1:-1]]


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


