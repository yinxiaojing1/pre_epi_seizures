from pre_epi_seizures.logging_utils.formatter_logging\
import logger as _logger

from pre_epi_seizures.storage_utils.storage_utils_hdf5\
import save_signal, load_signal, delete_signal

from pre_epi_seizures.storage_utils.data_handlers import *

from Filtering.eksmoothing import *

from resampling import *

from biosppy.signals import ecg

import numpy as np

import matplotlib.pyplot as plt

import functools



def rpeak_detection(signal_arguments, sampling_rate):
    signal_list = signal_arguments['feature_group_to_process']
    print signal_list
    # stop
    rpeaks = map(functools.partial(detect_rpeaks,
                 sampling_rate=sampling_rate), signal_list)
    print rpeaks
    mdata = [''] * len(rpeaks)
    return rpeaks, mdata


def QRS_fixed_segmentation(signal_arguments, sampling_rate):
    signal_list = signal_arguments['feature_group_to_process']
    rpeaks_list = signal_arguments['rpeak_group_to_process']
    # print rpeaks_list

    # print signal_list
    # print rpeaks_list

    beats = [compute_QRS(signal, rpeaks, sampling_rate) 
        for signal, rpeaks in zip(signal_list, rpeaks_list)]

    print 'done'
    mdata = [''] * len(rpeaks)
    return beats, mdata


def beat_phase_segmentation(signal_arguments, sampling_rate):
    print 'fdkfjsadkfjla'
    signal_list = signal_arguments['feature_group_to_process']
    rpeaks_list = signal_arguments['rpeak_group_to_process']

    beats = [compute_beat_phase(signal, rpeaks, sampling_rate) for signal, rpeaks in zip(signal_list, rpeaks_list)]

    mdata = ['']*len(beats)


    # the signals are saved in reverse order since 
    # it isn't possible to save them in h5py datasets
    return beats, mdata


def compute_beat_phase(signal, rpeaks, sampling_rate):

    phase = get_phase(signal, rpeaks)
    idx_up = np.where(abs(np.diff(phase)) > 6)[0]

    beats = [signal[i:f+1] for i,f in zip(idx_up[0:-1], idx_up[1:])]
    domains = [np.linspace(-np.pi, np.pi, len(beat), endpoint=True) for beat in beats]

    new_domain = np.linspace(-np.pi, np.pi, 1000, endpoint=True)
    new_beats = [interpolate(beat, new_domain, domain) for beat, domain in zip(beats, domains)]

    return new_beats



def compute_hrv(rpeaks):
    print rpeaks
    return 1.0/np.diff(rpeaks)


def detect_rpeaks(feature_array, sampling_rate=1000):
    record = feature_array[0]
    rpeaks = ecg.hamilton_segmenter(signal=record,
                                    sampling_rate=sampling_rate)
    return np.asarray([rpeaks['rpeaks']])


def find_rpeaks(rpeaks, start, end): 
    samples = np.arange(start, end, 1)
    goodvalues = samples
    ix = np.in1d(rpeaks.ravel(), goodvalues).reshape(rpeaks.shape)
    return rpeaks[np.where(ix)[0]]


def compute_QRS(signal, rpeaks, sampling_rate):
    signal = signal[0]
    rpeaks = rpeaks[0]
    beats = np.asarray([signal[rpeak - int(0.04*sampling_rate):rpeak + int(0.06*sampling_rate)] for rpeak in rpeaks[1:-1]])
    return beats.T

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


