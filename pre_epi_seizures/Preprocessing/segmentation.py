from pre_epi_seizures.logging_utils.formatter_logging\
import logger as _logger

from pre_epi_seizures.storage_utils.storage_utils_hdf5\
import save_signal, load_signal, delete_signal

from pre_epi_seizures.storage_utils.data_handlers import *

# from pre_epi_seizures.Preprocessing.pre_processing\
# import input_default_params

from Filtering.eksmoothing import *

from resampling import *

from biosppy.signals import ecg

import numpy as np

import matplotlib.pyplot as plt

import functools



def rpeak_detection(signal_arguments, win_params, add_params, win_param_to_process, param_to_process):
    signal_list = signal_arguments['feature_group_to_process']
    

    # Compute feature
    method = add_params['method']
    sampling_rate = win_param_to_process['samplerate']
    rpeaks = map(functools.partial(run_univariate_rpeaks_detection,
                                   method=method,
                                   sampling_rate=sampling_rate),
                 signal_list)

    window_list = rpeaks
    mdata_list = [{'feature_legend':['rpeaks']}] * len(rpeaks)
    return rpeaks, mdata_list, window_list


def run_univariate_rpeaks_detection(feature_array, method, sampling_rate=1000):
    # Select function of rpeak detection
    if method == 'hamilton':
        detector = ecg.hamilton_segmenter
    
    # Compute Rpeaks for each feature_signal (ECG)
    rpeaks_feature_array = np.array([detect_rpeaks(feature_signal,
                                                   detector,
                                                   sampling_rate)
                                     for feature_signal in feature_array])
    
    return rpeaks_feature_array


def detect_rpeaks(feature_signal, detector, sampling_rate):
    # detect rpeaks
    rpeaks = detector(signal=feature_signal,
                      sampling_rate=sampling_rate)['rpeaks']
    return rpeaks


def visual_inspection(signal_list,
                      rpeaks,
                      begin_second,
                      end_second):
    signal = signal_list[0][0]
    l = len(signal)
  
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(np.linspace(0, l-1, l), signal)
    rpeaks = rpeaks[0][0]
    plt.plot(rpeaks, signal[rpeaks], 'o')
    plt.xlim(begin_second * 1000, end_second * 1000)
    plt.show()


def QRS_fixed_segmentation(signal_arguments,
                           win_params, add_params,
                           win_param_to_process, param_to_process):
    #CHANGE URGENTLY*************************************************************
    signal_list = signal_arguments['rpeak_group_to_process']
    rpeaks_list = signal_arguments['feature_group_to_process']
    sampling_rate = win_params['samplerate']
    
    try:

        beats = [compute_QRS(signal, rpeaks, sampling_rate) 
                 for signal, rpeaks in zip(signal_list, rpeaks_list)]

        domains = [rpeaks[0][1:-1] for rpeaks in rpeaks_list]
        
    except Exception as e:
        beats = [[]]
        domains = [[]]
    
    
    mdata = [''] * len(rpeaks_list)
    return beats, mdata, domains


def beat_phase_segmentation(signal_arguments,
                           win_params, add_params,
                           win_param_to_process, param_to_process):
    print 'fdkfjsadkfjla'
    signal_list = signal_arguments['feature_group_to_process']
    rpeaks_list = signal_arguments['rpeak_group_to_process']
    sampling_rate = win_params['samplerate']


    beats = [compute_beat_phase(signal, rpeaks, sampling_rate) for signal, rpeaks in zip(signal_list, rpeaks_list)]
    domains = rpeaks_list
    mdata = [''] * len(rpeaks_list)
    # the signals are saved in reverse order since 
    # it isn't possible to save them in h5py datasets
    return beats, mdata, domains


def compute_beat_phase(signal, rpeaks, sampling_rate):
    signal = signal[0]
    rpeaks = rpeaks[0]
    
    # stop
    phase = get_phase(signal, rpeaks)
    idx_up = np.where(abs(np.diff(phase)) > 6)[0]

    beats = [signal[i:f+1] for i,f in zip(idx_up[0:-1], idx_up[1:])]
    domains = [np.linspace(-np.pi, np.pi, len(beat), endpoint=True) for beat in beats]

    new_domain = np.linspace(-np.pi, np.pi, 1000, endpoint=True)
    new_beats = [interpolate(beat, new_domain, domain) for beat, domain in zip(beats, domains)]

    return np.array(new_beats).T


def compute_hrv(rpeaks):
    print rpeaks
    return 1.0/np.diff(rpeaks)


def find_rpeaks(rpeaks, start, end): 
    print rpeaks
    stop
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


