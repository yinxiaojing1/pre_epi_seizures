import scipy as sp
import numpy as np
import functools

def resample_rpeaks(rpeaks_interval, rpeaks, time_array): 
    inter = sp.interpolate.interp1d(rpeaks[1:], rpeaks_interval, kind='linear')
    print inter
    N = len(time_array)
    print N
    total_samples = np.linspace(0, N-1, N)
    total_samples_range = total_samples[rpeaks[1]:rpeaks[-1]]
    print total_samples_range
    new_rri = inter(total_samples_range)
    return new_rri

def decimation(signal_arguments, sampling_rate):
    signal_list = signal_arguments['feature_group_to_process']
    factor = 2
    decimated_signals = map(functools.partial(decimate_record,
                 factor=factor), signal_list)
    mdata = [{'fs': sampling_rate/factor}]*len(decimated_signals)
    return decimated_signals, mdata


def interpolation(signal_arguments, sampling_rate):
    signal_list = signal_arguments['feature_group_to_process']
    if sampling_rate != None:
    # if sampling_rate == None
        new_sampling_rate = 1000
        interpolated_signal = map(functools.partial(interpolate_signal,
                    new_sampling_rate=new_sampling_rate, sampling_rate=sampling_rate), signal_list)
        mdata = [{'fs': new_sampling_rate}]
        return interpolated_signal, mdata

def decimate_record(record, factor):
    print record
    return sp.signal.decimate(record, factor)

def interpolate_signal(signal_array, new_sampling_rate, sampling_rate):
    print signal_array
    N = len(signal_array)
    print N
    tf = (N-1.0)/sampling_rate
    total_samples_down = np.linspace(0, tf, N)
    inter = sp.interpolate.interp1d(total_samples_down, signal_array, kind='cubic')
    N_up = N * (new_sampling_rate/sampling_rate)
    total_samples_up = np.linspace(0, tf, N_up)
    print 'up'
    print total_samples_up
    signal_array_up = inter(total_samples_up)
    return signal_array_up