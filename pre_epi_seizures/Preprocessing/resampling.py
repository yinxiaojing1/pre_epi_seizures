import scipy as sp
import numpy as np

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


def interpolate_signal(signal_array, time_array, type='linear'):
    np.interp1d(signal_array, time_array,kind='linear')