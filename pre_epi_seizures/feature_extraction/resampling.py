import numpy as np
import scipy as sp


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


def interpolate(t, y, new_sampling_rate):
    inter = sp.interpolate.interp1d(t, y, kind='cubic')
    x_new = np.linspace(t[0], t[-1], (t[-1]-t[0])*new_sampling_rate)
    y_new = inter(x_new)
    return y_new, x_new




def interpolate_signal(signal_array, factor):
    N = len(signal_array)
    total_samples_down = np.linspace(0, N-1, N)
    print 'down'
    print total_samples_down
    inter = sp.interpolate.interp1d(total_samples_down, signal_array, kind='cubic')
    N_up = N*factor
    total_samples_up = np.linspace(0, N-1, N_up)
    print 'up'
    print total_samples_up

    below_bounds = total_samples_up < total_samples_down[0]
    above_bounds = total_samples_up > total_samples_down[-1]

    print np.where(below_bounds)
    signal_array_up = inter(total_samples_up)
    return signal_array_up



def resample(signal_to_resample, new_sampling_rate):
    new_sampling_rate = 2
