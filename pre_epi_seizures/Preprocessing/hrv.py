import numpy as np

from resampling import *


def hrv_computation(signal_arguments, sampling_rate, window_params, add_params):
    signal_list = signal_arguments['feature_group_to_process']
    rpeaks_list = signal_arguments['rpeak_group_to_process']
    window_msec = params['window']


    print signal_list
    print rpeaks_list

    # stop
    hrv_list = map(compute_hrv, rpeaks_list[0])
    # print hrv_list[1]

    domains = [rpeak[1:] for rpeak in rpeaks_list[0]]
    new_domains = [np.linspace(rpeak[1], rpeak[-1], len(signal[rpeak[1]:rpeak[-1]+1])) for signal in signal_list[0]]
    hrv_list = [np.asarray([interpolate(hrv, new_domain, domain)]) for hrv, new_domain, domain in zip(hrv_list, new_domains, domains)]

    # hrv_mean_list = [compute_mean_NN(10, hrv_signal, sampling_rate) for hrv_signal in hrv_list]

    # hrv_sd_list = [compute_SD_NN(10, hrv_signal, sampling_rate) for hrv_signal in hrv_list]


    mdata = [{'fs':sampling_rate}] * len(hrv_list)

    return hrv_list, mdata


def hrv_time_features(window, hrv_signal_array, sampling_rate):
    hrv_signal = hrv_signal_array[0]
    mean_NN = compute_mean_NN(window, hrv_signal, sampling_rate)
    SD_NN = compute_SD_NN(window, hrv_signal, sampling_rate)
    print mean_NN
    print SD_NN

    # stop
    return np.asarray([mean_NN, SD_NN])

def compute_hrv(rpeaks):
    print rpeaks
    return 1.0*60*1000/np.diff(rpeaks)


def compute_mean_NN(window, hrv_signal, sampling_rate):
    # hrv_signal = hrv_signal_array[0]
    n_samples = window * sampling_rate
    return [np.mean(hrv_signal[i*n_samples:(i*n_samples) + n_samples]) for i in xrange(len(hrv_signal)/n_samples)]


def compute_SD_NN(window, hrv_signal, sampling_rate):
    n_samples = window * sampling_rate
    # hrv_signal = hrv_signal_array[0]
    return [np.std(hrv_signal[i*n_samples:(i*n_samples) + n_samples]) for i in xrange(len(hrv_signal)/n_samples)]



def hrv_time_domain_features(signal_arguments, sampling_rate, params):
    signal_list = signal_arguments['feature_group_to_process']
    window = params['window']

    print signal_list

    feature_list = [hrv_time_features(window, signal, sampling_rate)
                    for signal in signal_list]

    mdata_list = [{'fs':sampling_rate}] * len(feature_list)

    return feature_list, mdata_list

