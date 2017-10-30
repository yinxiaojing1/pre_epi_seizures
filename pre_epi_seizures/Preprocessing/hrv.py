import numpy as np

from resampling import *


def hrv_computation(signal_arguments, sampling_rate, window_params, add_params, win_param_to_process, param_to_process):
    signal_list = signal_arguments['feature_group_to_process']
    rpeaks_list = signal_arguments['feature_group_to_process']

    hrv_list = map(compute_hrv, rpeaks_list[0])

    # resampling 
    domains = [rpeak[1:] for rpeak in rpeaks_list[0]]
    new_domains_i = [np.arange(rpeak[1], rpeak[-1] + 1, 1)] * len(rpeaks_list)
    # new_domains = [np.linspace(rpeak[1], rpeak[-1], len(signal[rpeak[1]:rpeak[-1]+1])) for signal in signal_list[0]]
    
    # print 'here'
    # print new_domains_i[0]
    # # print new_domains[0]

    # stop
    
    # stop

    hrv_list = [np.asarray([interpolate(hrv, new_domain, domain)]) for hrv, new_domain, domain in zip(hrv_list, new_domains_i, domains)]

    # hrv_mean_list = [compute_mean_NN(10, hrv_signal, sampling_rate) for hrv_signal in hrv_list]

    # hrv_sd_list = [compute_SD_NN(10, hrv_signal, sampling_rate) for hrv_signal in hrv_list]


    mdata = [{'feature_legend': ['hrv']}] * len(hrv_list)

    return hrv_list, mdata, new_domains_i


def hrv_time_features(signal_arguments, sampling_rate, window_params, add_params, win_param_to_process, param_to_process):

    hrv_signal_list = signal_arguments['feature_group_to_process']
    window = window_params['win']

    hrv_time_features = [_hrv_time_features(window, hrv_signal, sampling_rate) for hrv_signal in hrv_signal_list]
    mdata = [{'hrv_time_series': 0}] * len(hrv_list)

    stop
    return hrv_list, mdata

def _hrv_time_features(window, hrv_signal_array, sampling_rate):
    print hrv_signal_array
    hrv_signal = hrv_signal_array[0]
    print hrv_signal
    mean_NN = compute_mean_NN(window, hrv_signal, sampling_rate)
    SD_NN = compute_SD_NN(window, hrv_signal, sampling_rate)
    p_NN50 = compute_pNN50(window, hrv_signal, sampling_rate)
    print mean_NN
    print SD_NN
    print p_NN50
    stop
    return np.asarray([mean_NN, SD_NN])


def rri_corrected_computation(rpeaks, criterion='Malik'):
    raw_rri = np.diff(rpeaks)
    diff_rri = np.true_divide(abs(np.diff(raw_rri)),raw_rri[:-1])
    remove_index_diff_rri = np.where(diff_rri > 0.32)[0]
    corrected_rri = np.delete(raw_rri, remove_index_diff_rri + 1)
    corrected_rpeaks = np.delete(rpeaks, remove_index_diff_rri + 2)

    return corrected_rri, corrected_rpeaks


def compute_hrv(rpeaks):
    # print rpeaks
    # c_rri, c_rpeaks = rri_corrected_computation(rpeaks)


    return 1.0*60*1000/np.diff(rpeaks)


def compute_mean_NN(window, hrv_signal, sampling_rate):
    # hrv_signal = hrv_signal_array[0]
    n_samples = window * sampling_rate
    return [np.mean(hrv_signal[i*n_samples:(i*n_samples) + n_samples]) for i in xrange(len(hrv_signal)/n_samples)]


def compute_SD_NN(window, hrv_signal, sampling_rate):
    n_samples = window * sampling_rate
    # hrv_signal = hrv_signal_array[0]
    return [np.std(hrv_signal[i*n_samples:(i*n_samples) + n_samples]) for i in xrange(len(hrv_signal)/n_samples)]


def compute_pNN50(window, hrv_signal, sampling_rate):
    n_samples = window * sampling_rate
    return [_compute_pNN50(hrv_signal[i*n_samples:(i*n_samples) + n_samples]) for i in xrange(len(hrv_signal)/n_samples)]


def _compute_pNN50(segment):
    print 'segment'
    print segment
    p_NN50 = len(np.where(segment == 1/(60*0.05)))
    return p_NN50


def hrv_time_domain_features(signal_arguments, sampling_rate, params):
    signal_list = signal_arguments['feature_group_to_process']
    window = params['window']

    print signal_list

    feature_list = [hrv_time_features(window, signal, sampling_rate)
                    for signal in signal_list]

    mdata_list = [{'fs':sampling_rate}] * len(feature_list)

    return feature_list, mdata_list