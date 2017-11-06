import numpy as np

from resampling import *


def hrv_computation(signal_arguments, sampling_rate, window_params, add_params, win_param_to_process, param_to_process):
    signal_list = signal_arguments['feature_group_to_process']
    rpeaks_list = signal_arguments['feature_group_to_process']

    hrv_list = map(compute_hrv, rpeaks_list[0])

    # resampling 
    domains = [rpeak[1:] for rpeak in rpeaks_list[0]]
    new_domains_i = [np.arsange(rpeak[1], rpeak[-1] + 1, 1)] * len(rpeaks_list)
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

    sampling_rate = window_params['samplerate']
    # stop
    n_samples = window * sampling_rate

    finish = window_params['finish']

    window = np.arange(n_samples, finish * sampling_rate, n_samples)
    windows_list = [window] * len(hrv_signal_list)

    split_list = fixed_window_split(hrv_signal_list, windows_list)
    # print split
    # print hrv_time_features

    hrv_time_features = map(_hrv_time_features, split_list)
    mdata = [{'feature_legend': ['mean_NN', 'SD_NN', 'p_NN50']}] * len(hrv_time_features)


    return hrv_time_features, mdata, windows_list


def fixed_window_split(feature_list, windows_list):
    map_arg = zip(feature_list, windows_list)
    return map(_fixed_window_split, map_arg)


def _fixed_window_split(feature_array_window_tuple):
    return np.split(feature_array_window_tuple[0], feature_array_window_tuple[1], axis=1)[0:-1]
# def _split(feature, n_samples):
#     win = np.arange(0, len(feature), n_samples)
#     split = np.split(hrv_signal_list[0][0], win[1:])
#     return split


def _hrv_time_features(hrv_split_array):
    print hrv_split_array
    hrv_feature = [hrv[0] for hrv in hrv_split_array]

    print hrv_feature

    mean_NN = map(compute_mean_NN, hrv_feature)
    SD_NN = map(compute_SD_NN, hrv_feature)
    p_NN50 = map(compute_pNN50, hrv_feature)
    return np.asarray([mean_NN, SD_NN, p_NN50])


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


def compute_mean_NN(hrv_signal):
    return np.mean(hrv_signal)
    # return [np.mean(hrv_signal[i*n_samples:(i*n_samples) + n_samples]) for i in xrange(len(hrv_signal)/n_samples)]


def compute_SD_NN(hrv_signal):
    # hrv_signal = hrv_signal_array[0]
    return np.std(hrv_signal)



# def compute_pNN50(n_samples, hrv_signal, sampling_rate):
#     return [_compute_pNN50(hrv_signal[i*n_samples:(i*n_samples) + n_samples]) for i in xrange(len(hrv_signal)/n_samples)]


def compute_pNN50(hrv_signal):
    # print 'hrv_signal'
    # print hrv_signal
    p_NN50 = len(np.where(hrv_signal > 1/(60 * 0.05))[0])
    return p_NN50


def hrv_time_domain_features(signal_arguments, sampling_rate, params):
    signal_list = signal_arguments['feature_group_to_process']
    window = params['window']

    print signal_list

    feature_list = [hrv_time_features(window, signal, sampling_rate)
                    for signal in signal_list]

    mdata_list = [{'fs':sampling_rate}] * len(feature_list)

    return feature_list, mdata_list