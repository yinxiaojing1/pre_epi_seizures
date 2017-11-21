import numpy as np

from scipy import signal

from resampling import *

from decimal import *

def hrv_computation(signal_arguments, window_params, add_params, win_param_to_process, param_to_process):
    signal_list = signal_arguments['feature_group_to_process']
    rpeaks_list = signal_arguments['feature_group_to_process']

    print rpeaks_list


    hrv_list = map(compute_hrv, rpeaks_list[0])
    print hrv_list
 
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


def hrv_time_features(signal_arguments, window_params, add_params, win_param_to_process, param_to_process):
    rpeaks_list = signal_arguments['rpeak_group_to_process']
    hrv_signal_list = signal_arguments['feature_group_to_process']
    window = window_params['win']


    init = window_params['init']

    sampling_rate = window_params['samplerate']

    overlap = 0.75


    # stop
    n_samples = (window * sampling_rate)

    # stop

    finish = window_params['finish']



    window_tuple_list = create_window_tuple(rpeaks_list, window, overlap, sampling_rate)


    windows_list = create_window_time_domain(window_tuple_list)
    # split_list = 

    split_list = get_hrv_window(hrv_signal_list, window_tuple_list)


    hrv_features = [compute_hrv_features(hrv_window, sampling_rate)
                    for hrv_window in split_list]

    # windows_list = [window] * len(hrv_signal_list)

    # split_list = fixed_window_split(hrv_signal_list, windows_list)
    # print split
    # print hrv_time_features


    mdata = [{'feature_legend': ['mean_NN', 'SD_NN', 'p_NN50', 'var_NN',
                                 'LF', 'HF', 'LF_HF']}] * len(hrv_features)
    # stop

    return hrv_features, mdata, windows_list


# create window ------------------------------------------------------------------------
def _create_window_tuple(rpeaks, window_sec, overlap_frac, sampling_rate):
    step = int((1 - overlap_frac) * window_sec * sampling_rate)
    return [(i, i + (window_sec * sampling_rate)) for i in xrange(rpeaks[1], rpeaks[-1], step)]


def create_window_tuple(rpeaks_list, window_sec, overlap_frac, sampling_rate):
    return [_create_window_tuple(rpeaks[0], window_sec, overlap_frac, sampling_rate)
            for rpeaks in rpeaks_list]


def _create_window_time_domain(window_tuple_l):
    return [window_tuple[0] for window_tuple in window_tuple_l]


def create_window_time_domain(window_tuple_list):
    return map(_create_window_time_domain, window_tuple_list)


def _get_hrv_window(hrv_signal, window_tuple_list):
    hrv_signal = hrv_signal[0]
    signal =[hrv_signal[window_tuple[0]:window_tuple[1]]
              for window_tuple in window_tuple_list]
    return signal


def get_hrv_window(hrv_signal_list, window_tuple_list):
    return [_get_hrv_window(hrv_signal, window_tuple_l)
            for hrv_signal, window_tuple_l\
             in zip(hrv_signal_list, window_tuple_list)]


def fixed_window_split(feature_list, windows_list):
    map_arg = zip(feature_list, windows_list)
    return map(_fixed_window_split, map_arg)


def _fixed_window_split(feature_array_window_tuple):
    return np.split(feature_array_window_tuple[0], feature_array_window_tuple[1], axis=1)[0:-1]
# def _split(feature, n_samples):
#     win = np.arange(0, len(feature), n_samples)
#     split = np.split(hrv_signal_list[0][0], win[1:])
#     return split


# 

def compute_hrv_features(hrv_split_array, sampling_rate):

    mean_NN = map(compute_mean_NN, hrv_split_array)
    SD_NN = map(compute_SD_NN, hrv_split_array)
    p_NN50 = map(compute_pNN50, hrv_split_array)
    var_NN = map(compute_var_NN, hrv_split_array)

    PSD_array = [compute_PSD(hrv_signal, sampling_rate) 
                 for hrv_signal in hrv_split_array]
    LF = map(compute_LF, PSD_array)
    HF = map(compute_HF, PSD_array)

    LF_HF_ratio = map(compute_LF_HF_ratio, zip(LF, HF))

    features =  np.asarray([mean_NN, SD_NN, p_NN50, var_NN,
                            LF, HF, LF_HF_ratio])
    return features


def compute_PSD(hrv_signal, sampling_rate):
    fs = sampling_rate
    f, Pxx_den = signal.periodogram(hrv_signal, fs)


    return f, Pxx_den


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


    return np.diff(rpeaks)/1000.0


def compute_mean_NN(hrv_signal):
    return np.mean(hrv_signal)
    # return [np.mean(hrv_signal[i*n_samples:(i*n_samples) + n_samples]) for i in xrange(len(hrv_signal)/n_samples)]


def compute_SD_NN(hrv_signal):
    # hrv_signal = hrv_signal_array[0]
    return np.std(hrv_signal)


def compute_var_NN(hrv_signal):
    return np.var(hrv_signal)


def compute_pNN50(hrv_signal):
    p_NN50 = len(np.where(hrv_signal < 60*0.05)[0])
    return p_NN50


def compute_LF(PSD_array):
    lf_lower = 0.04
    lf_upper = 0.15
    f = PSD_array[0]
    Pxx_den = PSD_array[1]

    lf = np.where(np.logical_and(f > lf_lower, f < lf_upper))[0]
    LF = sum(Pxx_den[lf])

    return LF


def compute_HF(PSD_array):
    hf_lower = 0.15
    hf_upper = 0.4
    f = PSD_array[0]
    Pxx_den = PSD_array[1]
    # print f
    # print Pxx_den


    hf = np.where(np.logical_and(f > hf_lower, f < hf_upper))[0]

    HF = sum(Pxx_den[hf])
    return HF


def compute_LF_HF_ratio(LF_HF):
    LF = LF_HF[0]
    HF = LF_HF[1]
    return LF/HF


def hrv_time_domain_features(signal_arguments, sampling_rate, params):
    signal_list = signal_arguments['feature_group_to_process']
    window = params['window']

    print signal_list

    feature_list = [hrv_time_features(window, signal, sampling_rate)
                    for signal in signal_list]

    mdata_list = [{'fs':sampling_rate}] * len(feature_list)

    return feature_list, mdata_list