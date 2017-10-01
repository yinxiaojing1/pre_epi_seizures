import numpy as np
from sklearn.decomposition import PCA


def pca_beat_amp_computation(signal_arguments, sampling_rate, params):
    seizure_list = signal_arguments['feature_group_to_process']
    begin = params['begin']
    parameters_list = [trace_evol_PC(beats_list, begin) for beats_list in seizure_list]
    # print parameters_list
    mdata = [''] * len(parameters_list)
    return parameters_list, mdata


def compute_PCA_sklearn(beats_array):
    pca = PCA(n_components=5, svd_solver='full')
    return pca.fit(np.asarray(beats_array).T)


def compute_PC(beats_array):
    norm_covariance = np.cov(np.asarray(beats_array))
    return np.linalg.eigh(norm_covariance)[0]


def trace_evol_PC(beats_total_array, begin):
    print len(beats_total_array)
    return np.asarray([compute_PC(beats_total_array[i-begin:i]) 
                for i in xrange(begin, len(beats_total_array)+1, 1)])

