import numpy as np
from sklearn.decomposition import PCA

import numpy as np

from scipy import signal

from resampling import *

from decimal import *



def pca_beat_amp_computation(signal_arguments,
                             win_params, add_params,
                             win_param_to_process, param_to_process):
    signal_list = signal_arguments['rpeak_group_to_process']
    rpeaks_list = signal_arguments['feature_group_to_process']
    nr_comp = add_params['nr_comp']
    
    parameters_list = [trace_evol_PC(beats_list.T, nr_comp) for beats_list in signal_list]
    # print parameters_list
    mdata = [{'feature_legend': ['eig_value_' + str(i) for i in xrange(1, nr_comp + 1)]}] * len(parameters_list)
    # print parameters_list
    window_list = [rpeaks[0][nr_comp:-1] for rpeaks in rpeaks_list]
    



    return parameters_list, mdata, window_list


def compute_PCA_sklearn(beats_array):
    pca = PCA(n_components=5, svd_solver='full')
    return pca.fit(np.asarray(beats_array).T)


def compute_PC(beats_array):
    print len(beats_array)
    print len(beats_array[0])
    # stop
    norm_covariance = np.cov(np.asarray(beats_array))
    return np.linalg.eigh(norm_covariance)[0]


def trace_evol_PC(beats_total_array, begin):
    print beats_total_array
    # print len(beats_total_array)
    # stop
    # stop
    return np.asarray([compute_PC(beats_total_array[i-begin:i]) 
                for i in xrange(begin, len(beats_total_array)+1, 1)]).T

