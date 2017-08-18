import numpy as np
from sklearn.decomposition import PCA



def compute_PCA_sklearn(beats_array):
    pca = PCA(n_components=5,svd_solver='full')
    print pca
    return pca.fit(np.asarray(beats_array).T)


def compute_PC(beats_array):
    norm_covariance = np.cov(np.asarray(beats_array))
    return np.linalg.eigh(norm_covariance)[0]


def trace_evol_PC(beats_total_array):
    print len(beats_total_array)
    return np.asarray([compute_PC(beats_total_array[i-5:i]) 
                for i in xrange(5, len(beats_total_array)+1, 1)])



# def trace_eigein_evol(beats): 
#     for beat in beats
