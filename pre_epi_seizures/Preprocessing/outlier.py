import numpy as np
import scipy
import scipy.spatial
from scipy import linalg

def cosv(a,b):
    return (scipy.dot(a,b)/(linalg.norm(a)*linalg.norm(b)))

def msedistance(s1, s2):
    return scipy.spatial.distance.euclidean(s1, s2)

def cosdistance(s1,s2):
    return 1-abs(cosv(s1, s2))

def wavedistance(testwave, trainwaves, fdistance):
    return scipy.array([fdistance(wave, testwave) for wave in trainwaves])

def dmean(data=None, R_Position=200, alpha=0.5, beta=1.5, metric="euclidean", checkR=True):
    """
    DMEAN outlier detection heuristic.
    
    Determines which (if any) samples in a block of ECG templates are outliers.
    A sample is considered valid if it cumulatively verifies:
        - distance to average template smaller than a (data derived) threshold T;
        - sample minimum greater than a (data derived) threshold M;
        - sample maximum smaller than a (data derived) threshold N;
        - [optional] position of the sample maximum is the same as the given R position.
    
    For a set of {X_1, ..., X_n} n samples,
    Y = \frac{1}{n} \sum_{i=1}^{n}{X_i}
    d_i = dist(X_i, Y)
    D_m = \frac{1}{n} \sum_{i=1}^{n}{d_i}
    D_s = \sqrt{\frac{1}{n - 1} \sum_{i=1}^{n}{(d_i - D_m)^2}}
    T = D_m + \alpha * D_s
    M = \beta * median({max(X_i), i=1, ..., n})
    N = \beta * median({min(X_i), i=1, ..., n})
    
    Input:
        data (array): input data (number of samples x number of features)  
        R_Position (int): Position of the R peak. 
        alpha (float): Parameter for the distance threshold.
        beta (float): Parameter for the maximum and minimum thresholds. 
        metric (string): Distance metric to use (euclidean or cosine).  
        checkR (bool): If True checks the R peak position.
                   
    Output:
        '0' (list): Indices of the normal samples. 
        '-1' (list): Indices of the outlier samples.
    """
    
    # get distance function
    if metric == 'euclidean':
        dist_fcn = msedistance
    elif metric == 'cosine':
        dist_fcn = cosdistance
    else:
        raise ValueError, "Distance %s not implemented." % metric
    
    lsegs = len(data)
    
    # distance to mean wave
    mean_wave = np.mean(data, 0)
    dists = wavedistance(mean_wave, data, dist_fcn)
    
    # distance threshold
    th = np.mean(dists) + alpha * np.std(dists, ddof=1)
    
    # median of max and min
    M = np.median(np.max(data, 1)) * beta
    m = np.median(np.min(data, 1)) * beta
    
    # search for outliers
    outliers = []
    for i in xrange(lsegs):
        R = np.argmax(data[i])
        if checkR and (R != R_Position):
            outliers.append(i)
        elif data[i][R] > M:
            outliers.append(i)
        elif np.min(data[i]) < m:
            outliers.append(i)
        elif dists[i] > th:
            outliers.append(i)
    
    outliers = set(outliers)
    normal = list(set(range(lsegs)) - outliers)
    
    return {'0': normal, '-1': list(outliers)}