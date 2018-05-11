"""
.. module:: outlier detection 
   :platform: Unix, Windows
   :synopsis: This module provides various functions to...

.. moduleauthor:: Filipe Canento, Andre Lourenco


"""
import os
from multiprocessing import Queue

import numpy as np
import scipy
import pylab
import scipy.spatial.distance as dist

from cluster import cluster
from Cloud import parallel
from datamanager import datamanager
from misc import misc

# Notes:

# input structure of type n observations x n features
# output of type dict {'outliers': outliers indexes, 'normal': normal indexes}



def selector(method):
    """
    Selector for the outlier detection functions and methods.
    
    Input:
        method (str): The desired function or method.
    
    Output:
        fcn (function): The function pointer.
    
    Configurable fields:
    
    See Also:
    
    
    Example:
    
    
    References:
        .. [1]
    """
    
    if method == 'eleazar':
        fcn = eleazar_eskin
    elif method == 'dbscan':
        fcn = outliers_dbscan
    elif method == 'fiducialMedianFilter':
        fcn = fiducialMedianFilter
    elif method == 'meanfilter':
        fcn = meanfilter
    elif method == 'dmean':
        fcn = dmean
    elif method == 'dmean_tst':
        fcn = dmean_tst
    else:
        raise TypeError, "Method %s not implemented." % method
    
    return fcn


def runMethod(data=None, method=None, **kwargs):
    """
    Runs an outlier detection and removal method.
    
    Input:
        data (array): The initial data set.
        
        method (str): The desired function or method.
        
        Additional kwargs are passed to the outlier detection method.
    
    Output:
        templates (array): The data set without the outliers.
        
        partition (dict): The partition dictionary between normal and outlier templates.
        
        nbGood (int): Number of normal templates
    
    Configurable fields:
    
    See Also:
    
    
    Example:
    
    
    References:
        .. [1]
    """
    
    # check inputs
    if data is None:
        raise TypeError, "Please specify an input data set."
    if method is None:
        raise TypeError, "Please specify a method."
    
    # get outlier detection function
    fcn = selector(method)
    
    # detect outliers
    partition = fcn(data, **kwargs)
    
    # remove outliers
    output = removeOutliers(data, partition)
    
    # output
    output['partition'] = partition
    output['nbGood'] = len(output['templates'])
    
    return output


def removeOutliers(data=None, outliers=None):
    """
    Method to remove the outliers from a given data set.
    
    Input:
        data (array): The initial data set.
        
        outliers (dict): The dictionary with the oulier labels.
    
    Output:
        templates (array): The data set without the outliers.
    
    Configurable fields:
    
    See Also:
    
    
    Example:
    
    
    References:
        .. [1]
    """
    
    # check inputs
    if data is None:
        raise TypeError, "Please specify an input data set."
    if outliers is None:
        raise TypeError, "Please provide the outliers."
    
    # make sure data set is array
    data = np.array(data)
    
    # remove the outliers
    try:
        res = data[outliers['0']]
    except KeyError:
        raise KeyError, "The outliers' dictionary does not have the correct key ('0')."
    
    return {'templates': res}
    

def eleazar_eskin(data=None, metric='euclidean', threshold=.99):
    """
    A Geometric Framework for Unsupervised Anomaly Detection:
    Detecting based on Fixed widht-clustering (Alg1)
    intrusions in Unlabeled Data Eleazar Eskin et Al....

    Input:
        data (array): input data (number of observations x number of features).
        
        metric (string): distance metric to be used.
        
        threshold (float): threshold between 0.0 and 1.0.
        
    Output:
        res (dict): output dict with keys "normal" and "outlier" and the corresponding data indexes.

    Configurable fields:{"name": "outlier.eleazar", "config": {"metric": "euclidean", "threshold": 0.99}, "inputs": ["data"]}

    See Also:
    

    Notes:
    

    Example:
    

    References:
        .. [1] Eleazar Eskin et Al ...
        
    """    
    # check inputs
    if data is None:
        raise TypeError, "Input data is needed."
    
    # compute histogram of the distances
    Y = dist.pdist(data, metric) # n*(n-1) / 2
    count, bins = np.histogram(Y, bins=100, range=(min(Y), max(Y)), density=True)
    count = np.cumsum(count)/sum(count)
    
    # find the first bin above threshold
    I = np.nonzero(count > threshold)[0]
    w = bins[I[0]]

    # fig = pylab.figure()
    # ax = fig.add_subplot(111)
    # ax.bar(bins[:-1], count, width = bins[1]-bins[0])
    # ax.axvline(x=w, color='m')
    # pylab.show()
    
    # find outliers
    outlier = []
    Ys = dist.squareform(Y)
    dl = len(data)/2
    for l in xrange(len(Ys)):
        # print np.nonzero(Ys[l] >= w)[0], len(np.nonzero(Ys[l] >= w)[0])
        if len(np.nonzero(Ys[l] >= w)[0]) >= dl:
            outlier.append(l)
    normal = list(set(range(len(data))) - set(outlier))    
    res = {'0': normal, '-1': outlier}
    # raw_input('hist')
    # pylab.close()

    return res

def plotECG(data,res,figProps):
    fig=pylab.figure()
    pylab.clf()
    ax = fig.add_subplot(111)
    
    if res['0'] != []: ax.plot(scipy.array(data[res['0'],:]).T, 'k--', lw=1, label='normal')
    if res['-1'] != []: ax.plot(scipy.array(data[res['-1'],:]).T, 'r--', lw=1, label='outlier')

    ax.grid()
    ax.axis('tight')

    #pylab.legend([h1,h2],'normal','outliers')
    
    ax.set_title(figProps['title'])

    fig.savefig(figProps['figname'])
    # pl.show()

def outliers_dbscan(data= None, min_samples = 10, eps = 0.95, metric = "euclidean"):
    """
    Detect outliers using DBSCAN

    Input:
        data (array): input data (number of observations x number of features).
        
        min_samples (int): minimum number of samples in a cluster. 
        
        eps (float): maximum distance between two samples in the same cluster.
        
        metric (string): distance metric
    
        
    Output:
        res (dict): output dict with keys "0":normal and "-1":outliers and the corresponding data indexes.

    Configurable fields:{"name": "outlier.outliers_dbscan", "config": {"metric": "euclidean", "eps": 0.99, "min_samples": 10}, "inputs": ["data"]}

    See Also:
        BiometricsPyKit.cluster.dbscan

    Notes:
    

    Example:
       
    """    
    
    res = cluster.dbscan(data, min_samples, eps, metric)['clusters']

    try:
        outlier = res['-1']
    except KeyError:
        outlier = []
    
    normal = list(set(range(len(data))) - set(outlier))
        
    res = {'0': normal, '-1': outlier}
    
    return res

def fiducialMedianFilter(data = None,fil=0.4,NS=20):
    """
    Detect outliers using features and medians 

    Input:
        data (array): input data (number of observations x number of features).
                
        fil (float): percentage of median
        
        NS (int): number of points
            
    Output:
        res (dict): output dict with keys "0":normal and "-1":outliers and the corresponding data indexes.

    Configurable fields:{"name": "outlier.fiducialMedianFilter", "config": {"fil": 0.4, "NS": 20}, "inputs": ["data"]}

    References:
        .. [1] Marta Santos et Al, "EIGEN HEARTBEATS FOR USER IDENTIFICATION"
        
    """    

    checkat=np.array([75, 150, 200,300])

    #1st pass
    outlier = []
    for a in checkat:
        mediana=np.median(data[:,a]) 
        for i in range(len(data)):
            if abs(data[i,a]-mediana)>abs(mediana*fil):#+10:#+50:
                outlier.append(i)
    temp = list(set(range(len(data))) - set(outlier))
    
    #2nd pass (over the remaining segments)
    for a in checkat:
        mediana=np.median(data[temp,a])
        for i in temp:
            if abs(data[i,a]-mediana)>abs(mediana*fil):#+10:#+50:
                outlier.append(i)
    
    #mapping over original indices
    #for i in outlier2:
    #    outlier.append(temp[i])
    
    #final "normal" list    
    normal = list(set(set(range(len(data))) - set(outlier)))
        
    res = {'0': normal, '-1': list(set(outlier))}
    
    return res

def meanfilter(data=None, metric="euclidean", th=0.8):
    """
    Detect outliers using distances over the mean  

    Input:
        data (array): input data (number of observations x number of features).
        
        metric : type of metric
                
        th (float): percentage to be removed
        
            
    Output:
        res (dict): output dict with keys "0":normal and "-1":outliers and the corresponding data indexes.

    Configurable fields:{"name": "outlier.modefilter", "config": {"fil": 0.4, "NS": 20}, "inputs": ["data"]}

    References:
        .. [1] 
        
    """    
    # print 'DATA', np.shape(data)
    # if np.shape(data)[0] < 1:
    #     print data
    mean = scipy.mean(data, 0)
    # print 'MEAN', np.shape(mean)
    d = dist.cdist(data, np.mat(mean), metric)
    count, bins = np.histogram(d, 100, density=True)
    
    # i = np.argmax(count, 0) #mode
    
    density = np.cumsum(count, dtype='float')/sum(count)

    I = pylab.find(density > th)
    # print 'JESUS', I


    # fig = pylab.figure()
    # ax = fig.add_subplot(111)
    # ax.bar(bins[:-1], density, width = bins[1]-bins[0])
    w = bins[I[0]]
    # ax.axvline(x=w, color='m')
    # pylab.show()
    
    outlier = []
    for i in xrange(len(d)):
        if d[i] >= w:
            outlier.append(i)

    normal = list(set(range(len(data))) - set(outlier))
        
    res = {'0': normal, '-1': outlier}
    # raw_input('hist')
    # pylab.close()

    return res


def dmean_old(data=None, R_Position=200, alpha=0.5, metric="euclidean"):
    # DMEAN outlier identification
    
    # get distance function
    if metric == 'euclidean':
        dist_fcn = misc.msedistance
    elif metric == 'cosine':
        dist_fcn = misc.cosdistance
    else:
        raise ValueError, "Distance %s not implemented." % metric
    
    lsegs = len(data)
    
    # distance to mean wave
    mean_wave = np.mean(data, 0)
    dists = misc.wavedistance(mean_wave, data, dist_fcn)
    
    # distance threshold
    th = np.mean(dists) + alpha * np.std(dists, ddof=1)
    
    # median of max and min
    M = np.median(np.max(data, 1))*1.5
    m = np.median(np.min(data, 1))*1.5
    
    # search for outliers
    outliers = []
    for i in xrange(lsegs):
        R = np.argmax(data[i])
        if R != R_Position:
            outliers.append(i)
        elif data[i][R] > M:
            outliers.append(i)
        elif np.min(data[i]) < m:
            outliers.append(i)
        elif dists[i] > th:
            outliers.append(i)
    
    normal = list(set(range(lsegs)) - set(outliers))
    
    return {'0': normal, '-1': outliers}


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

    Configurable fields:{}

    References:
        .. [1]
        
    """
    
    # get distance function
    if metric == 'euclidean':
        dist_fcn = misc.msedistance
    elif metric == 'cosine':
        dist_fcn = misc.cosdistance
    else:
        raise ValueError, "Distance %s not implemented." % metric
    
    lsegs = len(data)
    
    # distance to mean wave
    mean_wave = np.mean(data, 0)
    dists = misc.wavedistance(mean_wave, data, dist_fcn)
    
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


def dmean_tst(data=None, R_Position=200, alpha=0.5, beta=1.5, metric="euclidean", checkR=True, absolute=False):
    # test
    
    if absolute:
        data = np.abs(data)
    
    return dmean(data=data, R_Position=R_Position, alpha=alpha, beta=beta, metric=metric, checkR=checkR)


def run(data, tasks, parameters, outPath):
    """
    Run the outlier detection methods for each item in data, according to the given parameters.

    Input:
        data
        
        tasks
        
        parameters
        
        outPath
                                                  
    Output:
        output

    Configurable fields:{"name": "??.??", "config": {}, "inputs": [], "outputs": []}

    See Also:

    Notes:

    Example:
       
    """     
    
    if parameters['method'] == 'eleazar':
        method = eleazar_eskin
    elif parameters['method'] == 'dbscan':
        method = outliers_dbscan
    elif parameters['method'] == 'fiducialMedianFilter':
        method = fiducialMedianFilter
    elif parameters['method'] == 'meanfilter':
        method = meanfilter
    else:
        raise TypeError, "Method %s not implemented."%parameters['method']
    
    # create work queue
    workQueue = Queue()
    parameters.pop('method')
    
    # fill queue
    for recid in data.keys():
        if recid == 'info': continue
        for i in xrange(len(data[recid])):
            # maybe pass only the item (has signal/event + metadata)
            workQueue.put({'function': method,
                           'data': data[recid][i]['signal'], 
                           'parameters': parameters,
                           'taskid': tasks[recid][i]})
    
    # run in multiprocess
    parallel.runMultiprocess(workQueue, outPath)
    
    # load from temp files
    output = {}
    for recid in tasks.keys():
        nTasks = len(tasks[recid])
        output[recid] = range(nTasks)
        for i in xrange(nTasks):
            taskid = tasks[recid][i]
            try:
                res = datamanager.skStore(os.path.join(outPath, 'output-%d' % taskid))
            except IOError:
                raise IOError, "The work queue was not totally processed. File %s missing" % os.path.join(outPath, 'output-%d' % taskid)
            output[recid][i] = res
        
    return output


if __name__=='__main__':
    
    #===========================================================================
    # ECG Segments Outlier detection using the Eleazar & Eskin's algorithm
    print "Testing ECG Segments Outlier detection using the Eleazar & Eskin's algorithm.\n"
    # Get ECG Segments
    from database import mongoH5
    config = {'dbName': 'CruzVermelhaPortuguesa',
              'host': '193.136.222.234',
              'port': 27017,
              'path': r'\\193.136.222.220\cybh\data\CVP\hdf5'}
    db = mongoH5.bioDB(**config)
    recs = mongoH5.records(**db)
    ids_recs = recs.getAll()['idList']
    data = scipy.array(recs.getData(ids_recs[10], 'ECG/hand/Zee5to20', 0)['signal'][:,:,0]).T
    # Eleazar & Eskin's algorithm
    res = eleazar_eskin(data, metric='euclidean', threshold=.90)
    
    # Plot Results
    title='ECG Segments Outlier Detection. Outliers are in red.'
    figname = '../temp/ecg_segs_eleazar_eskin_outlier_detection_fig1.png'
    figProps={'title':title,'figname':figname}
    plotECG(data,res,figProps)
        
    print "Done. Results saved in %s"%figname
    #===========================================================================
    
    #===========================================================================
    # ECG Segments Outlier detection using DBSCAN algorithm
    print "Testing ECG Segments Outlier detection using DBSCAN.\n"
    res = outliers_dbscan(data, min_samples = 10, eps = 0.95, metric = "euclidean")
    
    # Plot Results
    title='ECG Segments Outlier Detection. Outliers are in red.'
    figname = '../temp/ecg_segs_dbscan_outlier_detection_fig1.png'
    figProps={'title':title,'figname':figname}
    plotECG(data,res,figProps)
    print "Done. Results saved in %s"%figname
    #===========================================================================
    
    #===========================================================================
    # ECG Segments Outlier detection using DBSCAN algorithm
    res = fiducialMedianFilter(data)
    #===========================================================================
    
    #===========================================================================
    # ECG Segments Outlier detection using DBSCAN algorithm
    res = meanfilter(data, th=0.8)
    #===========================================================================