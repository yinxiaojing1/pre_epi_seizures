"""
.. module:: cluster
   :platform: Unix, Windows
   :synopsis: This module provides various functions to cluster data

.. moduleauthor:: Filipe Canento, Andre Lourenco, Carlos Carreiras

.. conventions:
    output of type dict {'-1': indexes, '0': indexes, '1': indexes, ...}
"""

# Notes:
# input structure of type n observations x n features
# output of type dict {'-1': indexes, '0': indexes, '1': indexes, ...}
# old: output of type dict {'noise': indexes, 'cluster 0': indexes, 'cluster 1': indexes, ...}

# Imports
import os
import glob
from time import time
from itertools import izip, cycle
from math import floor

import scipy
import numpy
import pylab
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
from scipy.cluster import hierarchy
from scipy.io import loadmat
from scipy.cluster.vq import kmeans

import config
from Cloud import parallel
from datamanager import datamanager
from misc import misc # wavedistance, msedistance, cosdistance
from wavelets import wavelets

from clusteringCombination import life_time
import templateSelection as tpl # dend, mdist, centroids

#Hierarchical aglomerative clustering methods from scipy:
#http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html

#Alternative packages:
#http://math.stanford.edu/~muellner/fastcluster.html
#import fastcluster as fc



def selector(method):
    """
    Selector for the clustering functions and methods.
    
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
    
    if method == 'hierarchical':
        fcn = hierarchical
    elif method == 'leadHierarchical':
        fcn = leadHierarchical
    elif method == 'dbscan':
        fcn = dbscan
    elif method == 'fromFile':
        fcn = clustersFromFile
    else:
        raise TypeError, "Method %s not implemented." % method
    
    return fcn


def hierarchical(data, k=0, metric="euclidean", method='average', showDendrogram=True):
    """
    Clustering using hierarchical agglomerative methods

    Input:
        data : (array): input data array of shape (number samples x number features).
        k: (int) : number of clusters (k=0 using life-time criteria)
        method: (str) 'single'   --- nearest distance (default)
                      'complete' --- furthest distance
                      'average'  --- average distance
                      'centroid' --- center of mass distance
                      'ward'     --- inner squared distance

    Output:
        res (dict): output dict with indexes for each cluster determined.
                    Example: res = {"labels": list(labels),
                                    "k":k,
                                    "nsamples_in_cluster":list(nsamples_in_cluster),
                                    "algorithm":"hierarchical-"+method}

    Configurable fields:{"name": "cluster.dbscan", "config": {"min_samples": "10", "eps": "0.95", "metric": "euclidean"}, "inputs": ["data"], "outputs": ["core_samples", "labels"]}

    See Also:


    Example:


    References:
        .. [1]

    """


    # t0 = time()

    #scipy clustering lib:
    #http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    # see also:
    # http://docs.scipy.org/doc/scipy/reference/spatial.distance.html
    Y = distance.pdist(data)

    #Z=hierarchy.linkage(c,method)
    # make sure method is string, not unicode (see scipy ticket http://projects.scipy.org/scipy/ticket/1887)
    method = str(method)
    Z=hierarchy.linkage(Y,method)
    #fast clustering lib:
    #http://math.stanford.edu/~muellner/fastcluster.html
    #fc.linkage(c,method)


    if showDendrogram==1:
        hierarchy.dendrogram(Z)

    if k!=0:
        #http://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
        #Note that the function reference has an error. The correct definition of the t parameter is: The cut-off threshold for the cluster function or the maximum number of clusters criterion="maxclust".
        #http://stackoverflow.com/questions/9873840/cant-get-scipy-hierarchical-clustering-to-work
        labels=hierarchy.fcluster(Z,k,'maxclust')
        #http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html#scipy.cluster.hierarchy.fcluster
    else:
        #life-time criteria
        ns=data.shape[0]
        if ns < 3:
            labels = numpy.arange(ns,dtype='int')
        else:
            labels = life_time(Z,ns)
        #other criteria:
        #http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html#scipy.cluster.hierarchy.fcluster

    labels_unique = numpy.unique(labels)
    nsamples_in_cluster=numpy.zeros(len(labels_unique))

    res = {}

    clusters={}
    for j in range(len(labels_unique)):
        match = pylab.find(labels==labels_unique[j])
        nsamples_in_cluster[j]=len(match)
        lbl = '%d'%j
        clusters[lbl] = list(match)

    res["clusters"]=clusters
    #res= {"labels": list(labels), "k":k, "nsamples_in_cluster":list(nsamples_in_cluster) ,"algorithm":"hiearchical-"+method}

    config={}
    config["k"]=k
    config["nsamples_in_cluster"]=list(nsamples_in_cluster)
    config["algorithm"]= "hiearchical-" +method
    res["config"]=config

    # t_batch = time() - t0
    # print "done in %0.3fs" % (t_batch)

    return res


def leadHierarchical(data, **kwargs):
    """
    Perform hierarchical clustering for each lead independently.
    """
    
    out = {}
    for key in data.iterkeys():
        out[key] = hierarchical(data=data[key], **kwargs)
    
    return out


def transform2dissimilaritySpace(data, templates, distMeasure=None):
    """
    Transform to Dissimilarity Representation

    This function computes a new representation for the data, based on the distance to the templates.
    This representation has as number of dimensions the number of templates (columns)

    """

    n = len(data)
    #new dissimilarity representation
    newData = numpy.zeros((n, len(templates)))

    if distMeasure == None or distMeasure == 'Euclidean':
        distMeasure = 'Euclidean'
        fdist = misc.msedistance
    elif distMeasure == 'Cosine':
        fdist = misc.cosdistance

    for i,d in enumerate(data):
        for j,t in enumerate(templates):
            wDist = misc.wavedistance(d, t, fdist)
            newData[i,j] = wDist

    return newData

def templateSelection(data, clusters=None, method='MDIST', ntemplatesPerCluster=1, metric='euclidean'):
    """Template Selection

    This function implements several template selection methods. It may use the information of cluster labels
    [passed as argument] in conjunction with the criteria that are implemented. The number of templates per cluster
    can be controlled by ntemplatesPerCluster argument. The number returned templates is variable and depends on the
    number of clusters and on the number of templates per clusters [If clusters are singletons the number of templates
    per cluster may be violated]. It also works with different distance Measures [distMeasure argument]
    Currently the implemented criteria are:
    1) Centroids: requires that clusters are provided; if number of templates per cluster is 1, then templates are just
    the centroids; if the number of clusters is > 1 then k-means is used to determine more templates on each cluster.
    2) MDIST: criterion described in  U. Uludag, A. Ross, A. Jain, Pattern Recognition 37 (2004). Originally it does not
    require any clustering labels, computing the templates based on distances to the mean wave. The number of templates
    supplied by the used, in this cases we assume ntemplatesPerCluster argument, as the number of templates. We extended
    the idea, including cluster label information. We compute for each cluster ntemplatesPerCluster using the distance to
    mean wave criterion.
    3) DEND: criterion described in  U. Uludag, A. Ross, A. Jain, Pattern Recognition 37 (2004). The implemented version
    is an extension that works with any clustering algorithm (uses the clusters labels passed by argument). The original
    version used Complete-linkage. It computes medoids on each cluster, based on the average pairwise distance to other
    points in the cluster. Templates are the points whose distance is minimum; we allow to chose more than 1 template,
    selecting ntemplatesPerCluster using the same criterion

    Input:
        data : (array): input data array of shape (number samples x number features).
        clusters: (dict): dictionary with labels (can be None)
        method: (string): method to be used ('Centroids', 'MDIST', 'DEND'/'Medoid')
        ntemplatesPerCluster: (int): number of templates on each cluster
        distMeasure: (string): distance measure to be used ('Euclidean' or 'Cosine')
    Output:
        res (dict): output dict with indexes of prototypes for each cluster .
                    Example: res = {'templates': list(),
                                    'algorithm': method}

    Configurable fields:

    See Also:


    Example:


    References: [1] U. Uludag, A. Ross, A. Jain, Pattern Recognition 37 (2004)

    """

    # distance measure
    if metric == 'euclidean':
        fdist = 'Euclidean'
    elif metric == 'cosine':
        fdist = 'Cosine'
    else:
        raise TypeError, "Distance %s not implemented." % metric

    if method == 'Centroids':
        # Naive method - k centroid for each cluster (when using more than one centroid/prototype uses kmeans)
        res = tpl.centroids(data, clusters, ntemplatesPerCluster, fdist)
    elif method == 'MDIST':
        # MDIST in: U. Uludag, A. Ross, A. Jain, Pattern Recognition 37 (2004)
        res = tpl.mdist(data, clusters, ntemplatesPerCluster, fdist)
    else:
        raise TypeError, "Method %s not implemented." % method

    return res

def templateSelectionFixingK(data, clusters=None, method='MDIST', ntemplates=10, distMeasure=None):

    """
    Template Selection Fixing the number of templates (argument ntemplates)

    """

    n = len(data)

    if clusters==None:
        ks=[]
        n_clusters_=0
    else:
        ks = clusters["clusters"].keys()
        n_clusters_ = len(ks) - (1 if '-1' in ks else 0)

    #distribute the templates by the clusters (idea put more on clusters with more objects)
    if n_clusters_ != ntemplates:
        nsamples_in_clusters = numpy.zeros(n_clusters_)
        for i,k in enumerate(ks):    #for each cluster, carefull because index can be different (outlier case)
            if k != '-1': #outlier
                nsamples_in_clusters[i]= len(clusters["clusters"][k])
        print "nsamples_in_clusters"
        print nsamples_in_clusters
        #I=argsort(nsamples_in_clusters)
        #reverse for ascending order (in terms of number of elements per cluster)
        #I=I[::-1]
        fractions=nsamples_in_clusters*1./sum(nsamples_in_clusters)
        templatesPerClusters=sum(numpy.round(fractions*ntemplates))
    else:
        templatesPerClusters = numpy.ones(n_clusters_)


    if distMeasure == None or distMeasure == 'Euclidean':
        distMeasure='Euclidean'
        fdist = misc.msedistance
    elif distMeasure == 'Cosine':
        fdist = misc.cosdistance        #defined on misc

    #Naive method - k centroid for each cluster (when using more than one centroid/prototype uses kmeans)
    if method == 'Centroids':
        res = tpl.centroids(data, clusters, templatesPerClusters, distMeasure)


    #MDIST in: U. Uludag, A. Ross, A. Jain, Pattern Recognition 37 (2004)
    elif method == 'MDIST':
        res = tpl.mdist(data, clusters, templatesPerClusters, distMeasure)

    # DEND related on Uludag and Jain/ Basically this are medoids
    elif method == 'DEND' or method == 'Medoid':
        res = tpl.centroids(data, clusters, templatesPerClusters, distMeasure)

    else:
        print "template method method not recognized"
        res= []

    return res


# Wrapper to sklearn.cluster.DBSCAN
def dbscan(data, min_samples=10, eps=0.95, metric="euclidean"):
    """

    Perform clustering using the DBSCAN method. Uses: sklearn.cluster.DBSCAN.

    Input:
        data (array): input data array of shape (number samples x number features).

        min_samples (int): minimum number of samples in a cluster.

        eps (float): maximum distance between two samples in the same cluster.

        metric (string): distance metric

    Output:
        res (dict): output dict with indexes for each cluster determined.
                    Example: res = {'clusters': {{'-1': noise indexes list,
                                                  '0': cluster 0 indexes list,
                                                  '1': cluster 1 indexes list}}}

    Configurable fields:{"name": "cluster.dbscan", "config": {"min_samples": "10", "eps": "0.95", "metric": "euclidean"}, "inputs": ["data"], "outputs": ["core_samples", "labels"]}

    See Also:
        sklearn.cluster.DBSCAN
        scipy.spatial.distance

    Notes:
        conversion to similarity uses the same metric distance as dbscan

    Example:


    References:
        .. [1]    DBSCAN: Ester, M., Kriegel, H.-P., Sander,J. and Xu,X. (1996), "A density-based algorithm for discovering clusters in large spatial databases with noise". Proc of 2nd Int Conf on Knowledge Discovery and Data Mining (KDD-96)
        .. [2]    dbscan: http://scikit-learn.org/dev/modules/generated/sklearn.cluster.DBSCAN.html
        .. [3]    pdist: http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist

    """
    # Compute similarities
    if metric == 'wavelet':
        metric = 'euclidean'
        nb = data.shape[0]
        D = numpy.zeros((nb, nb))
        for i in xrange(nb):
            for j in xrange(i+1, nb):
                D[i, j] = D[j, i] = wavelets.waveDist(data[i, :, :], data[j, :, :])
    else:
        #metric = 'euclidean'|'seuclidean'|'cosine',...
        D = distance.squareform(distance.pdist(data, metric))
    S = 1 - (D / numpy.max(D))
    # Compute DBSCAN
    db = DBSCAN(eps=eps, metric=metric, min_samples=min_samples)
#    db = DBSCAN(eps=eps, metric='precomputed', min_samples=min_samples) # http://scikit-learn.org/dev/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN.fit
    db.fit(S)
#    db.fit(D)
    # core_samples = db.core_sample_indices_
    labels = db.labels_
    # Output
    res = {'clusters': {}}
    for c in set(labels):
        #if c == -1: lbl = 'noise'
        #else: lbl = 'cluster %d'%c
        if c == -1: lbl = '-1'
        else: lbl = '%d'%c
        res['clusters'][lbl] = list(pylab.find(labels==c))

    return res

def run(data, tasks, parameters, outPath):
    """
    Run the clustering methods for each item in data, according to the given parameters, then perform the prototype selection.

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

    if parameters['method'] == 'hierarchical':
        method = hierarchical
    elif parameters['method'] == 'dbscan':
        method = dbscan
    elif parameters['method'] == 'fromFile':
        method = False
    else:
        raise TypeError, "Method %s not implemented." % parameters['method']

    clusterPath = os.path.join(outPath, 'clusters')
    # perform clustering
    clusterQueue = parallel.getQueue()
    lbl = parameters.pop('method')
    if lbl != 'fromFile':
        # fill queue
        for recid in data.keys():
            if recid == 'info': continue
            for i in xrange(len(data[recid])):
                # maybe pass only the item (has signal/event + metadata)
                clusterQueue.put({'function': method,
                               'data': data[recid][i]['signal'],
                               'parameters': parameters,
                               'taskid': tasks[recid][i]})

        # run in multiprocess
        parallel.runMultiprocess(clusterQueue, clusterPath)

    # load from temp files and create templates
    ### maybe parallelize???
    output = {}
    prototypePath = os.path.join(outPath, 'templates')
    for recid in tasks.keys():
        nTasks = len(tasks[recid])
        output[recid] = range(nTasks)
        for i in xrange(nTasks):
            taskid = tasks[recid][i]
            try:
                clusters = datamanager.skLoad(os.path.join(clusterPath, 'output-%d' % taskid))
            except IOError:
                raise IOError, "The work queue was not totally processed. File %s missing" % os.path.join(outPath, 'output-%d' % taskid)
            # create new templates
            print recid, taskid
            res = templateSelection(data[recid][i]['signal'], clusters)
            # save to tmp file
            datamanager.skStore(os.path.join(prototypePath, 'output-%d' % taskid), res)
            output[recid][i] = res

    return output


def convertMatFiles(srcPath, dstPath, method, criterion, nsegs=None):
    files = glob.glob(os.path.join(srcPath, 'rec_*.mat'))

    # sort files
    files.sort(key=lambda item: int(item.split('_')[-1].split('.')[0]))

    i = 0
    for fname in files:

        fileNb = fname.split('.')[0].split('_')[-1]

        # load from mat file
        mat = loadmat(fname)
        if method in ['WL','CL','AL','SL']:
            if criterion in ['lifeTime']:
                k = mat[method][0][0][0][0][0]
                labels = mat[method][0][0][1].flatten()
            else:
                print "Criterion %s not available in file %s." % (criterion, fname)
                continue
        elif method in ['CLDID','ALDID','SLDID', 'WLDID']:
            if criterion in ['lifeTime']:
                k = mat[method][0][0][2][0][0]
                labels = mat[method][0][0][3].flatten()
            elif criterion in ['MDL']:
                k = mat[method][0][0][0][0][0]
                labels = mat[method][0][0][1].flatten()
            else:
                print "Criterion %s not available in file %s." % (criterion, fname)
                continue
        else:
                print "Method %s not available in file %s." % (method, fname)
                continue

        # convert to cluster structure
        clusters = {'clusters': {}, 'config': {'k': k, 'method': method, 'criterion': criterion}}
        labelSet = set(labels)
        if nsegs is not None:
            labels = labels[:nsegs[int(fileNb)]]
        for lbl in labelSet:
            clusters['clusters'][str(lbl - 1)] = list(pylab.find(labels == lbl)) # -1 because matlab starts at 1

        # save to temp file
        datamanager.skStore(os.path.join(dstPath, 'output-%s' % fileNb), clusters)
        # datamanager.gzStore(os.path.join(dstPath, 'output-%d' % i), clusters)
        i += 1


def clustersFromFile(filePath, method, criterion, nsegs=None):
    
    pass



if __name__=='__main__':

    #===========================================================================
    # Generate Data
    data_set1 = scipy.random.normal(10, 2.5, [200, 2])
    data_set2 = scipy.random.normal(30, 2.5, [200, 2])
    noise = scipy.array([[-5, -5], [-5, -15], [0, -9], [-9, 20], [25, -10]])
    data = []
    c = 0
    for ds in [data_set1, data_set2, noise]:
        for x,y in ds: data.append([x,y])
        c += 1
    data = scipy.array(data)
    #===========================================================================
    print "Testing DBSCAN.\n"
    # Plot data
    fig = pylab.figure(1)
    fig.suptitle('DBSCAN example.')
    ax = fig.add_subplot(211)
    ax.cla()
    ax.set_title('Original data')
    ax.plot(data[:,0], data[:,1], 'bo', label='data')
    ax.legend(loc='lower right')
    ax.grid()
    #===========================================================================
    # DBSCAN method
    res = dbscan(data)
    ks = res.keys()
    n_clusters_ = len(ks) - (1 if '-1' in ks else 0)
    # Plot result
    ax = fig.add_subplot(212)
    colors = cycle('bgcmykbgcmyk')
    for k, col in izip(ks, colors):
        if k == '-1': col = 'r'
        ax.plot(data[:,0][res[k]], data[:,1][res[k]], col+'o', label=k)
    ax.set_title('Estimated number of clusters: %d'%n_clusters_)
    ax.grid()
    ax.axis('tight')
    ax.legend(loc='lower right')
    figname = '../temp/dbscan_fig1.png'
    fig.savefig(figname)

    print "Done. Results saved in %s"%figname
    #===========================================================================
    # Hiearchical method
    reshiear = hierarchical(data, k=2)
    ks = reshiear["clusters"].keys()
    n_clusters_ = len(ks) - (1 if '-1' in ks else 0)
    print "k:" + n_clusters_
    #===========================================================================
    # Prototypes
    resProto=templateSelection(data, reshiear)
    print resProto["nprototypes_in_cluster"]
    for centroid in resProto["prototypes"]:
        print centroid

