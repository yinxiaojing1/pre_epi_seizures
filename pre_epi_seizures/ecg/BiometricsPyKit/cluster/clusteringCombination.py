#!/usr/bin/python

####################################
# Evidence Accumulation Clustering (EAC) 
# Script used for building the clustering ensembles its combination using EAC
#
# Andre Lourenco, arlourenco@lx.it.pt
# Ana Fred, afred@lx.it.pt
# Instituto de Telecomunicaces, Instituto Superior Tecnico,
# Technical University of Lisbon, Portugal
# IST - Torre Norte, Av. Rovisco Pais, 1049-001, Lisbon, Portugal
####################################

####################################
#history:
#0.1: 07/05/2012 - mini-kmeans ensemble generation
#                - ensemble generation including list to allow json dump
#0.0: April 2012
####################################
#TODO:
# - validation of life-time algorithm
# - sparse co-assocs implementation
####################################

import os
# import logging
# import sys
from time import time

import numpy as np
import scipy.sparse as sp
import pylab as pl
import scipy.spatial.distance
from pylab import find

#Using sklearn python package
#notes on installation: http://scikit-learn.sourceforge.net/dev/install.html
from sklearn.cluster import MiniBatchKMeans, KMeans
# from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets.samples_generator import make_blobs

#Hierarchical aglomerative clustering methods from scipy:
#http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
from scipy.cluster import hierarchy
#Alternative packages:
#http://math.stanford.edu/~muellner/fastcluster.html
#import fastcluster as fc

from Cloud import parallel
import config
from datamanager import datamanager



def plotClustering(X,labels,figNumber=2):
    """
    plot clustering
    
    Input:
        
    
    Output:
         
    """
    
    from itertools import cycle
    
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    
    pl.figure(figNumber)
    pl.clf()

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == labels_unique[k]
        #cluster_center = cluster_centers[k]
        pl.plot(X[my_members, 0], X[my_members, 1], col + '.')
        #pl.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
        #                            markeredgecolor='k', markersize=14)
    pl.title('Estimated number of clusters: %d' % n_clusters_)
    pl.show()


def ensembleCreation(data, nensemble=100, showHist=False, fun="KMeans", kmin=None, kmax=None):
#out:
# ensemble --- clustering ensemble
    """
    creation of the ensemble using kmeans
    Input:
        data (array): input data array of shape (number samples x number features).
        
        N_ensemble (int):  number of partitions on the ensemble (if empty using 100) 
        
        eps (float): maximum distance between two samples in the same cluster.
        
        fun (string): algorithm for producing clusters (default KMeans)
    
    Output:
        res (dict): output dict with partitions 
    
    Configurable fields:
    
    See Also:
    
    Example:
    
    
    References:
        .. [1]    
        
    """

    ns= data.shape[0]
    if kmin is None:
        kmin= round(np.sqrt(ns)/2.)
    if kmax is None:
        kmax= round(np.sqrt(ns))
    print "ns:", ns, " kmin:", kmin, " kmax:", kmax
    ks=config.random.random_integers(low=kmin, high=kmax, size=nensemble)
    if showHist:
        #count, bins, ignored = plt.hist(ks, 11, normed=True)
        count, bins= np.histogram(ks, 11, density=True)
        pl.bar(bins,count)
        pl.show()
    
    print "Using:",fun
    ensemble={}
    # Compute clustering with K-Means or MiniBatchKmeans
    t0 = time()
    for i in range(nensemble):
        print '.',
#        k=ks[i]
#        if fun == "KMeans":
#            k_means = KMeans(k=k, init='random', max_iter=100)
#                #k_means = KMeans(init='k-means++', k=3, n_init=10)
#        elif fun=="MiniBatchKMeans":
#            k_means = MiniBatchKMeans(k=k,max_iter=100,batch_size=100,init='random')
#        else: 
#            print "error..."
#            return ensemble
#
#        k_means.fit(data)
#        labels = k_means.labels_
#        k_means_cluster_centers = k_means.cluster_centers_
#        labels_unique = np.unique(labels)
#        nsamples_in_cluster=np.zeros(len(labels_unique))
#        for j in range(len(labels_unique)):
#            nsamples_in_cluster[j]=len(pl.find(labels==labels_unique[j]))
#            
#        ensemble[i] = {"labels": list(labels), "k":k, "nsamples_in_cluster":list(nsamples_in_cluster) ,"algorithm":fun}
        
        ensemble[i] = ensembleInstance(data, fun='KMeans', k=ks[1], init='random', max_iter=100, batch_size=100)
    
    t_batch = time() - t0
    print "done in %0.3fs" % (t_batch)
    
    return ensemble


def ensembleCreationParallel(data, nensemble=100, fun="KMeans", kmin=None, kmax=None, dstPath=None):
    
    ns = data.shape[0]
    if kmin is None:
        kmin = round(np.sqrt(ns)/2.)
    if kmax is None:
        kmax = round(np.sqrt(ns))
    ks = config.random.random_integers(low=kmin, high=kmax, size=nensemble)
    
    # choose store
    if dstPath is None:
        store = parallel.getDictManager()
    else:
        store = dstPath
    
    # populate work queue
    workQueue = parallel.getQueue()
    for i in xrange(nensemble):
        workQueue.put({'function': ensembleInstance,
                       'data': data,
                       'parameters': {'fun': 'KMeans',
                                      'k': ks[1],
                                      'init': 'random',
                                      'max_iter': 100,
                                      'batch_size': 100
                                      },
                       'taskid': i
                       })
    
    # run in multiprocess
    parallel.runMultiprocess(workQueue, store)
    
    # load from store
    ensemble = {}
    for i in xrange(nensemble):
        ensemble[i] = parallel.loadStore(store, i)
    
    # clean up store
    parallel.cleanStore(store)
    
    return ensemble


def ensembleInstance(data, fun='KMeans', k=1, init='random', max_iter=100, batch_size=100):
    # one instance of clustering for the ensemble
    
    if fun == "KMeans":
        k_means = KMeans(n_clusters=k, init='random', max_iter=100)
        # k_means = KMeans(k=k, init='random', max_iter=100)
    elif fun=="MiniBatchKMeans":
        k_means = MiniBatchKMeans(n_clusters=k, max_iter=100, batch_size=100, init='random')
    else:
        raise TypeError, "Method %s not implemented." % fun
    
    k_means.fit(data)
    labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    labels_unique = np.unique(labels)
    nsamples_in_cluster = np.zeros(len(labels_unique))
    for j in range(len(labels_unique)):
        nsamples_in_cluster[j] = len(pl.find(labels == labels_unique[j]))
    
    res = {'labels': list(labels),
           'k': k,
           'nsamples_in_cluster': list(nsamples_in_cluster),
           'algorithm': fun,
           'cluster_centers': k_means_cluster_centers}
    
    return res


def runEnsembleCreation(data, tasks, parameters):
    # run ensemble creation in parallel

    outPath = os.path.join(config.folder, 'ensembles')
    workQueue = parallel.getQueue()
    
    # populate work queue
    for recid in data.keys():
        for i in xrange(len(data[recid])):
            workQueue.put({'function': ensembleCreationParallel,
                              'data': data[recid][i],
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
                ensemble = datamanager.skLoad(os.path.join(outPath, 'output-%d' % taskid))
            except IOError:
                raise IOError, "The work queue was not totally processed. File %s missing" % os.path.join(outPath, 'output-%d' % taskid)
            else:
                output[recid][i] = ensemble

    return output
    

def coassociationCreation(ensemble, ns, debug=False):
    """
    Construction of the co-association matrix
    
    Input:
        ensemble (dic): 
        
        ns (int): number of samples  
        
    Output:
        assocs : co-association matrix
        
    
    Configurable fields:{"name": "cluster.dbscan", "config": {"min_samples": "10", "eps": "0.95", "metric": "euclidean"}, "inputs": ["data"], "outputs": ["core_samples", "labels"]}
    
    See Also:
        
    Example:
    
    
    References:
        .. [1]    
        
    """
    
    t0 = time()
    n_clusterings = len(ensemble)
    for i in xrange(n_clusterings):
        if debug:
            print "CE:",i
        nsamples_in_cluster = np.array(ensemble[i]["nsamples_in_cluster"])
        labels = np.array(ensemble[i]["labels"])
        dim = sum(nsamples_in_cluster * (nsamples_in_cluster - 1)) / 2
        I = np.zeros(dim)
        J = np.zeros(dim)
        X = np.ones(dim)
        ntriplets = 0
        if debug:
            print "nclusters:", len(nsamples_in_cluster)
        for j in xrange(len(nsamples_in_cluster)):
            v = pl.find(labels == j)
            #print v
            if len(v) > 0:
                for h in xrange(len(v)):
                    for f in xrange(h+1, len(v)):
                        I[ntriplets] = v[h]
                        J[ntriplets] = v[f]
                        ntriplets = ntriplets + 1
        #print "len(I)",len(I)
        #print "len(J)",len(J)
        #print "ntriplets",ntriplets," dim:",dim
        assocs_aux = sp.csc_matrix( (X,(I,J)), shape=(ns,ns) )#.todense()
        #lil_matrix more efficient?
        #assocs_aux=sp.lil_matrix( (X,(I,J)), shape=(ns,ns) )#.todense()
        #pl.matshow(assocs_aux.todense())
        if i == 0:
            assocs = assocs_aux
            
        else:
            assocs = assocs + assocs_aux
    
    if debug:
        t_batch = time() - t0
        print "done in %0.3fs" % (t_batch)
    
    a = assocs + assocs.T
    a.setdiag(n_clusterings * np.ones(ns))
    
    return a.todense()


def consensusExtraction(coassocs, k=0, method='single', showDendrogram=False):
    """
    Extraction of the consensus partition using hierarchical agglomerative methods
    
    Input:
        assos : co-association matrix
        method='single'   --- nearest distance (default)
               'complete' --- furthest distance
               'average'  --- average distance
               'centroid' --- center of mass distance
                'ward'     --- inner squared distance
         k : number of clusters (k=0 using life-time criteria)
    
    Output:
        res (dict): output dict with indexes for each cluster determined.
                    Example: res = {"labels": list(labels),
                                    "k":k, 
                                    "nsamples_in_cluster":list(nsamples_in_cluster),
                                    "algorithm":"EAC-"+method}
    
    Configurable fields:{"name": "cluster.dbscan", "config": {"min_samples": "10", "eps": "0.95", "metric": "euclidean"}, "inputs": ["data"], "outputs": ["core_samples", "labels"]}
    
    See Also:
    
    
    Example:
    
    
    References:
        .. [1]    
        
    """
    
    # t0 = time()
    t = np.max(coassocs)
    c = t - coassocs
    #fast clustering lib:
    #http://math.stanford.edu/~muellner/fastcluster.html
    #fc.linkage(c,method)
    
    #scipy clustering lib:
    #http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    # see also:
    # http://docs.scipy.org/doc/scipy/reference/spatial.distance.html
    
    condensed = scipy.spatial.distance.squareform(c) #(converts full matrix to condensed or redundant distance matrix)

    #Z=hierarchy.linkage(c,method)
    if method in ['single', 'complete', 'weighted', 'average']:
        Z = hierarchy.linkage(condensed, method)
    else:
        Z = hierarchy.linkage(coassocs, method)
    
    if showDendrogram:
        hierarchy.dendrogram(Z)
    
    if k != 0:
        #http://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
        #Note that the function reference has an error. The correct definition of the t parameter is: The cut-off threshold for the cluster function or the maximum number of clusters criterion="maxclust". 
        #http://stackoverflow.com/questions/9873840/cant-get-scipy-hierarchical-clustering-to-work
        labels = hierarchy.fcluster(Z, k, 'maxclust')
    else:
        #life-time criteria
        ns = coassocs.shape[0]
        labels = life_time(Z, ns)
            
    labels_unique = np.unique(labels)
    nsamples_in_cluster = np.zeros(len(labels_unique))
    clusters = {}
    for j in xrange(len(labels_unique)):
        aux = find(labels == labels_unique[j])
        nsamples_in_cluster[j] = len(aux)
        clusters[str(labels_unique[j])] = aux
    #res= {"labels": labels, "k":k, "nsamples_in_cluster":nsamples_in_cluster ,"algorithm":"EAC-"+method}
    res= {"clusters": clusters,
          "config": {"k": k,
                     "nsamples_in_cluster": list(nsamples_in_cluster),
                     "algorithm":"EAC-" + method,
                     },
          }
    
    # t_batch = time() - t0
    # print "done in %0.3fs" % (t_batch)
    
    return res

def life_time(Zpy, ns):
    """
    life-time criterion for automatic selection of the number of clusters
    [porting from life-time implementation on matlab]
    
    Input:
        Zpy (array): input data array of shape (number samples x number features).
        
        ns (int): number of samples. 
        
    
    Output:
        res (dict): output dict with indexes for each cluster determined. 
                    Example: res = {'-1': noise indexes list,
                                    '0': cluster 0 indexes list,
                                    '1': cluster 1 indexes list}
    
    Configurable fields:{"name": "cluster.dbscan", "config": {"min_samples": "10", "eps": "0.95", "metric": "euclidean"}, "inputs": ["data"], "outputs": ["core_samples", "labels"]}
    
    See Also:
        
    Example:
    
    
    References:
        .. [1]    
        
    """

    
    Z = hierarchy.to_mlab_linkage(Zpy)
    #dif=Z[1:,2]-Z[0:-1,2]
    dif = np.diff(Z[:, 2])
    indice = np.argmax(dif)
    maximo = dif[indice]
    
    indice = Z[find(Z[:, 2] > Z[indice, 2]), 2]
    if indice == []:
        cont = 1
    else:
        cont = len(indice) + 1
        
    # th = maximo
    
    #testing the situation when only 1 cluster is present
    #max>2*min_interval -> nc=1
    minimo = np.min(dif[pl.find(dif != 0)])
    if minimo != maximo: #se maximo=minimo e' porque temos um matriz de assocs perfeita de 0s e 1s
        if maximo < 2 * minimo:
            cont = 1
    
    nc_stable = cont
    if nc_stable > 1:
        labels = hierarchy.fcluster(hierarchy.from_mlab_linkage(Z), nc_stable, 'maxclust')
        
    else: #ns_stable=1
        labels = np.arange(ns, dtype="int")
    
    return labels


def relativeSimilarities(sim, clusters, n):
    """
    sim - similarity matrix
    clusters - clusters
    n - number of samples
    """
    
    labels = clusters['clusters'].keys()
    k = len(labels)
    nsamples_in_cluster = {}
    for cl in labels:
        nsamples_in_cluster[cl] = len(clusters['clusters'][cl])
    
    ##### silhouette method
    
    # intracluster
    a = np.zeros(n)
    # intercluster
    b = np.zeros((k, n))
    s = np.zeros(n)
    
    for i in xrange(k):
        for j in xrange(nsamples_in_cluster[labels[i]]):
            if nsamples_in_cluster[labels[i]] == 1:
                # singleton
                a[clusters['clusters'][labels[i]][j]] = 1
                b[i, :] = 1
                break
            else:
                a[clusters['clusters'][labels[i]][j]] = np.sum(sim[clusters['clusters'][labels[i]][j], clusters['clusters'][labels[i]]]) / (nsamples_in_cluster[labels[i]] - 1)
                
                for l in xrange(k):
                    if l != i:
                        b[l, clusters['clusters'][labels[i]][j]] = np.sum(sim[clusters['clusters'][labels[i]][j], clusters['clusters'][labels[l]]]) / (n - nsamples_in_cluster[labels[i]] - 1)
    
    aux = np.zeros(n)
    if k == 1:
        aux[:] = 0
    else:
        for i in xrange(k):
            for j in xrange(nsamples_in_cluster[labels[i]]):
                sel = range(k)
                sel.remove(i)
                aux[clusters['clusters'][labels[i]][j]] = np.max(b[sel, clusters['clusters'][labels[i]][j]])
    
    b = aux
    s = (a - b) / np.maximum(a, b)
    
    S = np.zeros(k)
    for i in xrange(k):
        S[i] = np.sum(s[clusters['clusters'][labels[i]]]) / nsamples_in_cluster[labels[i]]
    
    GS = np.sum(S) / k
    
    
    ##### Dunn's index
    
    # intercluster
    intercluster = np.zeros((k, k), dtype='float')
    s_max = -np.inf
    for i in xrange(k):
        for m in xrange(i+1, k):
            for j in xrange(nsamples_in_cluster[labels[i]]):
                for l in xrange(nsamples_in_cluster[labels[m]]):
                    if sim[clusters['clusters'][labels[i]][j], clusters['clusters'][labels[m]][l]] > s_max:
                        s_max = sim[clusters['clusters'][labels[i]][j], clusters['clusters'][labels[m]][l]]
#                    if l == 0:
#                        s_max = sim[clusters['clusters'][labels[i]][j], clusters['clusters'][labels[m]][l]]
#                    else:
#                        haha = sim[clusters['clusters'][labels[i]][j], clusters['clusters'][labels[m]][l]]
#                        if sim[clusters['clusters'][labels[i]][j], clusters['clusters'][labels[m]][l]] > s_max:
#                            s_max = sim[clusters['clusters'][labels[i]][j], clusters['clusters'][labels[m]][l]]
            intercluster[i, m] = s_max
            intercluster[m, i] = s_max
    
    # intracluster
    intracluster = np.zeros(k)
    for i in xrange(k):
        if nsamples_in_cluster[labels[i]] == 1:
            intracluster[i] = np.inf
        else:
            for j in xrange(nsamples_in_cluster[labels[i]]):
                sel = range(nsamples_in_cluster[labels[i]])
                sel.remove(j)
                intracluster[i] = np.min(sim[clusters['clusters'][labels[i]][j], clusters['clusters'][labels[i]][sel]])
    
    IntraCluster = np.min(intracluster)
    interaux = intercluster.copy()
    interaux[intercluster == 0] = np.inf
    D = np.max(np.max(interaux**(-1) * IntraCluster))
    
    
    ##### Davies-Bouldin index
    # print intercluster, intracluster
    DB = np.zeros((k, k))
    for i in xrange(k):
        for m in xrange(i+1, k):
            if i != m:
                DB[i, m] = intercluster[i, m] / (intracluster[i] + intracluster[m])
    
    DB = np.sum(np.max(DB, axis=1)) / k
    
    return GS, D, DB



##############################################################################
def main():
    #Unit Testing...
    pl.close('all')
    # Generate sample data
    np.random.seed(0)
    batch_size = 45
    centers = [[1, 1], [-1, -1], [1, -1]]
    n_clusters = len(centers)
    #X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)
    X, labels_true = make_blobs(n_samples=1000, centers=centers, cluster_std=0.7)

    #Generate Ensemble
    nensemble=100
    print "*"*20
    print "Ensemble Creation- ns:", X.shape[0]
    print "n_ensemble:",nensemble
    ensemble=ensembleCreation(X,nensemble)
    plotClustering(X,ensemble[0]["labels"])

    #Coassociation Generation
    print "*"*20
    print "Coassocs Generation"
    ns= X.shape[0]
    coassocs=coassociationCreation(ensemble,ns)
    pl.matshow(coassocs)
    pl.colorbar()
    #pl.hot()

    #Consensus extraction: examples using different extraction methods
    print "*"*20
    print "Coassocs Extraction: Single"
    partition=consensusExtraction(coassocs=coassocs,k=3,method='single')
    plotClustering(X=X,labels=partition["labels"],figNumber=4)

    print "*"*20
    print "Coassocs Extraction: Average"
    partition=consensusExtraction(coassocs=coassocs,k=3,method='average')
    plotClustering(X=X,labels=partition["labels"],figNumber=5)

    print "*"*20
    print "Coassocs Extraction: weighted"
    partition=consensusExtraction(coassocs=coassocs,k=3,method='weighted')
    plotClustering(X=X,labels=partition["labels"],figNumber=6)

    print "*"*20
    print "Coassocs Extraction: complete"
    partition=consensusExtraction(coassocs=coassocs,k=3,method='complete')
    plotClustering(X=X,labels=partition["labels"],figNumber=7)

    print "*"*20
    print "Coassocs Extraction: life-time"
    partition=consensusExtraction(coassocs=coassocs,k=0)
    plotClustering(X=X,labels=partition["labels"],figNumber=8)

    #print "*"*20
    #print "Coassocs Extraction: Ward": 
    #ValueError: Valid methods when the raw observations are omitted are 'single', 'complete', 'weighted', and 'average'.
    #partition=consensusExtraction(coassocs=coassocs,k=3,method='ward')
    #plotClustering(X=X,labels=partition["labels"],figNumber=6)
    
#if __name__=='__main__': main()