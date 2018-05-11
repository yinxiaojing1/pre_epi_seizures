"""
.. module:: templateSelection
   :platform: Unix, Windows
   :synopsis: This module provides various functions to cluster data

.. moduleauthor:: Andre Lourenco, Carlos Carreiras

.. conventions: 
    output of type dict {'-1': indexes, '0': indexes, '1': indexes, ...}
"""

# Notes:
# input structure of type n observations x n features
# output of type dict {'-1': indexes, '0': indexes, '1': indexes, ...}

# Imports
# 3rd party
from scipy.spatial import distance
from scipy.cluster.vq import kmeans2
import numpy

# BiometricsPyKit
import config



def selector(method):
    """
    Selector for the template selection functions and methods.
    
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
    
    if method == 'mdist':
        fcn = mdist
    elif method == 'centroids':
        fcn = centroids
    elif method == 'leadCentroids':
        fcn = leadCentroids
    else:
        raise TypeError, "Method %s not implemented." % method
    
    return fcn


def remainderAllocator(cardinals, k, reverse=True, check=False):
    # allocate k seats to cardinals
    # aka Hare-Niemeyer Method, with Hare quota (reverse=True)
    
    # check total number of cardinals
    if check:
        tot = numpy.sum(cardinals)
        if k > tot:
            k = tot
    
    # frequencies
    length = len(cardinals)
    freqs = numpy.array(cardinals, dtype='float') / numpy.sum(cardinals)
    
    # assign items
    aux = k * freqs
    out = aux.astype('int')
    
    # leftovers
    nb = k - out.sum()
    if nb > 0:
        if reverse:
            ind = numpy.argsort(freqs - out)[::-1]
        else:
            ind = numpy.argsort(freqs - out)
        for i in xrange(nb):
            out[ind[i % length]] += 1
    
    return out


def highestAveragesAllocator(cardinals, k, divisor='dHondt', check=False):
    # allocate k seats to cardinals, based on Highest Averages Method
    
    # check total number of cardinals
    if check:
        tot = numpy.sum(cardinals)
        if k > tot:
            k = tot
    
    # select divisor
    if divisor == 'dHondt':
        fcn = lambda i: float(i)
    elif divisor == 'Huntington-Hill':
        fcn = lambda i: numpy.sqrt(i * (i + 1.))
    elif divisor == 'Sainte-Lague':
        fcn = lambda i: i - 0.5
    elif divisor == 'Imperiali':
        fcn = lambda i: float(i + 1)
    elif divisor == 'Danish':
        fcn = lambda i: 3. * (i - 1.) + 1.
    else:
        raise ValueError, "Unknown divisor method."
    
    # compute coefficients
    tab = []
    length = len(cardinals)
    D = [fcn(i) for i in xrange(1, k + 1)]
    for i in xrange(length):
        for j in xrange(k):
            tab.append((i, cardinals[i] / D[j]))
    
    # sort
    tab.sort(key=lambda item: item[1], reverse=True)
    tab = tab[:k]
    tab = numpy.array([item[0] for item in tab], dtype='int')
    
    out = numpy.zeros(length, dtype='int')
    for i in xrange(length):
        out[i] = numpy.sum(tab == i)
    
    return out


def avgDist(data=None, fdist='Euclidean', norm=None):
    
    # check inputs
    if data is None:
        raise TypeError, "Please specify the input data."
    if norm is None:
        norm = len(data)
    
    # compute distances
    Y = distance.pdist(data, fdist)
    sY = distance.squareform(Y)
    sumY = sum(sY, 0) / norm
    s = sumY.argsort()
    
    return sY, sumY, s


def mdist(data, clusters=None, ntemplates=1, distMeasure='Euclidean'):
    """
    
    Template Selection using MDIST criterion
    Proposed in  U. Uludag, A. Ross, A. Jain, Pattern Recognition 37 (2004).
    Originally it does not require any clustering labels, computing the templates
    based on mean distances. The number of templates supplied by the user,
    in this cases we assume ntemplates argument, as the number of templates.
    We extended the idea, including cluster label information. We compute,
    for each cluster, the mean distance criterion.
    We select, at most, ntemplates. With cluster labels, templates are attributed
    to clusters using an allocation method (e.g. d'Hondt)
    
    """
    
    # check inputs
    if numpy.isscalar(ntemplates):
        if ntemplates < 1:
            raise ValueError, "The total number of templates has to be at least 1."
    else:
        aux = numpy.sum(ntemplates)
        if aux < 1:
            raise ValueError, "The total number of templates has to be at least 1."
    
    templates = []
    distances = []
    
    if clusters is None:
        # original method (does not require clusters)
        n_clusters = 0
        if numpy.isscalar(ntemplates):
            ntemplatesPerCluster = [ntemplates]
        else:
            ntemplatesPerCluster = [ntemplates[0]]
        
        nt = ntemplatesPerCluster[0]
        length = len(data)
        
        if length == 0:
            pass
        elif length == 1:
            templates.append(data[0])
            distances.append(0.)
        elif length == 2:
            if nt == 1:
                # choose randomly
                r = round(config.random.rand())
                templates.append(data[r])
                distances.append(0.)
            else:
                for i in range(length):
                    templates.append(data[i])
                    distances.append(0.)
        else:
            # length > 2
            
            # compute mean distances
            _, sumY, s = avgDist(data, distMeasure, length)
            
            # select templates
            templates = data[s[:nt]]
            distances = sumY[s[:nt]]
    else:
        ks = clusters["clusters"].keys()
        
        # remove the outliers' cluster, if present
        if '-1' in ks:
            ks.remove('-1')
        
        n_clusters = len(ks)
        cardinals = [len(clusters["clusters"][k]) for k in ks]
        
        # verify if we have a vector or a scalar
        if numpy.isscalar(ntemplates):
            # allocate templates per cluster
            ntemplatesPerCluster = highestAveragesAllocator(cardinals, ntemplates, divisor='dHondt', check=True)
        else:
            # just copy
            ntemplatesPerCluster = ntemplates
        
        for i, k in enumerate(ks):
            c = numpy.array(clusters["clusters"][k])
            length = cardinals[i]
            nt = ntemplatesPerCluster[i]
            
            if nt == 0:
                continue
            
            if length == 0:
                continue
            elif length == 1:
                templates.append(data[c][0])
                distances.append(0.)
            elif length == 2:
                if nt == 1:
                    # choose randomly
                    r = round(config.random.rand())
                    templates.append(data[c][r])
                    distances.append(0.)
                else:
                    for j in range(length):
                        templates.append(data[c][j])
                        distances.append(0.)
            else:
                # length > 2
                
                # compute mean distances
                _, sumY, s = avgDist(data[c], distMeasure, length)
                
                # select templates
                sel = s[:nt]
                for item in sel:
                    templates.append(data[c][item])
                    distances.append(sumY[item])
                        
    res = {'templates': numpy.array(templates),
           'distances': numpy.array(distances),
           'ntemplates': len(templates),
           'algorithm': {'name': 'MDIST',
                         'parameters': {'clusters': n_clusters,
                                        'ntemplatesPerCluster': ntemplatesPerCluster
                                        }
                         }
           }
    
    return res


def centroids(data, clusters=None, ntemplates=1):
    """
    
    Template Selection: Centroids

    Requires that clusters are provided;
    if number of templates per cluster is 1, then templates
    are just the centroids; if the number of clusters is >1
    then k-means is used to determine more templates on each cluster. 
    
    """
    
    # check inputs
    if clusters is None:
        raise TypeError, "Centroids method requires the definition of clusters."
    
    # cluster labels
    ks = clusters["clusters"].keys()
    
    # remove the outliers' cluster, if present
    if '-1' in ks:
        ks.remove('-1')
    
    n_clusters = len(ks)
    cardinals = [len(clusters["clusters"][k]) for k in ks]
    
    # check number of templates
    if numpy.isscalar(ntemplates):
        if ntemplates < 1:
            raise ValueError, "The total number of templates has to be at least 1."
        # allocate templates per cluster
        ntemplatesPerCluster = highestAveragesAllocator(cardinals, ntemplates, divisor='dHondt', check=True)
    else:
        aux = numpy.sum(ntemplates)
        if aux < 1:
            raise ValueError, "The total number of templates has to be at least 1."
        # just copy
        ntemplatesPerCluster = ntemplates
    
    # select templates
    templates = []
    for i, k in enumerate(ks):
        c = numpy.array(clusters["clusters"][k])
        length = cardinals[i]
        nt = ntemplatesPerCluster[i]
        
        # ignore cases
        if nt == 0 or length == 0:
            continue
        
        if nt == 1:
            # cluster centroid
            templates.append(numpy.mean(data[c], axis=0))
        elif nt == length:
            # centroids are the samples
            templates.extend(data[c])
        else:
            # divide space using k-means
            nb = min([nt, length])
            # centroidsKmeans, _ = kmeans(data[c], nb, 50) # bug in initialization
            centroidsKmeans, _ = kmeans2(data[c], k=nb, iter=50, minit='points')
            for item in centroidsKmeans:
                templates.append(item)
    
    res = {'templates': numpy.array(templates),
           'ntemplates': len(templates),
           'algorithm': {'name': 'Centroids',
                         'parameters': {'clusters': n_clusters,
                                        'ntemplatesPerCluster': ntemplatesPerCluster,
                                        },
                         },
           }
    
    return res


def leadCentroids(data, **kwargs):
    """
    Perform centroids template selection for each lead independently.
    
    WARNING: Has a dirty hack to be compatible with Francis' dissimilarity code.
    """
    
    try:
        clusters = kwargs.pop('clusters')
    except KeyError:
        raise KeyError, "Please provide the input clusters."
    
    out = {}
    for key in data.iterkeys():
        out[key + '-tpl'] = centroids(data=data[key], clusters=clusters[key], **kwargs)['templates']
        out[key] = data[key]
    
    return out

