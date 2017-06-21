"""
Outlier removal test
@author: Afonso Eduardo
"""
from __future__ import division

import os
import itertools

import numpy as np
import scipy
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from pyclustering.cluster.kmedians import kmedians
from sklearn.preprocessing import scale


from preprocessing import read_dataset_csv, save_dataset_csv, plot_dataset


def check_prototypes(X, y, scaler='minmax', ceval=np.mean, norm=2,
                     k=100, random_state=42, n_init=10, max_iter=1000,
                     visualize=False, discard_class=True):
    if scaler=='minmax':
        X = (X - X.min(axis=1)[:,None]) / (X.max(axis=1) - X.min(axis=1))[:,None]
    elif scaler=='standard':
        X = scale(X1, axis=1)
    if norm==1:
        centroids =  KMeans(n_clusters=k, init='k-means++', n_init=1,
                            max_iter=1, algorithm='full', random_state=random_state).fit(X).cluster_centers_
        centroids = kmedians(X, centroids).get_medians()
    elif norm==2:
        centroids = KMeans(n_clusters=k, init='k-means++', n_init=n_init,
                       max_iter=max_iter, algorithm='full', random_state=random_state).fit(X).cluster_centers_
    if visualize:
        plt.figure(); plt.plot(centroids.T); plt.show()
    cdists =  ceval(cdist(X, centroids), axis=1)
    cstats = pd.DataFrame(cdists, columns=['wvdist']).describe()
    dist_thre = (cstats.loc['75%']+1.5*(cstats.loc['75%'] - cstats.loc['25%'])).values[0]
    discard_idx = np.where(cdists > dist_thre)[0]
    discard_uids = np.unique(y[discard_idx])
    if discard_class:
       discard_idx = np.arange(len(y))[np.in1d(y, discard_uids)]
    return discard_idx, discard_uids

def check_wave(X, y, wvtype=np.mean, wvdist=np.linalg.norm, wveval=np.mean, discard_class=True):
    X, uids = np.atleast_2d(X), np.unique(y)
    waves = np.vstack([wvtype(X[np.where(y==yy)[0]], axis=0) for yy in uids])
    waves = np.repeat(waves, np.unique(y, return_counts=True)[1], axis=0)
    wvdists = wvdist(X - waves, axis=1)
    wvdists = np.array([wveval(wvdists[np.where(y==yy)[0]]) for yy in uids])
    wvstats = pd.DataFrame(wvdists).describe()
    wvdist_thre = (wvstats.loc['75%']+1.5*(wvstats.loc['75%'] - wvstats.loc['25%'])).values[0]
    discard_idx = np.where(wvdists > wvdist_thre)[0]
    discard_uids = np.unique(y[discard_idx])
    if discard_class:
       discard_idx = np.arange(len(y))[np.in1d(y, discard_uids)]
    return discard_idx, discard_uids

def check_reconstruction(X, X0, y, discard_class=True):
    dists = np.linalg.norm(X - X0, axis=1)
    stats = pd.DataFrame(dists, columns=['wvdist']).describe()
    dist_thre = (stats.loc['75%']+1.5*(stats.loc['75%'] - stats.loc['25%'])).values[0]
    discard_idx = np.where(dists > dist_thre)[0]
    discard_uids = np.unique(y[discard_idx])
    if discard_class:
       discard_idx = np.arange(len(y))[np.in1d(y, discard_uids)]
    return discard_idx, discard_uids

def check_wraparound(X, y, value=2, discard_class=True):
    discard_idx = np.unique(np.hstack((np.where(X[:,0]>value)[0],np.where(X[:,-1]>value)[0])))
    discard_uids = np.unique(y[discard_idx])
    if discard_class:
       discard_idx = np.arange(len(y))[np.in1d(y, discard_uids)]
    return discard_idx, discard_uids


if __name__ == '__main__':
    f_path = lambda datafile: os.path.join(os.path.split(os.getcwd())[0], datafolder, datafile)
    datafolder = 'Data'
    X0, y0 = read_dataset_csv(f_path('healthy_medianFIR_segments.csv'), multicolumn=True)
    X1, y1 = read_dataset_csv(f_path('healthy_medianFIR_segments_gaussians_modified.csv'), multicolumn=True)

    # Match datasets
    uids = np.unique(y1)
    keep_idx = np.in1d(y0, y1)
    X0, y0 = X0[keep_idx], y0[keep_idx]
    assert np.all(y0==y1)
    assert np.all(X0.shape == X1.shape)
    # Outlier removal
    uid_rec = check_reconstruction(X1, X0, y1)[1]
    uid_wrap = check_wraparound(X1, y1, value=2)[1]
    uid_waveX0 = check_wave(X0, y1, wvtype=np.median, wveval=np.max)[1]
    uid_waveX1 = check_wave(X1, y1, wvtype=np.median, wveval=np.max)[1]
    uid_proto_m = check_prototypes(X1, y1, scaler='minmax', k=100, norm=1, ceval=np.min)[1]
    uid_proto_s = check_prototypes(X1, y1, scaler='standard', k=100, norm=1, ceval=np.min)[1]
    discard_uids = np.unique(np.hstack((uid_rec, uid_wrap, uid_waveX0, uid_waveX1, uid_proto_m, uid_proto_s)))
    new_uids = np.array(list(set(uids) - set(discard_uids)))
    keep_idx = np.in1d(y1, new_uids)
    X1, y1 = X1[keep_idx], y1[keep_idx]
    print len(np.unique(y1))
    print len(X1)
    save_dataset_csv(X1, y1, f_path('selected_gaussians.csv'))
    plot_dataset(f_path('selected_gaussians.csv'), multicolumn=True)


