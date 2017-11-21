
# ------------------------------
from load_for_class import *
from ploting_temp import *

# ------------------------------
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import pandas as pd

# # Memory friendly probably slower. Also easier to read.
# def compute_save_cluster(path_to_load, path_to_save,
#                          set_feature_group_name):

#     # For Each Record
#     for feature_group_name in set_feature_group_name:

#         feature_group = feature_group_name[0]
#         name_record = feature_group_name[1]


def kmeans_clustering_sklearn(n_clusters, init):
    sklearn_clustering_obj = KMeans(n_clusters=n_clusters, random_state=None, init=init)
    return sklearn_clustering_obj


def gauss_mixture_sklearn(n_clusters, init):
    sklearn_clustering_obj = GaussianMixture(n_components=1,
                                             covariance_type='full',
                                             tol=0.001, reg_covar=1e-06, 
                                             max_iter=100, n_init=1, 
                                             init_params='kmeans', 
                                             weights_init=None, means_init=None,
                                             precisions_init=None, random_state=None,
                                             warm_start=False, verbose=0,
                                             verbose_interval=10)
    return sklearn_clustering_obj


def agglomerative_clustering_sklearn(n_clusters, init):
    sklearn_clustering_obj = AgglomerativeClustering(n_clusters=n_clusters, 
                                                     affinity='euclidean',
                                                     compute_full_tree='auto',
                                                     linkage='ward')

    return sklearn_clustering_obj


def compute_partition(feature_array, clustering_obj):
    # cluster based on cluster settings
    clustering_obj.fit(feature_array)
    return clustering_obj


def setup_clustering(method, n_clusters, init):
    # init kmeans
    if method == 'kmeans':
        clustering_obj = kmeans_clustering_sklearn(n_clusters=n_clusters,
                                                   init=init)
    if method == 'gauss':
        clustering_obj = gauss_mixture_sklearn(n_clusters=n_clusters,
                                                   init=init)
    if method == 'agglom':
        clustering_obj = agglomerative_clustering_sklearn(n_clusters=n_clusters,
                                                   init=init)
    return clustering_obj


def get_labels(clustering_obj):
    labels = clustering_obj.labels_
    return labels


def get_partition_result(feature_array, clustering_obj):
    partition = compute_partition(feature_array=feature_array,
                                  clustering_obj=clustering_obj)
    labels = get_labels(clustering_obj=partition)
    return labels


def create_ensemble(feature_array, method, n_runs, n_clusters, init):
    # iterate through number of clusters
    clustering_obj = setup_clustering(method=method, n_clusters=n_clusters, init=init)
    ensemble = np.array([get_partition_result(feature_array=feature_array,
                                              clustering_obj=clustering_obj)
                         for i in xrange(0, n_runs)
                         ]
                        )
    return ensemble









