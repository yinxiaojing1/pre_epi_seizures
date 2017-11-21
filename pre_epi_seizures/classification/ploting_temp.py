
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set()
def fetch_directory(directory):
        # ** Create directory
    if not os.path.exists(directory):
        print 'MAKING'
        os.makedirs(directory)



def _plot_results_clustering(nr_clusters, labels, feature, feature_window):
    f1 = plt.figure()
    ax1 = f1.add_subplot(1,1,1)
    for i in xrange(0, nr_clusters):
        ix_cluster_i = np.where(labels == i)[0]
        ax1.plot(feature_window[ix_cluster_i], feature[ix_cluster_i], 'o')

    return f1



def plot_results_clustering(directory, cluster, feature, feature_window, feature_name, record_name):
    nr_clusters = cluster.n_clusters
    labels = cluster.labels_
    f1 = _plot_results_clustering(nr_clusters, labels, feature, feature_window)

    f1.savefig(directory + '/' + feature_name)






