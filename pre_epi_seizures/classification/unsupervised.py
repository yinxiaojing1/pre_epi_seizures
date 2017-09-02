from kmeans import *

import random
import numpy as np


def random_centroids_from_data(labels):
    inter_ictal = np.random.randint(0, labels['inter_ictal'][0])
    pre_ictal = np.random.randint(labels['inter_ictal'][0], labels['pre_ictal'][0])
    ictal = np.random.randint(labels['pre_ictal'][0], labels['ictal'][0])
    post_ictal = np.random.randint(labels['ictal'][0], labels['post_ictal'][0])
    print inter_ictal
    return np.asarray([inter_ictal, pre_ictal, ictal, post_ictal])

def compute_cluster_labels(n_cluster, cluster_assignment, labels):

    hist_inter_ictal = [len(np.where(cluster_assignment[0:labels['inter_ictal'][0]]==i)[0]) for i in xrange(n_cluster)]

    print hist_inter_ictal

    hist_pre_ictal = [len(np.where(cluster_assignment[labels['inter_ictal'][0]:labels['pre_ictal'][0]]==i)[0]) for i in xrange(n_cluster)]

    print hist_pre_ictal

    hist_ictal = [len(np.where(cluster_assignment[labels['pre_ictal'][0]:labels['ictal'][0]]==i)[0]) for i in xrange(n_cluster)]

    print hist_ictal

    hist_post_ictal = [len(np.where(cluster_assignment[labels['ictal'][0]:labels['post_ictal'][0]]==i)[0]) for i in xrange(n_cluster)]

    return np.asarray([hist_inter_ictal, hist_pre_ictal, hist_ictal, hist_post_ictal])

def unsurpervised_exploration(data, labels, method='kmeans'):
    initial_centroids = random_centroids_from_data(labels)
    print initial_centroids
    n_cluster=4
    kmeans = KMeans(n_clusters=n_cluster, init=data[initial_centroids]).fit(data)
    hist = compute_cluster_labels(n_cluster, kmeans.labels_, labels)
    # print np.true_divide(hist[0],sum(hist[0]))
    # print hist[0]/sum(hist[0])
    # stop
    plt.figure()
    plt.subplot(4, 1, 1)
    plt.bar(range(len(hist[0])), np.true_divide(hist[0],sum(hist[0])),width=0.1, color = 'green')
    plt.ylim([0, 1])
    plt.subplot(4, 1, 2)
    plt.ylim([0, 1])
    plt.bar(range(len(hist[1])), np.true_divide(hist[1],sum(hist[1])),width=0.1, color = 'orange')
    plt.subplot(4, 1, 3)
    plt.ylim([0, 1])
    plt.bar(range(len(hist[2])), np.true_divide(hist[2],sum(hist[2])),width=0.1, color = 'red')
    plt.ylim([0, 1])
    plt.subplot(4, 1, 4)
    plt.bar(range(len(hist[3])), np.true_divide(hist[3],sum(hist[3])),width=0.1, color = 'yellow')
    plt.ylim([0, 1])


    print kmeans.cluster_centers_
    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(kmeans.cluster_centers_[0,:])
    plt.subplot(4, 1, 2)
    plt.plot(kmeans.cluster_centers_[1,:])
    plt.subplot(4, 1, 3)
    plt.plot(kmeans.cluster_centers_[2,:])
    plt.subplot(4, 1, 4)
    plt.plot(kmeans.cluster_centers_[3,:])
    plt.show()

    stop


