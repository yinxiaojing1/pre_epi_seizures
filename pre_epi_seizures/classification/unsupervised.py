from kmeans import *

import random


def random_centroids_from_data(labels):
    inter_ictal = np.random.randint(0, labels['inter_ictal'][0])
    pre_ictal = np.random.randint(labels['inter_ictal'][0], labels['pre_ictal'][0])
    ictal = np.random.randint(labels['pre_ictal'][0], labels['ictal'][0])
    post_ictal = np.random.randint(labels['ictal'][0], labels['post_ictal'][0])
    print inter_ictal
    return [inter_ictal, pre_ictal, ictal, post_ictal]

def compute_cluster_labels(cluster_assignment, labels):
    hist_inter_ictal = [len(np.where(cluster_assignment[0:labels['inter_ictal'][0]]==0)[0]),
        len(np.where(kmeans.labels_[0:labels['inter_ictal'][0]]==1)[0]),
        len(np.where(kmeans.labels_[0:labels['inter_ictal'][0]]==2)[0]),
        len(np.where(kmeans.labels_[0:labels['inter_ictal'][0]]==3)[0])]

    print hist_inter_ictal

    hist_pre_ictal = [len(np.where(kmeans.labels_[labels['inter_ictal'][0]:labels['pre_ictal'][0]]==0)[0]),
        len(np.where(kmeans.labels_[labels['inter_ictal'][0]:labels['pre_ictal'][0]]==1)[0]),
        len(np.where(kmeans.labels_[labels['inter_ictal'][0]:labels['pre_ictal'][0]]==2)[0]),
        len(np.where(kmeans.labels_[labels['inter_ictal'][0]:labels['pre_ictal'][0]]==3)[0])]

    print hist_pre_ictal

    hist_ictal = [len(np.where(kmeans.labels_[labels['pre_ictal'][0]:labels['ictal'][0]]==1)[0]),
        len(np.where(kmeans.labels_[labels['pre_ictal'][0]:labels['ictal'][0]]==1)[0]),
        len(np.where(kmeans.labels_[labels['pre_ictal'][0]:labels['ictal'][0]]==1)[0]),
        len(np.where(kmeans.labels_[labels['pre_ictal'][0]:labels['ictal'][0]]==1)[0]),

    print hist_ictal

    hist_ictal = [len(np.where(kmeans.labels_[labels['ictal'][0]:labels['post_ictal'][0]]==1)[0]),
        len(np.where(kmeans.labels_[labels['ictal'][0]:labels['post_ictal'][0]]==1)[0]),
        len(np.where(kmeans.labels_[labels['ictal'][0]:labels['post_ictal'][0]]==1)[0]),
        len(np.where(kmeans.labels_[labels['ictal'][0]:labels['post_ictal'][0]]==1)[0]),

    print hist_post_ictal
def unsurpervised_exploration(data, labels, method='kmeans'):
    initial_centroids = random_centroids_from_data(labels)
    
    kmeans = KMeans(n_clusters=4, init=initial_centroids).fit(data)
    stop


