from scaling import scale


import numpy as np
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def kmeans_class(data, rpeaks, time_before_seizure, time_after_seizure):
    # Time labels



    beat_inter_ictal = np.where(np.logical_and(rpeaks>=i_sample_inter_ictal, rpeaks<f_sample_inter_ictal))[0]
    beat_pre_ictal = np.where(np.logical_and(rpeaks>=i_sample_pre_ictal, rpeaks<f_sample_pre_ictal))[0]
    beat_ictal = np.where(np.logical_and(rpeaks>=i_sample_ictal, rpeaks<f_sample_ictal))[0]
    beat_post_ictal = np.where(np.logical_and(rpeaks>=i_sample_post_ictal, rpeaks<f_sample_post_ictal))[0]



    init = 100
    cluster_inter_ictal = beat_inter_ictal[random.choice(list(enumerate(beat_inter_ictal)))[0]]
    cluster_pre_ictal = beat_pre_ictal[random.choice(list(enumerate(beat_pre_ictal)))[0]]
    cluster_ictal = beat_ictal[random.choice(list(enumerate(beat_ictal)))[0]]
    cluster_post_ictal = beat_post_ictal[random.choice(list(enumerate(beat_post_ictal)))[0]]

    centroids = data[np.asarray([cluster_inter_ictal, cluster_pre_ictal, cluster_ictal, cluster_post_ictal])]

    beat_inter_ictal = np.where(np.logical_and(rpeaks>=i_sample_inter_ictal, rpeaks<f_sample_inter_ictal))[0][-1]
    beat_pre_ictal = np.where(np.logical_and(rpeaks>=i_sample_pre_ictal, rpeaks<f_sample_pre_ictal))[0][-1]
    beat_ictal = np.where(np.logical_and(rpeaks>=i_sample_ictal, rpeaks<f_sample_ictal))[0][-1]
    beat_post_ictal = np.where(np.logical_and(rpeaks>=i_sample_post_ictal, rpeaks<f_sample_post_ictal))[0][-1]

    # clusters = {'first':[cluster_inter_ictal, cluster_pre_ictal, cluster_ictal, cluster_post_ictal]}
    # template = centroid_templates(data, clusters, 4)[0]
    kmeans = KMeans(n_clusters=6, init='random', n_init=30).fit(data)

    hist_inter_ictal = [len(np.where(kmeans.labels_[0:beat_inter_ictal]==0)[0]),
        len(np.where(kmeans.labels_[0:beat_inter_ictal]==1)[0]),
        len(np.where(kmeans.labels_[0:beat_inter_ictal]==2)[0]),
        len(np.where(kmeans.labels_[0:beat_inter_ictal]==3)[0])]

    print hist_inter_ictal

    hist_pre_ictal = [len(np.where(kmeans.labels_[beat_inter_ictal:beat_pre_ictal]==0)[0]),
        len(np.where(kmeans.labels_[beat_inter_ictal:beat_pre_ictal]==1)[0]),
        len(np.where(kmeans.labels_[beat_inter_ictal:beat_pre_ictal]==2)[0]),
        len(np.where(kmeans.labels_[beat_inter_ictal:beat_pre_ictal]==3)[0])]

    print hist_pre_ictal

    hist_ictal = [len(np.where(kmeans.labels_[beat_pre_ictal:beat_ictal]==0)[0]),
        len(np.where(kmeans.labels_[beat_pre_ictal:beat_ictal]==1)[0]),
        len(np.where(kmeans.labels_[beat_pre_ictal:beat_ictal]==2)[0]),
        len(np.where(kmeans.labels_[beat_pre_ictal:beat_ictal]==3)[0])]

    print hist_ictal

    hist_post_ictal = [len(np.where(kmeans.labels_[beat_ictal:beat_post_ictal]==0)[0]),
        len(np.where(kmeans.labels_[beat_ictal:beat_post_ictal]==1)[0]),
        len(np.where(kmeans.labels_[beat_ictal:beat_post_ictal]==2)[0]),
        len(np.where(kmeans.labels_[beat_ictal:beat_post_ictal]==3)[0])]

    print hist_post_ictal

    plt.subplot(4, 1, 1)
    plt.bar(range(len(hist_inter_ictal)), hist_inter_ictal,width=0.1, color = 'green')
    plt.ylim([0, 1000])
    plt.subplot(4, 1, 2)
    plt.ylim([0, 1000])
    plt.bar(range(len(hist_pre_ictal)), hist_pre_ictal,width=0.1, color = 'orange')
    plt.subplot(4, 1, 3)
    plt.ylim([0, 1000])
    plt.bar(range(len(hist_ictal)), hist_ictal,width=0.1, color = 'red')
    plt.ylim([0, 1000])
    plt.subplot(4, 1, 4)
    plt.bar(range(len(hist_post_ictal)), hist_post_ictal,width=0.1, color = 'yellow')
    plt.ylim([0, 1000])
    plt.show()

    stop
    # clusters = kmeans(data=data, k=4, init=centroids, max_iter=3000, n_init=10, tol=0.0001)
    centroids = clusters['clusters'].keys()
    cluster_0 = clusters['clusters'][centroids[1]]


    print len(cluster_0)
    # print np.shape(template)
    # print np.shape(data)

    hist0, bins = np.histogram(clusters['clusters'][centroids[0]], bins=[0, beat_inter_ictal, beat_pre_ictal, beat_ictal, beat_post_ictal])
    hist1, bins = np.histogram(clusters['clusters'][centroids[1]], bins=[0, beat_inter_ictal, beat_pre_ictal, beat_ictal, beat_post_ictal])
    hist2, bins = np.histogram(clusters['clusters'][centroids[2]], bins=[0, beat_inter_ictal, beat_pre_ictal, beat_ictal, beat_post_ictal])
    hist3, bins = np.histogram(clusters['clusters'][centroids[3]], bins=[0, beat_inter_ictal, beat_pre_ictal, beat_ictal, beat_post_ictal])

    print hist0
    print hist1
    print hist2
    print hist3

    plt.subplot(4, 1, 1)
    plt.bar(range(len(hist0)),hist0/len(beat_inter_ictal),width=0.5, color = 'g')
    plt.subplot(4, 1, 2)
    plt.bar(range(len(hist1)),hist1/len(beat_pre_ictal),width=0.5, color = 'orange')
    plt.subplot(4, 1, 3)
    plt.bar(range(len(hist2)),hist2/len(beat_ictal),width=0.5, color = 'red')
    plt.subplot(4, 1, 4)
    plt.bar(range(len(hist3)),hist3/len(beat_post_ictal),width=0.5, color = 'bu')
    # plt.subplot(4, 1, 1)
    # plt.plot(template[0], color=color_inter_ictal)
    # plt.subplot(4, 1, 2)
    # plt.plot(template[1], color=color_pre_ictal)
    # plt.subplot(4, 1, 3)
    # plt.plot(template[2], color=color_ictal)
    # plt.subplot(4, 1, 4)
    # plt.plot(template[3], color=color_post_ictal)
    # plt.show()

    # plt.subplot(4, 1, 1)
    # plt.plot(data[clusters[0]], color=color_inter_ictal)
    # plt.subplot(4, 1, 2)
    # plt.plot(template[1], color=color_pre_ictal)
    # plt.subplot(4, 1, 3)
    # plt.plot(template[2], color=color_ictal)
    # plt.subplot(4, 1, 4)
    # plt.plot(template[3], color=color_post_ictal)
    # plt.hist(cluster_0, bins=[0, beat_inter_ictal, beat_pre_ictal, beat_ictal, beat_post_ictal], histtype='bar', ec='black')
    # plt.hist(cluster_0, bins='auto', histtype='bar', ec='black')
    plt.show()
    print scaled_data