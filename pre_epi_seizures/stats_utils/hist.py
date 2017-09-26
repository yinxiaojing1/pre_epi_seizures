
def unspecific_hist(labels, sz, bins):

    hist = dict()
    for k in labels.keys():
        hist_per_seizure = np.asarray([np.histogram(sz[i][labels[k][0][i][0]:labels[k][0][i][1]], bins)[0]
                                  for i in xrange(len(sz))])
        hist_total = np.sum(hist_per_seizure, axis=0)
        hist[k] = (hist_total, labels[k][1])
    return hist


def histogram(signal_arguments, sampling_rate):
    signal_list = signal_arguments['feature_group_to_process']
    labels = signal_arguments['labels']
    bins = np.linspace(0, 1, 100)
    hist = [(label, sz, bins) 
        for label, rpeaks in zip(signal_list, rpeaks_list)]
    stop