import numpy as np


def create_labels(sampling_rate, t_inter_ictal, t_ictal, time_before_seizure, time_after_seizure):

    sampling_rate = 1000
    ti_inter_ictal = 0
    i_sample_inter_ictal = sampling_rate * ti_inter_ictal * 60
    tf_inter_ictal = t_inter_ictal
    f_sample_inter_ictal = sampling_rate * tf_inter_ictal * 60
    inter_ictal = np.arange(i_sample_inter_ictal, f_sample_inter_ictal)
    color_inter_ictal = 'green'

    ti_pre_ictal = tf_inter_ictal
    i_sample_pre_ictal = sampling_rate * ti_pre_ictal * 60
    tf_pre_ictal = time_before_seizure
    f_sample_pre_ictal = sampling_rate * tf_pre_ictal * 60
    pre_ictal = np.arange(i_sample_pre_ictal, f_sample_pre_ictal)
    color_pre_ictal = 'orange'

    ti_ictal = time_before_seizure
    i_sample_ictal = sampling_rate * ti_ictal * 60
    tf_ictal = time_before_seizure + t_ictal
    f_sample_ictal = sampling_rate * tf_ictal * 60
    ictal = np.arange(i_sample_ictal, f_sample_ictal)
    color_ictal = 'red'

    ti_post_ictal = tf_ictal
    i_sample_post_ictal = sampling_rate * ti_post_ictal * 60
    tf_post_ictal = time_before_seizure + time_after_seizure
    f_sample_post_ictal = sampling_rate * tf_post_ictal * 60
    post_ictal = np.arange(f_sample_post_ictal, f_sample_post_ictal)
    color_post_ictal = 'blue'

    labels = dict()
    labels['inter_ictal'] = (f_sample_inter_ictal, color_inter_ictal)
    labels['pre_ictal'] = (f_sample_pre_ictal, color_pre_ictal)
    labels['ictal'] = (f_sample_ictal, color_ictal)
    labels['post_ictal'] = (f_sample_post_ictal, color_post_ictal)

    return labels


def create_fiducial_labels(fiducial_points, ecg_labels): 
    inter_ictal = np.where(np.logical_and(fiducial_points>=0, fiducial_points<ecg_labels['inter_ictal'][0]))[0]
    pre_ictal = np.where(np.logical_and(fiducial_points>=ecg_labels['inter_ictal'][0], fiducial_points<ecg_labels['pre_ictal'][0]))[0]
    ictal = np.where(np.logical_and(fiducial_points>=ecg_labels['pre_ictal'][0], fiducial_points<ecg_labels['ictal'][0]))[0]
    post_ictal = np.where(np.logical_and(fiducial_points>=ecg_labels['ictal'][0], fiducial_points<ecg_labels['post_ictal'][0]))[0]

    labels = dict()
    labels['inter_ictal'] = (inter_ictal[-1], ecg_labels['inter_ictal'][1])
    labels['pre_ictal'] = (pre_ictal[-1], ecg_labels['pre_ictal'][1])
    labels['ictal'] = (ictal[-1], ecg_labels['ictal'][1])
    labels['post_ictal'] = (post_ictal[-1], ecg_labels['post_ictal'][1])

    return labels


def _create_label(sampling_rate, name, color, ti, tf):
    sampling_rate = 1000
    ti_inter_ictal = 0
    i_sample_inter_ictal = sampling_rate * ti_inter_ictal * 60
    tf_inter_ictal = t_inter_ictal
    f_sample_inter_ictal = sampling_rate * tf_inter_ictal * 60
    inter_ictal = np.arange(i_sample_inter_ictal, f_sample_inter_ictal)
    color_inter_ictal = 'green'


# labes = create_labels(1000, 25, 5, 30, 10)

# print labes