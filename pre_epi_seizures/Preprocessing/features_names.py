from filtering import baseline_removal, noise_removal, create_filtered_dataset

from segmentation import create_rpeak_dataset, create_heart_beat_dataset,\
    compute_beats, find_rpeaks, detect_rpeaks

from resampling import resample_rpeaks, interpolate_signal

from morphology import *

from Filtering.eksmoothing import *

a = locals()['create_rpeak_dataset']

{'baseline_removal': 'rpeaks': create_rpeak_dataset, }

