from pre_epi_seizures.Preprocessing.pre_processing import *

from labels import *
from scaling import scale
from kmeans import *
from unsupervised import *


#signal
sampling_rate = 1000
time_before_seizure = 30
time_after_seizure = 10
path_to_load = '~/Desktop/seizure_datasets_new.h5'
dataset_name = str(
time_before_seizure*60) + '_' + str(time_after_seizure*60)
raw_name = 'raw'
baseline_removal_name = 'baseline_removal'
raw_dataset_name = dataset_name + '/' + raw_name
baseline_removal_dataset_name = raw_dataset_name + '/' + baseline_removal_name
decimated_dataset_name = baseline_removal_dataset_name + '/' + 'decimation'
eks_dataset_name = decimated_dataset_name + '/' + 'eks_smoothing'
interpolated_dataset_name = eks_dataset_name + '/' + 'interpolation'

interpolated = load_feature(path_to_load, 'interpolation', sampling_rate=500, files='existent', feature_group_to_process=eks_dataset_name)[0]
rpeaks = load_feature(path_to_load, 'rpeak_detection', files='existent', feature_group_to_process=interpolated_dataset_name)[0]
beats = load_feature(path_to_load, 'beat_phase_segmentation', files='existent', feature_group_to_process=interpolated_dataset_name, rpeak_group_to_process=interpolated_dataset_name + '/' + 'rpeak_detection')[0]

len(beats)
len(rpeaks)
# print beats
# print rpeaks
# stop
start = 280
end = 290
sz_nr = 1
data = beats[sz_nr]
plt.plot(data[0])
plt.show()
len(beats)
len(rpeaks)
fiducial = rpeaks[sz_nr]
signal = interpolated
signal_t = rpeaks
n = np.linspace(0, (len(signal[sz_nr])-1)/1000, len(signal[sz_nr]))
n_t = np.linspace(0, (len(signal_t[sz_nr])-1)/1000, len(signal_t[sz_nr]))

plt.subplot(1,2,1)
plt.title('interpolated ECG')
plt.plot(n, signal[sz_nr])
plt.plot(n[rpeaks[sz_nr]], signal[sz_nr][rpeaks[sz_nr]], 'o', color='g')
plt.xlim([start, end])
plt.xlabel('time[s]')
plt.subplot(1,2,2)
plt.title('Detrended and Denoised ECG')
plt.plot(n_t, signal_t[sz_nr])
plt.xlim([start, end])
plt.xlabel('time[s]')
plt.show()


# print data

labels = create_labels(1000, 5, 5, 30, 10)

fiducial_labels = create_fiducial_labels(fiducial, labels)

data = scale(np.asarray(data))
print data
unsurpervised_exploration(data, fiducial_labels, method='kmeans')

# kmeans_class(QRS[sz_nr], rpeaks[sz_nr], time_before_seizure, time_after_seizure)