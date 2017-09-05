from pre_epi_seizures.Preprocessing.pre_processing import *

from pre_epi_seizures.classification.labels import *


sampling_rate = 1000
time_before_seizure = 30
time_after_seizure = 10
# path_to_load = '~/Desktop/phisionet_seizures_new.h5'
# sampling_rate = 1000
path_to_load = '~/Desktop/seizure_datasets_new.h5'
# name_list = [str(time_before_seizure*60) + '_' + str(time_after_seizure*60)]
# group_list_raw = ['raw']
# group_list_baseline_removal = ['medianFIR']
# group_list_noise_removal = ['FIR_lowpass_40hz']
# group_list_esksmooth = ['esksmooth']
dataset_name = str(
time_before_seizure*60) + '_' + str(time_after_seizure*60)
raw_name = 'raw'


raw = load_feature(path_to_load, raw_name, files='existent', feature_group_to_process=dataset_name)

labels = create_labels(1000, 29, 9, 30, 10)

print labels