# ------------------------------
from load_for_class import *
from ploting_temp import *
from clustering_new import *

# ------------------------------
from sklearn.cluster import KMeans
import pandas as pd




path_to_load = '/Volumes/ASSD/pre_epi_seizures/h5_files/processing_datasets/seizure_datasets_new.h5'
path_to_map= '/Volumes/ASSD/pre_epi_seizures/h5_files/processing_datasets/seizure_datasets_new_map.txt'
path_to_save = '/Volumes/ASSD/pre_epi_seizures/plotting_utils/clustering/'

feature_group = 'pca_beat_amp_computation'
feature_groups = get_feature_group_name_list(path_to_map,
                                             feature_group + '#')

feature_group_nr = 0
feature_group_to_load = feature_groups[feature_group_nr]

features,\
    features_mdata = load_all_features_from_disk(
                                                path_to_load,
                                                feature_group_to_load)
features_windows,\
    features_windows_mdata = load_all_feature_windows_from_disk(
                                                 path_to_load,
                                                 feature_group_to_load)

print feature_group_to_load


set_feature_group_name = create_set_from_disk(path_to_load, feature_group_to_load)


record_nr = 2
feature_group_name = set_feature_group_name[record_nr]
record_name = feature_group_name[1]
feature_group = feature_group_name[0]


data = features[record_nr]
data_window = features_windows[record_nr]
columns = features_mdata[record_nr]['feature_legend']
data = pd.DataFrame(data.T, columns=columns)



directory = path_to_save + '/' + feature_group + '/' + record_name
fetch_directory(directory)

nr_feature = 0

ensemble = create_ensemble(feature_array=data, method='agglom', n_runs=1, n_clusters=6, init='random')

print ensemble.shape

for partition in ensemble:
    print partition


# plot_results_clustering(directory, cluster=kmeans, feature=data[columns[nr_feature]],
#                         feature_window=data_window,
#                         feature_name=columns[nr_feature], record_name=record_name)
