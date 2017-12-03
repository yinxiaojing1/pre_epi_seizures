# native------------------------------
import os


# 3dparty packages--------------------
import numpy as np
import pandas as pd
from sklearn.metrics import *


# classification packages-------------
from supervised_new import *
from prep_inputs import *
from plot_classification import *
from supervised_new import *
from parameter_search import *
from cross_validation import *


# - Appendix
def get_name_to_save(path_to_save, seizure_nr, feature_names, model):

    test_set = '-||| Test set: ' + str(seizure_nr) + '|||-'
    feature_names = '-||| Feature Names: ' + str(feature_names) + '|||-'
    model = '-||| Model: ' + str(model) + '|||-'
    return test_set + feature_names + model


def get_features(list_columns):
    return [feature 
            for feature in list_columns
            if feature != 'labels' and feature != 'colors' and feature != 'sample_domain']


def normalize(df):
    print df
    return (df - df.mean()) / (df.max() - df.min())


path_to_save = '/Volumes/ASSD/pre_epi_seizures/h5_files/classification_datasets'

# def create_name_to_save(path_to_save, features, )

# def load_compute_feature(filename):
#     try:
#         pd.read_hdf(path=path_to_save + filename + '.h5', key='test')

#     else

def get_model_from_name(model_str):
    if model_str == 'rbf_svc_sklearn':
      model = rbf_svc_sklearn

    return model


def analyze_leave_one_out_performance(performance):
    # print performance.columns

    metrics = ['recall', 'precision', 'f1']

    parameters = [item 
                  for item in performance.columns
                  if item not in metrics]

    performance = performance.groupby(parameters)[metrics].mean()

    best = list(performance['f1'].idxmax())

    return dict(zip(parameters, best))



def create_baseline_seizure_labels(pre_seizure_window_intrevals,
                                    baseline_data_struct,
                                    baseline_feature_names,
                                    baseline_data_window_struct,
                                    seizure_data_struct,
                                    seizure_feature_names,
                                    seizure_data_window_struct):
    # create labels based on pre_seizure_window_values
    baseline_label_struct = [{'baseline': (1, [(0, 50 * 60)], 'g')}] * len(baseline_data_struct)
    seizure_label_struct = [{'pre_seizure': (2, pre_seizure_window_intrevals, 'r')}] * len(seizure_data_struct)

    # get_label_list from the structures
    baseline_labels = get_labels_list( 
                                      baseline_data_struct,
                                      baseline_data_window_struct,
                                      baseline_label_struct)
    seizure_labels = get_labels_list( 
                                      seizure_data_struct,
                                      seizure_data_window_struct,
                                      seizure_label_struct)

    return baseline_labels, seizure_labels


def divide_datasets(baseline_test_ix, seizure_test_ix,
                                      baseline_data_struct,
                                      baseline_feature_names,
                                      baseline_data_window_struct,
                                      baseline_labels,
                                      seizure_data_struct,
                                      seizure_feature_names,
                                      seizure_data_window_struct,
                                      seizure_labels):

    baseline_test_struct = get_set_ix(baseline_test_ix,
                                       baseline_data_struct,
                                       baseline_feature_names,
                                       baseline_data_window_struct,
                                       baseline_labels)

    seizure_test_struct = get_set_ix(seizure_test_ix,
                               seizure_data_struct,
                               seizure_feature_names,
                               seizure_data_window_struct,
                               seizure_labels)

    seizure_training_ix = get_training_ix(seizure_test_ix,
                                         seizure_data_struct)
    # print seizure_training_ix

    seizure_training_struct = get_set_ix(seizure_training_ix,
                               seizure_data_struct,
                               seizure_feature_names,
                               seizure_data_window_struct,
                               seizure_labels)

    baseline_training_struct = baseline_test_struct

    return (baseline_test_struct, seizure_test_struct),\
            (baseline_training_struct, seizure_training_struct)



def training_crossvalidation(model,
                            baseline_test_ix, seizure_test_ix,
                            baseline_data_struct,
                            baseline_feature_names,
                            baseline_data_window_struct,
                            baseline_labels,
                            seizure_data_struct,
                            seizure_feature_names,
                            seizure_data_window_struct,
                            seizure_labels,
                            **model_parameters):

    test_struct, training_struct = divide_datasets(baseline_test_ix, seizure_test_ix,
                        baseline_data_struct,
                        baseline_feature_names,
                        baseline_data_window_struct,
                        baseline_labels,
                        seizure_data_struct,
                        seizure_feature_names,
                        seizure_data_window_struct,
                        seizure_labels)

    test_dataframe = create_dataframe_baseline_seizure(test_struct[0], test_struct[1])
    training_dataframe = create_dataframe_baseline_seizure(training_struct[0], training_struct[1])

    features = get_features(training_dataframe.columns)

    # label_list = sorted(test_dataframe['labels'].unique())

    model_parameters = model(training_dataframe, test_dataframe, features, **model_parameters)

    # test_labels = test_dataframe['labels']

    return model_parameters




def _training_leave_one_out(model, baseline_training_struct,
                            seizure_training_struct
                                        ):
    # * Unravel loaded data structure from patient
    # baseline dataset


    baseline_data_struct = baseline_training_struct[0]
    baseline_feature_names = baseline_training_struct[1]
    baseline_data_window_struct = baseline_training_struct[2]
    baseline_labels = baseline_training_struct[3]

    seizure_data_struct = seizure_training_struct[0]
    seizure_feature_names = seizure_training_struct[1]
    seizure_data_window_struct = seizure_training_struct[2]
    seizure_labels = seizure_training_struct[3]

    print 'Crossvalidation training....might take awhile....'

    # !!!! Aproximation !!!!! Change please
    baseline_test_ix = range(0, len(baseline_data_struct))

    metrics_list = [training_crossvalidation(model,
                            baseline_test_ix, seizure_test_ix,
                            baseline_data_struct,
                            baseline_feature_names,
                            baseline_data_window_struct,
                            baseline_labels,
                            seizure_data_struct,
                            seizure_feature_names,
                            seizure_data_window_struct,
                            seizure_labels,
                            C=C, gamma=gamma)
                    for seizure_test_ix in\
                         xrange(0, len(seizure_data_struct))
                    for gamma in gamma_list
                    for C in C_list]

    metrics_total = pd.DataFrame(metrics_list)

    return metrics_total


def training_leave_one_out(model_str, baseline_test_ix, seizure_test_ix,
                            baseline_data_struct,
                            baseline_feature_names,
                            baseline_data_window_struct,
                            baseline_labels,
                            seizure_data_struct,
                            seizure_feature_names,
                            seizure_data_window_struct,
                            seizure_labels,
                            parameter_grid):

    model = get_model_from_name(model_str)

    # !!!! Aproximation !!!!! Change please
    baseline_test_ix = range(0, len(baseline_data_struct))

    test_struct, training_struct = divide_datasets(baseline_test_ix, seizure_test_ix,
                                    baseline_data_struct,
                                    baseline_feature_names,
                                    baseline_data_window_struct,
                                    baseline_labels,
                                    seizure_data_struct,
                                    seizure_feature_names,
                                    seizure_data_window_struct,
                                    seizure_labels)

    # stop
    filename = get_name_to_save(path_to_save,
                                seizure_test_ix,
                                baseline_feature_names[0],
                                model_str)

    print path_to_save + filename + '.h5'

    try:
        metrics_total = pd.read_hdf(path_to_save + filename + '.h5', 'test')
        # print metrics_total
    except Exception as e:
        print e

        metrics_total = _training_leave_one_out(model,
                                        training_struct[0],
                                        training_struct[1],
                                        parameter_grid)
        metrics_total.to_hdf(path_to_save + filename + '.h5', 'test', format='f')


    return metrics_total


def __pipeline_simple_baseline_seizure(grid_seizure_test_ix,
                                      grid_baseline_test_ix,
                                      baseline_data_struct,
                                      baseline_feature_names,
                                      baseline_data_window_struct,
                                      baseline_labels,
                                      seizure_data_struct,
                                      seizure_feature_names,
                                      seizure_data_window_struct,
                                      seizure_labels):
    model = 'rbf_svc_sklearn'

    #
    parameter_grid = C_gamma_grid()


    leave_one_out_performances = [training_leave_one_out(model,
                                            baseline_test_ix,
                                            seizure_test_ix,
                                            baseline_data_struct,
                                            baseline_feature_names,
                                            baseline_data_window_struct,
                                            baseline_labels,
                                            seizure_data_struct,
                                            seizure_feature_names,
                                            seizure_data_window_struct,
                                            seizure_labels, 
                                            parameter_grid)
                     for seizure_test_ix, baseline_test_ix \
                     in zip(grid_seizure_test_ix, grid_baseline_test_ix)]

    best_parameter = [analyze_leave_one_out_performance(performance)
                      for performance in leave_one_out_performances]

    print best_parameter

    print leave_one_out

    stop

    stop




def _pipeline_simple_baseline_seizure(grid_window_values, 
                                      data_struct):
    baseline_data_struct = data_struct[0]
    baseline_feature_names = data_struct[1]
    baseline_data_window_struct = data_struct[2]

    seizure_data_struct = data_struct[3]
    seizure_feature_names = data_struct[4]
    seizure_data_window_struct = data_struct[5]

    for pre_seizure_window_intrevals in grid_window_values:


        # ** Loop through pairs of testing baseline_seizure
        # create_test_criterion < ---
        grid_seizure_test_ix = range(0, len(seizure_data_struct))
        grid_baseline_test_ix = [0] * len(grid_seizure_test_ix)

        baseline_labels, seizure_labels = create_baseline_seizure_labels(
                                            pre_seizure_window_intrevals,
                                            baseline_data_struct,
                                            baseline_feature_names,
                                            baseline_data_window_struct,
                                            seizure_data_struct,
                                            seizure_feature_names,
                                            seizure_data_window_struct)


        __pipeline_simple_baseline_seizure(grid_seizure_test_ix,
                                      grid_baseline_test_ix,
                                      baseline_data_struct,
                                      baseline_feature_names,
                                      baseline_data_window_struct,
                                      baseline_labels,
                                      seizure_data_struct,
                                      seizure_feature_names,
                                      seizure_data_window_struct,
                                      seizure_labels)


def pipeline_simple_baseline_seizure():

    # Select group of features
    feature_namea = 'hrv_time_features#'

    # Select Patient to analyze
    pt = 3
    data_struct = load_feature_groups_baseline_seizure_per_patient(
                                                    patient_nr=pt
                                                    )

    global path_to_save
    path_to_save = path_to_save + '/patient/'

    try:
        os.stat(path_to_save)
    except:
        os.mkdir(path_to_save)
    # # * Unravel loaded data structure from patient
    # # baseline dataset
    # baseline_data_struct = data_struct[0][0]
    # baseline_feature_names = data_struct[0][1]
    # baseline_data_window_struct = data_struct[1][0]
    # # seizure dataset
    # seizure_data_struct = data_struct[2][0]
    # seizure_feature_names = data_struct[2][1]
    # seizure_data_window_struct = data_struct[3][0]

    # ** Loop through pre_seizure window values
    # set begin_end tuple list for pre_seizure_intervals (Equal for all seizures) 
    grid_window_values = [
                           [(0, 10*60)], # grid value
                           [(0, 20*60)],
                           [(0, 30*60)],
                           [(0, 40*60)],
                           [(0, 50*60)]
                         ]

    _pipeline_simple_baseline_seizure(grid_window_values, 
                                      data_struct)


pipeline_simple_baseline_seizure()



supervised_dataset = pd.concat(supervised_DataFrames)

features = get_features(supervised_dataset.columns)


supervised_dataset_norm = normalize(supervised_dataset[features])


supervised_dataset_norm['labels'] = supervised_dataset['labels']

supervised_dataset_norm['colors'] = supervised_dataset['colors']


print 'Learning ...'
model = classify_svc_sklearn()
trained_model = train_from_object(obj,
                      supervised_dataset_norm[features],
                      supervised_dataset_norm['labels'])

print trained_obj

stop

jointplot(data=supervised_dataset, x=features[-3], y=features[-2], labels='labels', colors='colors')