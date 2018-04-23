# Imports
# Modelation
import sklearn.svm as svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeavePGroupsOut
from sklearn.pipeline import *
import sklearn.naive_bayes as nb
from sklearn.feature_selection import *
from sklearn.neural_network import *
from sklearn.neighbors import *
from keras.utils import np_utils
from classification.keras_neural_nets import *

# Exploratory Data Analysis
from classification.load_for_class import *
import convertpandas as cv_pd
import pandas as pd
import sklearn.preprocessing as pp
from sklearn.kernel_approximation import RBFSampler
from interim_processing import *
import iopes


# import plotting utils from Exploratory Data Analysis
import classification.eda.hist as plt_hist
import classification.eda.andrews as plt_and
import classification.eda.series as plt_ts
import classification.eda.box as plt_box
import classification.eda.scatter as plt_sc
import matplotlib.pyplot as plt

import os



def get_hyper_param_results(label_struct, baseline_label_struct,
                                       pipe, scaler, param_grid,
                                       patient_list,
                                       feature_slot,
                                       hyper_param,
                                       plot_eda_all_new,
                                       learn_flag,
                                       compute_all_new
                                       ):
    
    # State the parameters of the pipeline
    disk = '/mnt/Seagate/pre_epi_seizures/'
    baseline_files = 'h5_files/processing_datasets/baseline_datasets_new'
    seizure_files = 'h5_files/processing_datasets/seizure_datasets_new'

    lead_list = ['ECG-']

    interim_processing = [scaler]
    hist_bins = None
    dist = None
    flag_hist = True
    flag_andrews = True
    flag_series = True
    flag_box = True
    flag_pair = True
    assign_baseline = 'assign_equal_baseline_seizure'
    

    general_dir = disk + 'EDanalysis_new/'

    eda_dir = read_disk_space_hyper_param_results(general_dir,
                                                        patient_list = patient_list,
                                                        lead_list = lead_list,
                                                        scaler = scaler,
                                                        interim_processing = interim_processing,
                                                        assign_baseline = assign_baseline,
                                                        label_struct = label_struct,
                                                        baseline_label_struct = baseline_label_struct,
                                                        feature_slot=feature_slot, 
                                                        hyper_param=0)


    # Define grouping of the data
    group_id = 'seizure_nr'
    label = 'label'


    # define cross-validation strategy 
    cv_out = LeavePGroupsOut(n_groups=1)
    cv_in = LeavePGroupsOut(n_groups=1)

    # choose scoring
    scoring = ['f1_micro']

    # choose the seach function
    search_function = GridSearchCV
    
    # Get pipe steps
    pipe_steps = [step[0] for step in pipe.steps]
    
    classification_dir =  read_disk_space_hyper_param_results(directory=eda_dir,
                                                               pipe = str(pipe_steps),
                                                               param_grid = param_grid,
                                                               cv_out = cv_out,
                                                               cv_in = cv_in,
                                                               scoring = scoring,
                                                               search_function = search_function,
                                                               group_id=group_id,
                                                               label=label)
    
    # After determined path_to_save, now load all the files
    # from hyperparameter optimization
    hyper_parameterization_results = [load_optimization_test_file(classification_dir + name)
                                      for name in os.listdir(classification_dir)
                                      if 'hp_opt_results' in name]
    return hyper_parameterization_results[0]
    
  
def load_optimization_test_file(full_path_hp):
    return_struct = dict()
    store = pd.HDFStore(full_path_hp)
    results = store['cv_results']
    mdata = store.get_storer('cv_results').attrs.metadata
    y_test = store['y_test']
    y_pred = store['y_pred']
 
    return_struct['cv_results'] = results
    return_struct['y_test'] = y_test
    return_struct['y_pred'] = y_pred
    return_struct['best_params'] = mdata['best_params']
    return_struct['best_estimator'] = mdata['best_estimator']
    
    return return_struct


def read_disk_space_hyper_param_results(directory,
                                           **kwargs):
    """
    Returns the appropriate path to save the return of the classification pipelines
    kwargs: parameters of the pipeline
    
    """
    
    # Generate string to identify
    params_str = iopes.generate_string_identifier(**kwargs)
    
    # Generate identification based on hyperparameters,
    # using table in .hdf5
    identifier = iopes.get_eda_params_path(directory=directory,
                                           params_str = params_str)
    
    # Return the path
    path = directory + identifier + '/'
    
    return path

    
def get_learning_results(label_struct, baseline_label_struct,
                        pipe, scaler, param_grid,
                        patient_list,
                        feature_slot,
                        hyper_param,
                        plot_eda_all_new,
                        learn_flag,
                        compute_all_new
                       ):
    
    # State the parameters of the pipeline

    disk = '/mnt/Seagate/pre_epi_seizures/'
    baseline_files = 'h5_files/processing_datasets/baseline_datasets_new'
    seizure_files = 'h5_files/processing_datasets/seizure_datasets_new'

    lead_list = ['ECG-']

    interim_processing = [scaler]
    hist_bins = None
    dist = None
    flag_hist = True
    flag_andrews = True
    flag_series = True
    flag_box = True
    flag_pair = True
    assign_baseline = 'assign_equal_baseline_seizure'
    

    general_dir = disk + 'EDanalysis_new/'
    
    # choose data grouping 
    group_keys= ['patient_nr',
                 'seizure_nr',
                 'types_of_seizure',
                 'location']
    
    group_id = 'seizure_nr'
    
    # Get initial directory, for Exploratory Data Analysis    
    eda_dir =  read_disk_space_hyper_param_results(directory=general_dir ,
                                                               patient_list = patient_list,
                                                               lead_list = lead_list,
                                                               scaler = scaler,
                                                               interim_processing = interim_processing,
                                                               assign_baseline = assign_baseline,
                                                               label_struct = label_struct,
                                                               baseline_label_struct = baseline_label_struct,
                                                               feature_slot=feature_slot, 
                                                               group_id=group_id,
                                                               hyper_param=0)
    
    # information for sklearn labeling
    label = 'label'

    # define cross-validation strategy 
    cv_out = LeavePGroupsOut(n_groups=1)
    cv_in = LeavePGroupsOut(n_groups=1)

    # choose scoring
    scoring = ['f1_micro']

    # choose hyperparameter search function
    search_function = GridSearchCV
    
    # get steps of the pipeline
    pipe_steps = [step[0] for step in pipe.steps]
    
    # Get directory (should be nested) to save classification objects
    classification_dir = read_disk_space_hyper_param_results(directory=eda_dir,
                                                               pipe = str(pipe_steps),
                                                               param_grid = param_grid,
                                                               cv_out = cv_out,
                                                               cv_in = cv_in,
                                                               scoring = scoring,
                                                               search_function = search_function,
                                                               label=label)
    
    file = pd.read_hdf(classification_dir + 'classification_report.h5', '/report' )
    
    return file
    
    


def prepare_disk_space_hyper_param_results(directory,
                                           **kwargs):
    """
    Returns the appropriate path to save the return of the classification pipelines
    kwargs: parameters of the pipeline
    
    """
    
    # Generate string to identify
    params_str = iopes.generate_string_identifier(**kwargs)
    
    # Generate identification based on hyperparameters,
    # using table in .hdf5
    identifier = iopes.get_eda_params_path(directory=directory,
                                           params_str = params_str)
    
    # Create the path, if it doesnt exist
    path = directory + identifier + '/'
    import os
    if not os.path.exists(path):
        print 'The path doesnt exist. Creating...'
        os.mkdir(path) 
    
    if os.path.exists(path): 
        print 'Be careful the path already exists!'
    
    # Generate .txt file to backup the hyper_parameters
    iopes.generate_txt_file_params(path, params_str)
    
    return path


def interim_process(disk, seizure_files, baseline_files,
                    feature_slot, hyper_param,
                    patient_list, lead_list,
                    label_struct, baseline_label_struct,
                    interim_processing):
    
    # Ingest Seizure Data
    path_to_load = disk + seizure_files + '.h5'
    path_to_map = disk + seizure_files + '_map.txt'

    # Feature group to analyse -- point of entry
    feature_name = get_feature_group_name_list(path_to_map,
                                                   feature_slot)[hyper_param]

    seizure_data = cv_pd.convert_to_pandas(path_to_load, path_to_map,
                            patient_list, feature_name,
                            lead_list, label_struct)

    # Ingest Baseline Data

    # set Labeling structure
    path_to_load = disk + baseline_files + '.h5'
    path_to_map = disk + baseline_files + '_map.txt'

    # Feature group to analyse -- point of entry
    feature_name = get_feature_group_name_list(path_to_map,
                                                   feature_slot)[hyper_param]

    print feature_name

    baseline_data = cv_pd.convert_to_pandas(path_to_load, path_to_map,
                            patient_list, feature_name,
                            lead_list, 
                            baseline_label_struct)


    # Treat Baseline Data
    baseline_data = baseline_data.dropna(axis=0, how='any').reset_index(drop=True)

    baseline_data = assign_equal_baseline_seizure(baseline_data,
                                              seizure_data,
                                             'seizure_nr',
                                             'patient_nr')

    seizure_data = seizure_data.dropna(axis=0, how='any').reset_index(drop=True)

    data = pd.concat([seizure_data, baseline_data],
                     ignore_index=True)



    # Add Seizure Type
    cv_pd.add_seizure_types(data,
                            'patient_nr',
                            'seizure_nr',
                            'types_of_seizure',
                            'location')


    # state the Data metafeatures
    metafeatures = ['patient_nr', 'seizure_nr', 'time_sample', 'label', 'color', 'types_of_seizure', 'location']
    features = [column
                for column in data.columns
                if column not in metafeatures]

    # Drop missing values
    data = data.dropna(axis=0, how='any').reset_index(drop=True)

    # Interim process the data
    for step in interim_processing:
        X = data[features]

        X_norm_np = step.fit_transform(X)

        #X_norm = pd.DataFrame(X_norm_np, columns=X.columns)

        data[features] = X_norm_np
        
    return data, features, metafeatures
    

def supervised_pipeline(label_struct, baseline_label_struct,
                        pipe, scaler, param_grid,
                        patient_list,
                        feature_slot,
                        hyper_param,
                        plot_eda_all_new,
                        learn_flag,
                        compute_all_new,
                        n_jobs
                       ):
    # State the parameters of the pipeline

    disk = '/mnt/Seagate/pre_epi_seizures/'
    baseline_files = 'h5_files/processing_datasets/baseline_datasets_new'
    seizure_files = 'h5_files/processing_datasets/seizure_datasets_new'

    lead_list = ['ECG-']

    interim_processing = [scaler]
    hist_bins = None
    dist = None
    flag_hist = True
    flag_andrews = True
    flag_series = True
    flag_box = True
    flag_pair = True
    assign_baseline = 'assign_equal_baseline_seizure'
    

    general_dir = disk + 'EDanalysis_new/'
    
    # choose data grouping 
    group_keys= ['patient_nr',
                 'seizure_nr',
                 'types_of_seizure',
                 'location']
    
    group_id = 'seizure_nr'
    
    # Get initial directory, for Exploratory Data Analysis    
    eda_dir =  prepare_disk_space_hyper_param_results(directory=general_dir ,
                                                               patient_list = patient_list,
                                                               lead_list = lead_list,
                                                               scaler = scaler,
                                                               interim_processing = interim_processing,
                                                               assign_baseline = assign_baseline,
                                                               label_struct = label_struct,
                                                               baseline_label_struct = baseline_label_struct,
                                                               feature_slot=feature_slot, 
                                                               group_id=group_id,
                                                               hyper_param=hyper_param)
    
    # information for sklearn labeling
    label = 'label'

    # define cross-validation strategy 
    cv_out = LeavePGroupsOut(n_groups=1)
    cv_in = LeavePGroupsOut(n_groups=1)

    # choose scoring
    scoring = ['f1_micro']

    # choose hyperparameter search function
    search_function = GridSearchCV
    
    # get steps of the pipeline
    pipe_steps = [step[0] for step in pipe.steps]
    
    # Get directory (should be nested) to save classification objects
    classification_dir = prepare_disk_space_hyper_param_results(directory=eda_dir,
                                                               pipe = str(pipe_steps),
                                                               param_grid = param_grid,
                                                               cv_out = cv_out,
                                                               cv_in = cv_in,
                                                               scoring = scoring,
                                                               search_function = search_function,
                                                               label=label)


    if plot_eda_all_new:
        plot_eda(directory=classification_dir,
                 data_groups_list=data_groups_list)

        

    import classification.cross_validation as cv

    # ***********************************Learning****************************
    # Learn from data_struct using nested cross_validation
    # learninig is an optimization and respective test results
    # for each partition of the dataset according to cv_out
    
    # Load the data, according to specification (loading made by convert pandas)
    data_struct = interim_process(disk, seizure_files, baseline_files,
                    feature_slot, hyper_param,
                    patient_list, lead_list,
                    label_struct, baseline_label_struct,
                    interim_processing)
    data = data_struct[0]
    features = data_struct[1]
    meta_features = data_struct[2]
    


    if learn_flag:
         # prepare data for classification - watch out for memory concerns
        X = data[features]
        y = data[label]
        groups = data[group_id]
        learning_results = cv.nested_cross_validation(classification_dir,
                                               X,y, groups,
                                               pipe,
                                               param_grid, scoring,
                                               compute_all_new, cv_out, cv_in,
                                               search_function,
                                               n_jobs=n_jobs)
        #************************************************************************
        
        print 'These are the learning results'
        print learning_results
            # get data groups
        data_groups = data.groupby(group_keys)
        groups = data_groups.groups.keys()

        for learning_result, group in zip(learning_results, groups):
                learning_result['group'] = group
                learning_result['group_keys'] = group_keys

        cv_object = learning_results

        report = cv.generate_classification_report(cv_object)
    
        report.to_hdf(classification_dir + 'classification_report.h5', '/report' )
        
        print report
    print 'Done!'

    
    
    
def plot_eda(label_struct, baseline_label_struct,
             pipe, scaler, param_grid,
             patient_list,
             feature_slot,
             hyper_param,
             plot_eda_all_new,
             learn_flag,
             compute_all_new,
            ):
    

    # State the parameters of the pipeline
    disk = '/mnt/Seagate/pre_epi_seizures/'
    baseline_files = 'h5_files/processing_datasets/baseline_datasets_new'
    seizure_files = 'h5_files/processing_datasets/seizure_datasets_new'

    lead_list = ['Ecg']

    interim_processing = [scaler]
    hist_bins = None
    dist = None
    flag_hist = False
    flag_andrews = False
    flag_series = True
    flag_box = False
    flag_pair = False
    assign_baseline = 'assign_equal_baseline_seizure'
    

    general_dir = disk + 'EDanalysis_new/'
    
    # choose data grouping 
    group_keys= ['patient_nr',
                 'seizure_nr',
                 'types_of_seizure',
                 'location']
    
    group_id = 'seizure_nr'
    
    # Get initial directory, for Exploratory Data Analysis    
    eda_dir = read_disk_space_hyper_param_results(directory=general_dir ,
                                                  patient_list = patient_list,
                                                  lead_list = lead_list,
                                                  scaler = scaler,
                                                  interim_processing = interim_processing,
                                                  assign_baseline = assign_baseline,
                                                  label_struct = label_struct,
                                                  baseline_label_struct = baseline_label_struct,
                                                  feature_slot=feature_slot, 
                                                  group_id=group_id,
                                                  hyper_param=hyper_param)
    
    print eda_dir
    
    if not os.path.exists(eda_dir)\
    and (feature_slot=='baseline_removal' or feature_slot=='hrv_computation'):
        print 'The path doesnt exist. Creating...'
        os.mkdir(eda_dir) 
        
    # Load the data, according to specification (loading made by convert pandas)
    data_struct = interim_process(disk, seizure_files, baseline_files,
                    feature_slot, hyper_param,
                    patient_list, lead_list,
                    label_struct, baseline_label_struct,
                    interim_processing)
    data = data_struct[0]
    features = data_struct[1]
    meta_features = data_struct[2]
    
    
    data_groups = data.groupby(group_keys)
    data_groups_list = list(data_groups)
    
    _plot_eda(eda_dir, data_groups_list,
              features,
              hist_bins, dist,
              flag_hist, flag_andrews,
              flag_series, flag_box,
              flag_pair)
    
    
    
    print 'Plotted!'
    
                        
def _plot_eda(directory, data_groups_list,
              features,
              hist_bins, dist,
              flag_hist, flag_andrews,
              flag_series, flag_box,
              flag_pair):
    
    path_to_save = directory
      
    """
    Plots and saves all the proposed plots for Exploratory data analysis to disk.
    If the files already exist, just show them.
    
    """
             
    # Loop for all groups of data
    for data_patient_seizure in data_groups_list:

        if flag_hist:
            plt_hist.histogram(path_to_save,
                               data_patient_seizure[1],
                               data_patient_seizure[0],
                               features,
                               'time_sample',
                               'patient_nr',
                               'seizure_nr',
                               'label',
                               'color',
                               bins=hist_bins,
                               dist=dist)


        if flag_series:
            plt_ts.time_series_plot(path_to_save, data_patient_seizure[1],
                                    features,
                                    'time_sample',
                                    'patient_nr',
                                    'seizure_nr',
                                    'label',
                                    'color')
        if flag_andrews:
            plt_and.andrews_curves(path_to_save,
                                   data_patient_seizure[1],
                                   data_patient_seizure[0],
                                   features,
                                   'time_sample',
                                   'patient_nr',
                                   'seizure_nr',
                                   'label',
                                   'color')
        if flag_box:
            plt_box.box_plot(path_to_save,
                             data_patient_seizure[1],
                             data_patient_seizure[0],
                             features,
                             'time_sample',
                             'patient_nr',
                             'seizure_nr',
                             'label',
                             'color')

        if flag_pair:
            plt_sc.pair_plot(path_to_save,
                             data_patient_seizure[1],
                             data_patient_seizure[0],
                             features,
                             'time_sample',
                             'patient_nr',
                             'seizure_nr',
                             'label',
                             'color')
            
            
            
def load_eda(label_struct, baseline_label_struct,
             scaler,
             patient_list,
             feature_slot,
             hyper_param
            ):
    
    # State the parameters of the pipeline
    disk = '/mnt/Seagate/pre_epi_seizures/'
    baseline_files = 'h5_files/processing_datasets/baseline_datasets_new'
    seizure_files = 'h5_files/processing_datasets/seizure_datasets_new'

    lead_list = ['ECG-']

    interim_processing = [scaler]
    hist_bins = None
    dist = None
    flag_hist = True
    flag_andrews = False
    flag_series = True
    flag_box = True
    flag_pair = True
    assign_baseline = 'assign_equal_baseline_seizure'
    

    general_dir = disk + 'EDanalysis_new/'
    
    # choose data grouping 
    group_keys= ['patient_nr',
                 'seizure_nr',
                 'types_of_seizure',
                 'location']
    
    group_id = 'seizure_nr'
    
    # Get initial directory, for Exploratory Data Analysis    
    eda_dir = read_disk_space_hyper_param_results(directory=general_dir ,
                                                  patient_list = patient_list,
                                                  lead_list = lead_list,
                                                  scaler = scaler,
                                                  interim_processing = interim_processing,
                                                  assign_baseline = assign_baseline,
                                                  label_struct = label_struct,
                                                  baseline_label_struct = baseline_label_struct,
                                                  feature_slot=feature_slot, 
                                                  group_id=group_id,
                                                  hyper_param=hyper_param)
    
    return _load_eda(eda_dir)
            
            
def _load_eda(eda_dir):
    
    path_to_save = eda_dir
    
    # if files in disk just show them        
    import os
    from IPython.display import Image
    a = [path_to_save + name for name in os.listdir(path_to_save) if name.endswith(".png")]
    
    print a
    return a

    
