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
    # This procedure is required to determine the path where the files reside.
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
    

    eda_dir = 'EDAnalysis/'
    
    eda_id = iopes.read_only_table(disk=disk,
                                   eda_dir=eda_dir,
                                   patient_list = patient_list,
                                   lead_list = lead_list,
                                   scaler = scaler,
                                   interim_processing = interim_processing,
                                   hist_bins = hist_bins,
                                   dist = dist,
                                   assign_baseline = assign_baseline,
                                   label_struct = label_struct,
                                   baseline_label_struct = baseline_label_struct,
                                   feature_slot=feature_slot, 
                                   hyper_param=0)

    
    path = disk + eda_dir + eda_id + '/'
    print path
    
    group_id = 'seizure_nr'
    label = 'label'

    # define cross-validation strategy 
    cv_out = LeavePGroupsOut(n_groups=1)
    cv_in = LeavePGroupsOut(n_groups=1)

    # choose scoring
    scoring = ['f1_micro']

    search_function = GridSearchCV

    pipe_steps = [step[0] for step in pipe.steps]
    
    clf_id = iopes.read_only_table(disk=disk,
                                   eda_dir=eda_dir + eda_id + '/',
                                   pipe = str(pipe_steps),
                                   param_grid = param_grid,
                                   cv_out = cv_out,
                                   cv_in = cv_in,
                                   scoring = scoring,
                                   search_function = search_function,
                                   group_id=group_id,
                                   label=label)

    path_to_save = disk + eda_dir + eda_id + '/' + clf_id + '/'
    
    # After determined path_to_save, now load all the files
    # from hyperparameter optimization
    hyper_parameterization_results = [load_optimization_test_file(path_to_save + name)
                                      for name in os.listdir(path_to_save)
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
    

    eda_dir = 'EDAnalysis/'
    
    
    

    eda_id = iopes.read_only_table(disk=disk,
                                        eda_dir=eda_dir,
                                        patient_list = patient_list,
                                        lead_list = lead_list,
                                        scaler = scaler,
                                        interim_processing = interim_processing,
                                        hist_bins = hist_bins,
                                        dist = dist,
                                        assign_baseline = assign_baseline,
                                        label_struct = label_struct,
                                        baseline_label_struct = baseline_label_struct,
                                        feature_slot=feature_slot, 
                                        hyper_param=0)
    
    
    path = disk + eda_dir + eda_id + '/'
    print path


    
    group_id = 'seizure_nr'
    label = 'label'


    # define cross-validation strategy 
    cv_out = LeavePGroupsOut(n_groups=1)
    cv_in = LeavePGroupsOut(n_groups=1)

    # choose scoring
    scoring = ['f1_micro']


    search_function = GridSearchCV

    pipe_steps = [step[0] for step in pipe.steps]
    
    clf_id = iopes.read_only_table(disk=disk,
                                       eda_dir=eda_dir + eda_id + '/',
                                       pipe = str(pipe_steps),
                                       param_grid = param_grid,
                                       cv_out = cv_out,
                                       cv_in = cv_in,
                                       scoring = scoring,
                                       search_function = search_function,
                                       group_id=group_id,
                                       label=label)

    path_to_save = disk + eda_dir + eda_id + '/' + clf_id + '/'
    
    file = pd.read_hdf(path_to_save + 'classification_resport.h5', '/report' )
    
    return file
    
    
    

def supervised_pipeline(label_struct, baseline_label_struct,
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
    

    eda_dir = 'EDAnalysis/'
    
    str_id = iopes.generate_string_identifier(patient_list = patient_list,
                                        lead_list = lead_list,
                                        scaler = scaler,
                                        interim_processing = interim_processing,
                                        hist_bins = hist_bins,
                                        dist = dist,
                                        assign_baseline = assign_baseline,
                                        label_struct = label_struct,
                                        baseline_label_struct = baseline_label_struct,
                                        feature_slot=feature_slot, 
                                        hyper_param=0)
    print str_id
    
    stop

    eda_id = iopes.get_eda_params_path(disk=disk,
                                        eda_dir=eda_dir,
                                        )
    path = disk + eda_dir + eda_id + '/'


    # In[3]:
    # Presumably the files exist
    import os
    if not os.path.exists(path):
        os.mkdir(path)


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
                                                   feature_slot)[0]

    print feature_name

    baseline_data = cv_pd.convert_to_pandas(path_to_load, path_to_map,
                            patient_list, feature_name,
                            lead_list, baseline_label_struct)

    baseline_data



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

    # Group the data
    group_keys= ['patient_nr',
                  'seizure_nr',
                  'types_of_seizure',
                  'location']
    
    data_groups = data.groupby(group_keys)
    group_id = 'seizure_nr'

    label = 'label'

    data_groups_list = list(data_groups)

    
    # prepare data for classification - watch out for memory concerns
    X = data[features]
    y = data[label]
    groups = data[group_id]


    # define cross-validation strategy 
    cv_out = LeavePGroupsOut(n_groups=1)
    cv_in = LeavePGroupsOut(n_groups=1)

    # choose scoring
    scoring = ['f1_micro']


    search_function = GridSearchCV
    
    pipe_steps = [step[0] for step in pipe.steps]

    

    clf_id = iopes.get_eda_params_path(disk=disk,
                                       eda_dir=eda_dir + '/' + eda_id + '/' ,
                                       pipe = str(pipe_steps),
                                       param_grid = param_grid,
                                       cv_out = cv_out,
                                       cv_in = cv_in,
                                       scoring = scoring,
                                       search_function = search_function,
                                       group_id=group_id,
                                       label=label)

    path_to_save = disk + eda_dir + eda_id + '/' + clf_id + '/'



    import classification.eda.hist as plt_hist
    import classification.eda.andrews as plt_and
    import classification.eda.series as plt_ts
    import classification.eda.box as plt_box
    import classification.eda.scatter as plt_sc
    import matplotlib.pyplot as plt

    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)



    if plot_eda_all_new:

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

    else:
        import os
        from IPython.display import Image
        a = [name for name in os.listdir(path_to_save) if name.endswith(".png")]
        for image in a:
            display(Image(filename=path_to_save + image))

    import classification.cross_validation as cv

    # ***********************************Learning****************************
    # Learn from data_struct using nested cross_validation
    # learninig is an optimization and respective test results
    # for each partition of the dataset according to cv_out
    


    if learn_flag:
        learning_results = cv.nested_cross_validation(path_to_save,
                                               X,y, groups,
                                               pipe,
                                               param_grid, scoring,
                                               compute_all_new, cv_out, cv_in,
                                               search_function)
        #************************************************************************
        groups = data_groups.groups.keys()

        for learning_result, group in zip(learning_results, groups):
                learning_result['group'] = group
                learning_result['group_keys'] = group_keys

        cv_object = learning_results

        report = cv.generate_classification_report(cv_object)
        report.to_hdf(path_to_save + 'classification_resport.h5', '/report' )
        
        print report
    print 'Done!'
    
    
def dask_supervised_pipeline(label_struct, baseline_label_struct,
                        pipe, scaler, param_grid,
                        patient_list,
                        feature_slot,
                        hyper_param,
                        plot_eda_all_new,
                        learn_flag,
                        compute_all_new,
                        dask_client
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
    

    eda_dir = 'EDAnalysis/'

    eda_id = iopes.get_eda_params_path(disk=disk,
                                        eda_dir=eda_dir,
                                        patient_list = patient_list,
                                        lead_list = lead_list,
                                        scaler = scaler,
                                        interim_processing = interim_processing,
                                        hist_bins = hist_bins,
                                        dist = dist,
                                        assign_baseline = assign_baseline,
                                        label_struct = label_struct,
                                        baseline_label_struct = baseline_label_struct,
                                        feature_slot=feature_slot, 
                                        hyper_param=0)
    path = disk + eda_dir + eda_id + '/'


    # In[3]:
    # Presumably the files exist
    import os
    if not os.path.exists(path):
        os.mkdir(path)


    # Ingest Seizure Data
    path_to_load = disk + seizure_files + '.h5'
    path_to_map = disk + seizure_files + '_map.txt'

    # Feature group to analyse -- point of entry
    feature_name = get_feature_group_name_list(path_to_map,
                                                   feature_slot)[hyper_param]

    print feature_name


    seizure_data = cv_pd.convert_to_pandas(path_to_load, path_to_map,
                            patient_list, feature_name,
                            lead_list, label_struct)
    seizure_data



    # Ingest Baseline Data

    # set Labeling structure
    path_to_load = disk + baseline_files + '.h5'
    path_to_map = disk + baseline_files + '_map.txt'

    # Feature group to analyse -- point of entry
    feature_name = get_feature_group_name_list(path_to_map,
                                                   feature_slot)[0]

    print feature_name

    baseline_data = cv_pd.convert_to_pandas(path_to_load, path_to_map,
                            patient_list, feature_name,
                            lead_list, baseline_label_struct)

    baseline_data



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

    # Group the data
    group_keys= ['patient_nr',
                  'seizure_nr',
                  'types_of_seizure',
                  'location']
    
    data_groups = data.groupby(group_keys)
    group_id = 'seizure_nr'

    label = 'label'

    data_groups_list = list(data_groups)

    # prepare data for classification - watch out for memory concerns
    X = data[features]
    y = data[label]
    groups = data[group_id]
    
    # Scatter the data on the cluster
    dask_client.scatter(X)


    # define cross-validation strategy 
    cv_out = LeavePGroupsOut(n_groups=1)
    cv_in = LeavePGroupsOut(n_groups=1)

    # choose scoring
    scoring = ['f1_micro']

    from dask_ml.model_selection import GridSearchCV as grid_search_dask
    search_function = grid_search_dask
    
    
    pipe_steps = [step[0] for step in pipe.steps]


    clf_id = iopes.get_eda_params_path(disk=disk,
                                       eda_dir=eda_dir + '/' + eda_id + '/' ,
                                       pipe = str(pipe_steps),
                                       param_grid = param_grid,
                                       cv_out = cv_out,
                                       cv_in = cv_in,
                                       scoring = scoring,
                                       search_function = search_function,
                                       group_id=group_id,
                                       label=label)

    path_to_save = disk + eda_dir + eda_id + '/' + clf_id + '/'


    # In[12]:



    import classification.eda.hist as plt_hist
    import classification.eda.andrews as plt_and
    import classification.eda.series as plt_ts
    import classification.eda.box as plt_box
    import classification.eda.scatter as plt_sc
    import matplotlib.pyplot as plt

    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)



    if plot_eda_all_new:

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

    else:
        import os
        from IPython.display import Image
        a = [name for name in os.listdir(path_to_save) if name.endswith(".png")]
        for image in a:
            display(Image(filename=path_to_save + image))

    import classification.cross_validation as cv

    # ***********************************Learning****************************
    # Learn from data_struct using nested cross_validation
    # learninig is an optimization and respective test results
    # for each partition of the dataset according to cv_out
    


    if learn_flag:
        learning_results = dask_client.compute(cv.nested_cross_validation(path_to_save,
                                                                          X,y, groups,
                                                                          pipe,
                                                                          param_grid, scoring,
                                                                          compute_all_new, cv_out, cv_in,
                                                                          search_function))
        #************************************************************************
        groups = data_groups.groups.keys()
        
        return learning_results

        for learning_result, group in zip(learning_results, groups):
                learning_result['group'] = group
                learning_result['group_keys'] = group_keys

        cv_object = learning_results

        report = cv.generate_classification_report(cv_object)
        report.to_hdf(path_to_save + 'classification_resport.h5', '/report' )
        
        print report
    print 'Done!'