# 3party
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import LeavePGroupsOut
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn import svm
from sklearn import preprocessing
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np

# python
import os

# classification packages-------------
from supervised_new import *
from prep_input_supervised import *
from plot_classification import *
from supervised_new import *
from parameter_search import *
from cross_validation import *
from save_for_class import *
# from plot_r_classification import *


path_to_save = '/Volumes/ASSD/pre_epi_seizures/h5_files/classification_datasets'
class_metadata = ['labels', 'group', 'sample_domain']


def get_data_struct(pt, features):
    # Select Patient to analyze
    # pt = 3
    global path_to_save
    path_to_save += '/patient_' + str(pt)
    data_struct = load_feature_groups_baseline_seizure_per_patient(
                                                    patient_nr=pt, 
                                                    feature_slot=features
                                                    )
    return data_struct


def make_dir(dir_):
    if not os.path.exists(dir_):
        print 'MAKING'
        os.makedirs(dir_)


def create_path_to_save(params):
    global path_to_save
    path_to_save += str(params)
    return path_to_save


def get_data(params):
    # Retrieve all data from the patient --Memory be careful
    pt_nr = params['pt_nr']
    data_struct = get_data_struct(3, params['features'])
    return data_struct


def get_data_frame(data, interval):
    # Transform retrieved data into Pandas DataFrame
    data_struct = prep_input_supervised_baseline_seizure(data, interval).reset_index()

    data_struct = data_struct.drop(['index'], axis=1)
    return data_struct


def get_balance_data_frame(data_struct):
    # retrieve the balance of the data points for analysis
    points_per_label = data_struct['labels'].reset_index().groupby(['labels']).count()
    points_per_label = list(dict(points_per_label)['index'])
    return points_per_label


def get_pipeline(params):
    pipeline = []

    for param in params.keys():
        if 'estimator' in param:
            for element in params[param]:
                pipeline.append(element)

    print pipeline
    pipeline = Pipeline(pipeline)

    return pipeline


def get_scoring(params):
    return params['scoring_opt']


def get_cross_validation(params):
    cv_out = params['cv_out']
    cv_in = params['cv_in']
    return cv_out, cv_in


def get_search_window_range(params, step):
    max_seizure_window = params['max_seizure_window']
    min_seizure_window = params['min_seizure_window']

    step = step * 60
    window_range_up = xrange(step,
                        max_seizure_window,
                        step)

    window_range_down = xrange(0,
                              max_seizure_window - step, step)

    window_range = zip(window_range_down, window_range_up)
    return window_range


def get_step_range(params):
    step_range = params['step']
    return step_range


def get_trial_range(params):
    nr_trials = params['nr_trials']
    return xrange(0, nr_trials)


def get_params_range(params):
    params_range = params['param_grid_model']
    return params_range


def create_contig_win_interval(down, up):
    interval = [(down*60, up*60)]
    return interval


def get_data_from_win(down_up_list):
    data_struct = prep_input_supervised_baseline_seizure(
                                    data_struct, down_up_list).reset_index()
    data_struct = data_struct.drop(['index'], axis=1)

    return data_struct


# def prop_classification(data, pipeline, cv_out, cv_in):



def general_supervised_patient_specific():

    params = dict()

    features = 'hrv_time_features#'
    params['features'] = features

    # Comment/Uncomment here to change output
    # Normalization chain
    std = preprocessing.StandardScaler()
    params['estimator_scaler'] = [('std', std)]

    # Model chain
    svc = svm.SVC()
    params['estimator_model'] = [('svc', svc)]

    # model parameter search
    param_grid_model = [{'svc__kernel': ['rbf'],
                   'svc__gamma': [2**i for i in xrange(-15, -1)],
                   'svc__C': [2**i for i in xrange(-5, 11)]},
                  ]
    params['param_grid_model'] = param_grid_model

    # CV
    lpgo = GroupKFold(n_splits=1)
    params['cv_out'] = lpgo

    # inner CV
    lpgo = LeavePGroupsOut(n_groups=1)
    params['cv_in'] = lpgo

    # Scoring - optimization
    acc = 'accuracy'
    params['scoring_opt'] = [('acc', acc)]

    # Nr_trials
    nr_trials = 4
    params['nr_trials'] = nr_trials

    # seizure window search
    step = [5, 10, 15, 20, 25]
    max_seizure_window = 50 * 60
    min_seizure_window = 0 * 60
    params['step'] = step
    params['max_seizure_window'] = max_seizure_window
    params['min_seizure_window'] = min_seizure_window

    # type baseline allocation
    bs_alloc = 'random_balanced'
    params['bs_alloc'] = bs_alloc

    # Label allocation

    # Number of patient
    pt_nr = 3
    params['pt_nr'] = [pt_nr]

    return params


def general(file_to_save, down, up, data_struct, trial):
    interval = [(down*60, up*60)]

    data_struct = prep_input_supervised_baseline_seizure(data_struct, interval).reset_index()
    data_struct = data_struct.drop(['index'], axis=1)

    # explore_r(data_struct)

    # stop
    # print data_struct
    points_per_label = data_struct['labels'].reset_index().groupby(['labels']).count()

    points_per_label = list(dict(points_per_label)['index'])

    feature_names = [name
                     for name in data_struct.columns
                     if name not in class_metadata]

    # prepare data for classification - watch out for memory concerns
    X = data_struct.drop(class_metadata, axis=1)
    y = data_struct['labels']
    groups = data_struct['group']

    # choose normalization criterion
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X) # apply
    X = pd.DataFrame(X, columns=feature_names)

    # choose estimator
    model = 'SVC'
    estimator = svm.SVC()

    # choose scoring
    scoring = 'accuracy'

    # -->**insert here cross-validation strategy ( Testing nested LOGO )
    groups = data_struct['group']
    lpgo = LeavePGroupsOut(n_groups=1)
    cv = lpgo.split(X, y, groups=groups)
    # lpgo

    # create pipeline object with all of the above steps
    pipe = Pipeline([('estimate', estimator)])

    # choose param grid for exhaustive search
    # param_grid = {'C': (1, 10, 100, 1000), 'gamma': (0.001, 0.0001), 'kernel': ('rbf')}
    param_grid = [{'estimate__kernel': ['rbf'],
                   'estimate__gamma': [2**i for i in xrange(-15, -1)],
                   'estimate__C': [2**i for i in xrange(-5, 11)]},
                  ]

    compute_all_new = True

    global path_to_save
    file_to_save_plots = file_to_save
    file_to_save = file_to_save + 'parameter_optimization' + '_' + str(interval) + '/'

    make_dir(file_to_save)

    # plot_scatter(path_to_save, data_struct, class_metadata)

    # plot_full(file_to_save, data_struct, class_metadata)

    best_params = LOGO(X, y, file_to_save,
                       data_struct, pipe,
                       param_grid, scoring,
                       feature_names, class_metadata,
                       model,
                       compute_all_new)

    path_ROC = file_to_save_plots + '/ROC/'
    make_dir(path_ROC)
    plot_roc(path_ROC, interval, points_per_label, best_params, trial)



# data_struct = get_data_struct(pt = 3)

# for step_min in xrange(5, 50, 5):

#     file_to_save = path_to_save + '/step_' + str(step_min) + '/'

#     if not os.path.exists(file_to_save):
#         print 'MAKING'
#         os.makedirs(file_to_save)

#     for up, down in zip(xrange(step_min, 50, step_min), xrange(0, 50 - step_min, step_min)):

#         for trial in xrange(0, 1):
#             general(file_to_save, down, up, data_struct, trial)


params = general_supervised_patient_specific()
data = get_data(params)

params_range = get_params_range(params)

step_range = get_step_range(params)
# stop
trials = get_trial_range(params)
# stop
pipeline = get_pipeline(params)

cv_out, cv_in = get_cross_validation(params)

scoring = get_scoring(params)

print cv_out, cv_in

for step in step_range:
    print step_range

    window_range = get_search_window_range(params, step)
    for down, up in window_range:

         # Classification prep
        interval = create_contig_win_interval(down, up)

        data_frame = get_data_frame(data, interval)
        for trial in trials:
            nested_cross_validation(data_frame, class_metadata,
                                    scoring,
                                    pipeline, params_range,
                                    cv_in, cv_out)