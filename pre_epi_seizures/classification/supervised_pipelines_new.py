# 3party
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import LeavePGroupsOut
from sklearn.model_selection import GridSearchCV
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




path_to_save = '/Volumes/ASSD/pre_epi_seizures/h5_files/classification_datasets'
class_metadata = ['labels', 'group', 'sample_domain']


def get_data_struct(pt):
    # Select Patient to analyze
    # pt = 3
    global path_to_save
    path_to_save += '/patient_' + str(pt)
    data_struct = load_feature_groups_baseline_seizure_per_patient(
                                                    patient_nr=pt
                                                    )
    return data_struct


def make_dir(dir_):
    if not os.path.exists(dir_):
        print 'MAKING'
        os.makedirs(dir_)



def general(file_to_save, down, up, data_struct):



    interval = [(down*60, up*60)]


    data_struct = prep_input_supervised_baseline_seizure(data_struct, interval).reset_index()
    data_struct = data_struct.drop(['index'], axis=1)

    # print data_struct
    points_per_label = data_struct['labels'].reset_index().groupby(['labels']).count()

    points_per_label = list(dict(points_per_label)['index'])

    # stop

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

    # best_params = LOGO(X, y, file_to_save,
    #                    data_struct, pipe,
    #                    param_grid, scoring,
    #                    feature_names, class_metadata,
    #                    model,
    #                    compute_all_new)


    best_params = pd.concat([hyper_parameter_optimization_LOGO(file_to_save,
                                         data_struct,
                                         train, test,
                                         pipe, param_grid,
                                         'accuracy', i,
                                         feature_names,
                                         class_metadata,
                                         model,
                                         compute_all_new)
                    for i, (train, test) in enumerate(cv)])

    # path_hist = file_to_save_plots + '/hist/'
    # make_dir(path_hist)
    # plot_hist(path_hist, interval, points_per_label, data_struct, class_metadata)

    path_ROC = file_to_save_plots + '/ROC/'
    make_dir(path_ROC)
    plot_roc(path_ROC, interval, points_per_label, best_params)


def hyper_parameter_optimization_LOGO(file_to_save,
                                 data_struct, train,
                                 test, optimization_pipe,
                                 param_grid, scoring, nr_seizure,
                                 feature_names, class_metadata,
                                 model_name,
                                 compute_all_new):

    # inner Parameter Tuning (LOGO)
    data_struct_inner = data_struct.loc[train] # get training data
    groups_inner = data_struct_inner['group'] # get data seizures
    X_inner = data_struct_inner.drop(class_metadata, axis=1) # get data features
    y_innner = data_struct_inner['labels'] # data labels
    lpgo = LeavePGroupsOut(n_groups=1) # divde data into training-test set (LOGO)
    cv_inner = lpgo.split(X_inner, y_innner, groups=groups_inner) # get iterator for cv

    filename = get_name_to_save(file_to_save, nr_seizure, feature_names, model_name)

    try:
        # try to load Hyperparameter optimization result
        if not compute_all_new:
            results = pd.read_hdf(file_to_save + filename + '.h5', 'test') 
            print 'Optimization Already in disk!'
            return results
        else:
            stop

    except Exception as e:
        print e
        # Grid search for Hyperparameters
        clf = GridSearchCV(optimization_pipe,
                           param_grid, scoring=scoring,
                           n_jobs=-1, verbose=1,
                           cv=cv_inner)

        # # Retrain
        clf.fit(X_inner, y_innner)

        # Evaluate
        X_test = data_struct.loc[test]
        y_test = X_test['labels']
        X_test = X_test.drop(class_metadata, axis=1)

        y_score = clf.decision_function(X_test)

        # Metrics
        fpr, tpr, thresholds = roc_curve(y_test, y_score)

        model_params = clf.best_params_


        results = pd.DataFrame(data=np.asarray([fpr, tpr, thresholds]).T,
                               columns=['FPR', 'TPR', 'thresholds'])

        results['model'] = [str(model_params)] * len(fpr)
        results['nr_seizure'] = [nr_seizure] * len(fpr)

        # # Save results
        results.to_hdf(file_to_save + filename + '.h5', 'test', format='f', mode='w')
        return results


data_struct = get_data_struct(pt = 3)

for step_min in xrange(5, 50, 5):

    file_to_save = path_to_save + '/step_' + str(step_min) + '/'

    if not os.path.exists(file_to_save):
        print 'MAKING'
        os.makedirs(file_to_save)

    for up, down in zip(xrange(step_min, 50, step_min), xrange(0, 50 - step_min, step_min)):
        general(file_to_save, down, up, data_struct)