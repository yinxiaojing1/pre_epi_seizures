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



def LOGO(X, y, file_to_save,
       data_struct, pipe,
       param_grid, scoring,
       feature_names, class_metadata,
       model,
       compute_all_new):

    # -->**insert here cross-validation strategy ( Testing nested LOGO )
    groups = data_struct['group']
    lpgo = LeavePGroupsOut(n_groups=1)
    cv = lpgo.split(X, y, groups=groups)

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
    return best_params

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