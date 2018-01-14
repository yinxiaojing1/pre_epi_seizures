# 3party
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import LeavePGroupsOut
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedKFold
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



def _nested_cross_validation(full_path,
                             X,y, groups,
                             pipe, 
                             train,
                             test,
                             i,
                             param_grid, scoring,
                             compute_all_new, cv_in):

    
    # get Hyper Parameter optimization
    clf = hyper_parameter_optimization(full_path,
                                     X,y, groups,
                                     pipe, 
                                     train,
                                     param_grid, scoring,
                                     compute_all_new, cv_in)


    # Handle optimization results
    hp_opt_results = handle_optimization(full_path, clf)    

    # Model Test
    test_results = model_test(full_path, 
                         clf, X, y,
                         test, i)

    return clf.best_params_, hp_opt_results, test_results


def nested_cross_validation(full_path,
                           X,y, groups,
                           pipe,
                           param_grid, scoring,
                           feature_names, class_metadata,
                           model,
                           compute_all_new, cv_out, cv_in):
    
    # Outer-loop cross-validation
    cv_out = cv_out.split(X, y, groups=groups)

    # cross-validation loop
    cv_results = []
    #clf_struct = {}
    cv_test = []

    struct = [_nested_cross_validation(full_path,
                             X,y, groups,
                             pipe, 
                             train,
                             test,
                             i,
                             param_grid, scoring,
                             compute_all_new, cv_in)
              for i, (train, test) in enumerate(cv_out)]
 
        
        
    
    return struct
        
        
def handle_optimization(full_path, clf):
    # try to load cv_results
    try:
        # Try to load from disk
        if not compute_all_new:
            print 'Loading Hyper-parameters......'
            hpt_opt_results = h5load(full_path + '**cv_results.h5')

    except Exception as e:
        
        print 'Optimizing Hyper-parameters......'
    
        # Get optimization results    
        hp_opt_results = pd.DataFrame(clf.cv_results_)
    
        # Save optimization results
        h5store(full_path + '**cv_results.h5', hp_opt_results, **{})
        
    
    return hp_opt_results



def nested_cross_validation_before(full_path,
                                   X,y, groups,
                                   pipe,
                                   param_grid, scoring,
                                   feature_names, class_metadata,
                                   model,
                                   compute_all_new, cv_out, cv_in, trial):

        nested_cross_validation(X,y, groups,
                             train, test,
                             pipe,
                             param_grid, scoring,
                             compute_all_new, cv_out, cv_in, trial)
        
        
        data, metadata = h5load(full_path + '**cv_results.h5')
        print data
        print metadata
       

        return best_params

    
def hyper_parameter_optimization(full_path,
                                 X,y, groups,
                                 pipe,
                                 train,
                                 param_grid, scoring,
                                 compute_all_new, cv_in):
    try:
        # Try to load from disk
        if not compute_all_new:
            clf = h5loadmodel(full_path + '**cv_results.h5')
            

    except Exception as e: 
        # compute new Hyper Paramter optimization
        clf = _hyper_parameter_optimization(full_path,
                                           X,y, groups,
                                           train,
                                           pipe,
                                           param_grid, scoring,
                                           compute_all_new, cv_in)
    
    return clf
        

def _hyper_parameter_optimization(file_to_save,
                                 X,y, groups,
                                 train,
                                 optimization_pipe,
                                 param_grid, scoring,
                                 compute_all_new, cv_in):
    
    # inner Parameter Tuning
    groups_inner = groups.iloc[train] # get data seizures
    X_inner = X.iloc[train] # get data features
    y_innner = y.iloc[train] # data labels

    # Fetch generator for inner cross-validation strategy
    cv_inner = cv_in.split(X_inner, y_innner, groups=groups_inner) # get iterator for cv

    # Grid search for Hyperparameters
    clf = GridSearchCV(optimization_pipe,
                       param_grid, scoring=scoring,
                       n_jobs=1, verbose=1,
                       cv=cv_inner,
                       refit='AUC')
    
    # Retrain
    clf.fit(X_inner, y_innner)
    
    return clf
    

    
def model_test(file_to_save, 
               clf, X, y,
               test, i):
    
    # Evaluate
    X_test, y_test = X.iloc[test], y.iloc[test]
    y_score = clf.decision_function(X_test)

    # Metrics
    fpr, tpr, thresholds = roc_curve(y_test, y_score)

    # Model Params
    model_params = clf.best_params_

    
    results = pd.DataFrame(data=np.asarray([fpr, tpr, thresholds]).T,
                           columns=['FPR', 'TPR', 'thresholds'])

    test_results = results
    return test_results


def temp():
    
    # Evaluate
    X_test, y_test = X.iloc[test], y.iloc[test]
    y_score = clf.decision_function(X_test)

