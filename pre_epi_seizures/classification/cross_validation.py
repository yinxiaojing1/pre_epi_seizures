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
#from supervised_new import *
#from prep_input_supervised import *
#from plot_classification import *
#from supervised_new import *
#from cross_validation import *
from save_for_class import *


def nested_cross_validation(full_path,
                           X,y, groups,
                           pipe,
                           param_grid, scoring,
                           compute_all_new, cv_out, cv_in):
    
    # Outer-loop cross-validation
    cv_out = cv_out.split(X, y, groups=groups)

    # Employ crossvalidation for each partion
    # according to cv_out
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


def _nested_cross_validation(full_path,
                             X,y, groups,
                             pipe, 
                             train,
                             test,
                             i,
                             param_grid, scoring,
                             compute_all_new, cv_in):
    
    return_struct = {}
    
    # Hyper Parameter optimization---------------------
    # 1. Hyper parameter grid-search optimization
    print 'CLFFFFFFFF'
    name_clf = 'clf__%s.h5' %i
    full_path_clf = full_path + name_clf
    clf, mdata = hyper_parameter_optimization(full_path_clf,
                                              compute_all_new,
                                              X,y, groups,
                                              pipe, 
                                              train,
                                              param_grid,
                                              scoring,
                                              cv_in)

    print 'HANDDLLLLEEEEEEEEE'
    # 2. Handle optimization results 
    name_hp = 'hp_opt_results__%s.h5' %i
    full_path_hp = full_path + name_hp
    hp_opt_results = handle_optimization(full_path_hp,
                                         compute_all_new,
                                         clf)
    return_struct['cv_results'] = hp_opt_results
    return_struct['clf'] = clf
    #-------------------------------------------------
    
    print 'TESSSSSSSSSSSST'
    # Model Test-------------------------------------- 
    # ROC (Reciever-Operator-Characteristic)
    #name_ROC = 'ROC__%s.h5' %i
    #full_path_ROC = full_path + name_ROC
    #ROC = _compute_ROC(full_path_ROC,
    #                   compute_all_new,
    #                   X, y,
    #                   test,
    #                   clf)
    #return_struct['ROC'] = ROC
    # -------------------------------------------------

    return (clf, test)


# Design Pattern
def get_cv_result(func):
    
    def call(full_path, 
             compute_all_new,
             *args, **kwargs):

         # Try to load file from disk 
        results, mdata = load_pandas_file_h5(full_path)
        
        # If empty compute result and save
        if results.empty and compute_all_new:
            results, mdata = func(full_path,
                                  compute_all_new,
                                  *args, **kwargs)
            
            print results
            h5store(full_path, results, **mdata)
               
        return results, mdata
               
    return call
                                 

@get_cv_result    
def hyper_parameter_optimization(full_path,
                                 compute_all_new,
                                 X,y, groups,
                                 optimization_pipe,
                                 train,
                                 param_grid, scoring,
                                 cv_in):
    
    # get inner data
    groups_inner = groups.iloc[train] # get data seizures
    X_inner = X.iloc[train] # get data features
    y_innner = y.iloc[train] # data labels

    # Grid search for Hyperparameters
    cv_inner = cv_in.split(X_inner, y_innner, groups=groups_inner) # get iterator for cv
    
    clf = GridSearchCV(optimization_pipe,
                       param_grid, scoring=scoring,
                       n_jobs=1, verbose=1,
                       cv=cv_inner,
                       return_train_score=True,
                       refit=scoring[0],
                       error_score=0)
    
    clf.fit(X_inner, y_innner) # retrain
    mdata = {}
    
    return clf, mdata


@get_cv_result        
def handle_optimization(full_path,
                        compute_all_new,
                        clf):
    
    # Get optimization results from clf struct    
    hp_opt_results = pd.DataFrame(clf.cv_results_)
    mdata = clf.best_params_
    
    return hp_opt_results, mdata
   

@get_cv_result    
def _compute_ROC(full_path,
                 compute_all_new,
                 X, y,
                 test, clf):
    # get data
    X_test, y_test = X.iloc[test], y.iloc[test]
    
    
    # evaluate data
    try:
        y_score = clf.decision_function(X_test)    
    except Exception as e:
        print e
        y_score = clf.predict_proba(X_test)[:,1]
        
    
    # Get results
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    ROC = pd.DataFrame(data=np.asarray([fpr, tpr, thresholds]).T,
                           columns=['FPR', 'TPR', 'thresholds'])
    mdata = {}
    
    return ROC, mdata


