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
                           compute_all_new, cv_out, cv_in,
                           search_function):
    
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
                             compute_all_new, cv_in,
                             search_function)
              for i, (train, test) in enumerate(cv_out)]
 
    return struct


def _nested_cross_validation(full_path,
                             X,y, groups,
                             pipe, 
                             train,
                             test,
                             i,
                             param_grid, scoring,
                             compute_all_new, cv_in,
                             search_function):
    
    return_struct = {}
    
    # Completely new trainig
    if compute_all_new:
        # Hyper Parameter optimization---------------------
        # 1. Hyper parameter grid-search optimization
        name_clf = 'clf__%s.h5' %i
        full_path_clf = full_path + name_clf
        clf, mdata = hyper_parameter_optimization(full_path_clf,
                                                  compute_all_new,
                                                  X,y, groups,
                                                  pipe, 
                                                  train,
                                                  param_grid,
                                                  scoring,
                                                  cv_in,
                                                  search_function)
        
        # 2. Test the Best Model
        X_test, y_test = X.iloc[test], y.iloc[test]
        y_pred = clf.predict(X_test)
        
        
        # 3. Compose a predefined return structure 
        return_struct['cv_results'] = pd.DataFrame(clf.cv_results_)
        return_struct['y_test'] = y_test
        return_struct['y_pred'] = y_pred
        return_struct['best_estimator'] = str(clf.best_estimator_)
        return_struct['best_params'] = clf.best_params_
        
        
        # 4. Save Results to disk
        name_hp = 'hp_opt_results__%s.h5' %i
        full_path_hp = full_path + name_hp
        h5store(full_path_hp, return_struct['cv_results'],
                **dict((k,return_struct[k]) 
                       for k in return_struct.keys()
                       if 'results' not in k))

     # Load Training from disk
    if not compute_all_new:
        
        
        # 5. Load Results from disk
        

        # 2. Handle optimization results 
        name_hp = 'hp_opt_results__%s.h5' %i
        full_path_hp = full_path + name_hp
        results, mdata = load_pandas_file_h5(full_path_hp)
        
        print 'These are the saved results'
        print results
        
        print 'This is the saved metadata'
        print mdata
        
        
        return_struct = mdata
        return_struct['cv_results'] = results


    #-------------------------------------------------
    

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
    
    return return_struct


# Design Pattern
def get_cv_result(func):
    
    def call(full_path, 
             compute_all_new,
             *args, **kwargs):

         # Try to load file from disk 
        results, mdata = load_pandas_file_h5(full_path)
        
        print 'Loaded Structres... see for yourself'
        
        print results
        print mdata
        
        # If empty compute result and save
        if results.empty and compute_all_new:
            
            print 'Loaded Results are empty'
            print ''
            results, mdata = func(full_path,
                                  compute_all_new,
                                  *args, **kwargs)
            
            print 'Saving the optimization results to disk ....'
            print ''
            print results
            print mdata
            h5store(full_path, results, **mdata)
               
        return results, mdata
               
    return call
                                 

#@get_cv_result    
def hyper_parameter_optimization(full_path,
                                 compute_all_new,
                                 X,y, groups,
                                 optimization_pipe,
                                 train,
                                 param_grid, scoring,
                                 cv_in,
                                 search_function):
    
    # get inner data
    groups_inner = groups.iloc[train] # get data seizures
    X_inner = X.iloc[train] # get data features
    y_innner = y.iloc[train] # data labels

    # search for Hyperparameters
    cv_inner = cv_in.split(X_inner, y_innner, groups=groups_inner) # get iterator for cv
    
    clf = search_function(optimization_pipe,
                       param_grid, scoring=scoring,
                       n_jobs=1, verbose=1,
                       cv=cv_inner,
                       return_train_score=True,
                       refit=scoring[0],
                       error_score=0)
    
    clf.fit(X_inner, y_innner) # retrain
    mdata = {}
    
    print 'ready to return optimization objects'
    return clf, mdata


def handle_optimization(full_path,
                        
                        compute_all_new,
                        clf):
    
    # Get optimization results from clf struct    
    hp_opt_results = pd.DataFrame(clf.cv_results_)
    best_params = clf.best_params_
    named_setps = named_steps.keys()
    
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


