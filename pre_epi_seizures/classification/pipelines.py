from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn import preprocessing
from sklearn import neural_network


def std_scaler_svm():
    pipe_id = []
    
    # body of pipeline
    pipe_id.append(('STD_SCALER', preprocessing.StandardScaler()))
    pipe_id.append(('SVC', svm.SVC()))
    
    # create pipeline object with all of the above steps
    pipe = Pipeline(steps = pipe_id)
    
    # choose parameter search method *coherent with Pipeline steps
    param_grid = [{'SVC__kernel': ['rbf'],
                   'SVC__gamma': [2**i for i in xrange(-15, -1)],
                   'SVC__C': [2**i for i in xrange(-5, 11)]},
                  ]
    
    return pipe, param_grid


def minmax_scaler_svm():
    pipe_id = []
    
    # body of pipeline
    pipe_id.append(('MINMAX_SCALER', preprocessing.MinMaxScaler(feature_range=(0, 1))))
    pipe_id.append(('SVC', svm.SVC()))
    
    # create pipeline object with all of the above steps
    pipe = Pipeline(steps = pipe_id)
    
    # choose parameter search method *coherent with Pipeline steps
    param_grid = [{'SVC__kernel': ['rbf'],
                   'SVC__gamma': [2**i for i in xrange(-15, -1)],
                   'SVC__C': [2**i for i in xrange(-5, 11)]},
                  ]
    
    return pipe, param_grid


def std_scaler_nn():
    pipe_id = []
    
    # body of pipeline
    pipe_id.append(('STD_SCALER', preprocessing.StandardScaler()))
    pipe_id.append(('NN', neural_network.MLPClassifier()))
    
    # create pipeline object with all of the above steps
    pipe = Pipeline(steps = pipe_id)
    
    # choose parameter search method *coherent with Pipeline steps
    param_grid = [{'NN__activation': ['relu']}]
    
    return pipe, param_grid


def minmax_scaler_nn():
    pipe_id = []
    
    # body of pipeline
    pipe_id.append(('MINMAX_SCALER', preprocessing.MinMaxScaler(feature_range=(0, 1))))
    pipe_id.append(('NN', neural_network.MLPClassifier()))
    
    # create pipeline object with all of the above steps
    pipe = Pipeline(steps = pipe_id)
    
    # choose parameter search method *coherent with Pipeline steps
    param_grid = [{'NN__activation': ['relu'],
                   'NN__solver':['adam']}]
    
    return pipe, param_grid

