from sklearn.metrics import *


def get_f1_precision_recall(prediction, labels, model_parameters):

    model_parameters['precision'] = precision_score(prediction, labels)
    model_parameters['recall'] = recall_score(prediction, labels)
    model_parameters['f1'] = f1_score(prediction, labels)


    return model_parameters