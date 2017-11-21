''' Attempted Supervised classification atempt'''
# ------------------------------
from load_for_class import *
from ploting_temp import *
from metrics import *
# Classifiers ------------------------------
from sklearn import svm





# def load_data(path_to_load,
#                        feature_name_to_load_list):

#     # Load baseline data and time domain in context of
#     # acquisition
#     load_data = load_features_from_input_list(
#                                   path_to_load,
#                                   feature_name_to_load_list)

#     load_time_window = load_window_features_from_input_list(
#                                   path_to_load,
#                                   feature_name_to_load_list)

#     return load_data, load_time_window

def general_training_crossvalidation(model):

    def decorated_function(training_data, test_data, features, **model_parameters):
        # training
        untrained_model = model(**model_parameters)
        trained_model = untrained_model.fit(training_data[features], training_data['labels'])

        # validation
        prediction = trained_model.predict(test_data[features])
        model_parameters = get_f1_precision_recall(prediction, test_data['labels'], model_parameters)
        return model_parameters

    return decorated_function


@general_training_crossvalidation
def rbf_svc_sklearn(**model_parameters):
    return svm.SVC(kernel='rbf',
                   gamma=model_parameters['gamma'],
                   C=model_parameters['C'])

# def train_from_model(untrained_model,
#                       training_data,
#                       training_labels):
#     untrained_model.fit(training_data, training_labels)
#     return untrained_model


# def validate_from_model(trained_model, 
#                          validation_data,
#                          validation_labels):
#     prediction = trained_class_object.predict(validation_data)
#     error_validation = validation_labels - prediction
#     return error_validation






# def classify_supervised(func, data, labels):

#     def classify(data, labels):
#         func(data, labels)