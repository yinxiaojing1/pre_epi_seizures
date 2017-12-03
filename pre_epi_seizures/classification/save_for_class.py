def get_name_to_save(path_to_save, seizure_nr, feature_names, model):
    test_set = '-||| Test set: ' + str(seizure_nr) + '|||-'
    feature_names = '-||| Feature Names: ' + str(feature_names) + '|||-'
    model = '-||| Model: ' + str(model) + '|||-'
    return test_set + feature_names + model
