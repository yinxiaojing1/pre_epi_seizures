# -------------------------------
import pandas as pd



# *********** NAIVE ++ CAREFUL: no particularites of data ******************************
def _create_trainig(data_record, feature_name, data_record_window,
                    label_record):

    # print ''
    # # print feature_names
    # # print data_record
    # # stop
    # print label_record
    # stop
    # print 'oplaa'
    # print data_record
    # print feature_name
    # print data_record_window
    # print label_record
    # stop
    training_data = pd.DataFrame(data_record.T, columns=feature_name['feature_legend'])

    training_data['sample_domain'] = data_record_window

    training_data['labels'] = label_record

    # stop

    return training_data


def create_training(data_record_list, feature_names, data_record_window_list,
                    label_record_list):
    # print data_record_list
    # print feature_names
    # print data_record_window_list
    # print label_record_list

    # stop

    training_data = pd.concat([_create_trainig(data_record, feature_name,
                                               data_record_window, label_record)
                              for data_record, feature_name, data_record_window, label_record,\
                              in zip(data_record_list, feature_names, data_record_window_list,
                                     label_record_list)])

    return training_data
