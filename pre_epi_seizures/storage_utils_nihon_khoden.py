import sys 

import os

import numpy as np

import pyedflib

from logging_utils.formatter_logging import logger

from storage_utils.storage_utils_hdf5 import  load_signal, save_signal


# @profile
def converter(patient_list, disk):
    path_list = get_file_list(patient_list, disk)
    # print path_list
    for path in path_list:

        name_file = path[path.index('F'): path.index('F') + 8]
        name_patient = path[path.index('P'): path.index('P') + 9]

        print name_patient
        print 
        print name_file
        # stop
        # stop
        edf_file = pyedflib.EdfReader(path)
        # print edf_file
        # stop
        ecg_signals = load_ecg_signal(edf_file)
        signals = ecg_signals[0]
        names = ecg_signals[1]
        date_time = ecg_signals[2]

        # stop

        names_files = [name_file + '_' + name + '_' + date_time
                       for name in names]
        print names_files

        # stop
        path = disk + 'h5_files/raw_fulldata/HSM_data.h5'
        # stop
        save_signal(path=path, signal_list= signals, mdata_list=['']*len(signals), name_list=names_files, group_list=[name_patient])


        # stop
        # logger.debug("the mdata is gr%s", mdata)
        # name = path[18:26]
        # name_patient = path[5:13]

        # if mdata['crysis_time_seconds']:
        #     group = '/'+name_patient +'/crysis'
        # else: 
        #     group = '/'+name_patient +'/free'
        # print 'Saving ...'
        # save_signal(path='/home/sargo/Desktop/HSM_data.h5', signal_list=[signal], mdata_list=[mdata], name_list=[name], group_list=[group])
        edf_file._close()

def create_mdata(header, crysis_time_seconds):
    header['crysis_time_seconds'] = list(crysis_time_seconds)
    return header


def load_ecg_signal(edf_file):
    logger.debug('Loading file %s ...', edf_file.file_name)
    list_signals = edf_file.getSignalLabels()
    print list_signals
    # stop
    try:
        annotations = edf_file.readAnnotations()
    except Exception as e:
        print e

    # stop

    # crysis_time_seconds = find_crysis(annotations)
    ecg_indexes= find_ecg_signal(edf_file, list_signals)

    signals = [np.array([edf_file.readSignal(ecg_index[0])]).T
               for ecg_index in ecg_indexes]

    date_time = get_date_time(edf_file)

    names = [ecg_index[1] for ecg_index in ecg_indexes]

    return signals, names, date_time

def return_index(name, list_signals):
    try:
            print name
            index = list_signals.index(name)
            return (index, name)
    except Exception as e:
            print e
            return None

def find_ecg_signal(edf_file, list_signals):
    names = ['Ecg', 'ECG-', 'ECG+']

    list_indexes = [return_index(name, list_signals) for name in names]
    list_indexes = [index for index in list_indexes if index is not None]
    return list_indexes


def get_date_time(edf_file):
    d = edf_file.getHeader()['startdate']
    return d.strftime('%Y-%m-%d %H:%M:%S.%f')


def find_crysis(annotations):
    name_annotations = annotations[2]
    print annotations
    crysis_index =  [i for i,item in enumerate(name_annotations) if ('CRISE' or 'Remote' or 'MARK ON') in item]
    return annotations[0][crysis_index]



def find_baseline(annotations):
    name_annotations = annotations[2]
    print annotations
    crysis_index =  [i for i,item in enumerate(name_annotations) if ('CRISE' or 'Remote' or 'MARK ON') in item]
    return annotations[0][crysis_index]


def get_annotations(edf_file):
    annotations = edf_file.readAnnotations()
    return annotations


def find_markers(edf_file, list_signals):
    names = ['Ref']
    for name in names:
        try:
            print name
            index = list_signals.index(name)
            return index
        except Exception as e:
            print e


def get_list_signals(edf_file):
    signal_labels = edf_file.getSignalLabels() 
    return signal_labels


def read_edf_file(file_name):
    edf_file = pyedflib.EdfReader(file_name)
    signal_labels = edf_file.getSignalLabels() 
    return edf_file, signal_labels


def get_file_list(patient_list, disk):
    path_list = fetch_path_list(patient_list, disk)
    files_list = fetch_files_list(path_list)
    print files_list
    # print files_list
    return files_list


def fetch_files_list(path_list):
    print path_list
    print 
    return [path + i for path in path_list for i in os.listdir(path) if os.path.isfile(os.path.join(path,i)) and \
         '.edf' in i]


def fetch_path_list(patient_list, disk):
    return [disk + 'PATIENT' + str(arg) + '/' for arg in patient_list]


