import sys 

import os

import numpy as np

import pyedflib

from pre_epi_seizures.logging_utils.formatter_logging import logger

from pre_epi_seizures.storage_utils.storage_utils_hdf5 import  load_signal, save_signal


# @profile
def converter(*args):
    path_list = get_file_list(*args)

    for path in path_list:
        edf_file = pyedflib.EdfReader(path)
        signal, mdata= load_ecg_signal(edf_file)
        print signal
        logger.debug("the mdata is %s", mdata)
        name = path[18:26]
        name_patient = path[5:13]

        if mdata['crysis_time_seconds']:
            group = '/'+name_patient +'/crysis'
        else: 
            group = '/'+name_patient +'/free'

        save_signal(path='/home/sargo/Desktop/HSM_data.h5', signal=signal, mdata=mdata, name=name, group=group)
        edf_file._close()

def create_mdata(header, crysis_time_seconds):
    header['crysis_time_seconds'] = list(crysis_time_seconds)
    return header


def load_ecg_signal(edf_file):
    logger.debug('Loading file %s ...', edf_file.file_name)
    list_signals = edf_file.getSignalLabels()
    annotations = edf_file.readAnnotations()
    crysis_time_seconds = find_crysis(annotations)
    ecg_index= find_ecg_signal(edf_file, list_signals)

    return np.array([edf_file.readSignal(ecg_index)]).T, \
    create_mdata(edf_file.getSignalHeader(ecg_index), 
                 crysis_time_seconds)



def find_ecg_signal(edf_file, list_signals):
    names = ['Ecg', 'ECG-', 'ECG+']

    for name in names:
        try:
            print name
            index = list_signals.index(name)
            return index
        except Exception as e:
            print e


def find_crysis(annotations):
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


def get_file_list(*args):
    path_list = fetch_path_list(*args)
    files_list = fetch_files_list(path_list)
    return files_list


def fetch_files_list(path_list):
    return [path + i for path in path_list for i in os.listdir(path) if os.path.isfile(os.path.join(path,i)) and \
         '.edf' in i]


def fetch_path_list(*args):
    return ['/mnt/' + 'PATIENT'+str(arg) + '/HSM/'for arg in args]


converter(1,2,4,5)