
from pre_epi_seizures.logging_utils.formatter_logging\
    import logger as _logger

from biosppy import storage as st_hdf5

import sys


def load_signal(path, name_list, group_list):
    opened_file = st_hdf5.HDF(path, 'a')
    return_list = [get_signal(opened_file=opened_file, name=name,
                              group=group)
                   for group in group_list
                   for name in name_list]
    opened_file.close()
    return return_list


def save_signal(path, signal, mdata, name, group):
    opened_file = st_hdf5.HDF(path, 'a')
    add_signal(opened_file=opened_file, signal=signal,
               mdata=mdata, name=name, group=group)
    opened_file.close()


def get_signal(opened_file, name, group):
    _logger.info('Loading [signal: %s][group: %s]', name, group)
    try:
        signal = opened_file.get_signal(name=name, group=group)
    except Exception as e:
        _logger.debug(e)
        signal = None
    return signal

def add_signal(opened_file, signal, mdata, name, group):
    _logger.info('Saving [signal: %s][group: %s]', name, group)
    try:
        signal = opened_file.add_signal(signal=signal, mdata=mdata,
                                        name=name, group=group)
    except Exception as e:
        _logger.debug(e)

def delete_signal(path, name_list, group_list):
    opened_file = st_hdf5.HDF(path, 'a')

    for group in group_list:
        for name in name_list:
            _delete_signal(opened_file, name, group)

    opened_file.close()

def _delete_signal(opened_file, name, group):
    _logger.info('deleting [signal: %s][group: %s]', name, group)
    try:
        opened_file.del_signal(group=group, name=name)
    except Exception as e:
        _logger.debug(e)