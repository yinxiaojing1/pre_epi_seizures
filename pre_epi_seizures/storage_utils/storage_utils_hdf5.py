
from pre_epi_seizures.logging_utils.formatter_logging\
    import logger as _logger

from biosppy import storage as st_hdf5

import sys


def load_signal(path, group_name_list):
    opened_file = st_hdf5.HDF(path, 'a')
    return_dict = {group_name: get_signal(opened_file=opened_file,
                                group_name=group_name)
         for group_name in group_name_list}
    opened_file.close()
    return return_dict


def save_signal(path, signal_list, mdata_list, name_list, group_list):
    opened_file = st_hdf5.HDF(path, 'a')

    for group in group_list:
        for signal, mdata, name in zip(signal_list,
                                       mdata_list, name_list):
            add_signal(opened_file=opened_file, signal=signal,
               mdata=mdata, name=name, group=group)

    opened_file.close()


def get_signal(opened_file, group_name):
    group = group_name[0]
    name = group_name[1]
    _logger.info('Loading [signal: %s][group: %s]', name, group)
    try:
        signal = opened_file.get_signal(name=name, group=group)
    except Exception as e:
        _logger.debug(e)

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


def list_group_signals(path, group):
    opened_file = st_hdf5.HDF(path, 'r')
    list_signals = opened_file.list_signals(group=group, recursive=False)
    opened_file.close()
    return list_signals

def _add_event(opened_file, ts, values, mdata, group, name):
    _logger.info('adding [event: %s][group: %s]', name, group)
    try:
        opened_file.add_event(ts, values, mdata, group, name)
    except Exception as e:
        _logger.debug(e)


def add_event(path, ts, values, mdata, group, name):
    opened_file = st_hdf5.HDF(path, 'a')
    _add_event(opened_file, ts, values, mdata, group, name)
    opened_file.close()


