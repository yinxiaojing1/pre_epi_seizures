from biosppy import storage as st_hdf5
import sys




def load_signal(path, name, group):
    opened_file = st_hdf5.HDF(path, 'r+')

    try:
        signal = opened_file.get_signal(name=name, group=group)
    except Exception as e:
        _logger.debug(e)
        signal = None

    opened_file.close()
    return signal


def save_signal(signal, mdata, path, name, group):
    opened_file = st_hdf5.HDF(path, 'r+')

    try:
        signal = opened_file.add_signal(signal=signal, mdata=mdata, name=name, group=group)
    except Exception as e:
        _logger.debug(e)

    opened_file.close()