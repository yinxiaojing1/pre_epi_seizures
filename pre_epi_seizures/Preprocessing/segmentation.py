from pre_epi_seizures.logging_utils.formatter_logging\
import logger as _logger

from pre_epi_seizures.storage_utils.storage_utils_hdf5\
import save_signal, load_signal, delete_signal


from biosppy.signals import ecg

import numpy as np

def shape_array(array):
    return np.array([array]).T


def create_rpeak_dataset(path, name, group):
    _logger.info('Detecting the R-peaks')
    X = load_signal(path=path, name_list=[name],
                     group_list=[group]) # 1 record per row
    signal_to_filter = X[0]['signal'][:,0]
    delete_signal(path,[name],['rpeaks'])
    # _logger.debug(signal_to_filter)
    rpeaks = ecg.hamilton_segmenter(signal=signal_to_filter.T,
                                    sampling_rate=1000)
    # _logger.info(rpeaks['rpeaks'])
    save_signal(path=path, signal_list=[shape_array(rpeaks['rpeaks'])],
                mdata_list=[''], name_list=['rpeaks_'+name], group_list=[group])


def create_heart_beat_dataset(path, name, group, save_dfile=None):
    X = load_signal(path=path, name=name, group=group) # 1 record per row
    rpeaks = load_signal(path=path, name='rpeaks', group=group)
    signal_to_segment = X['signal'][0,:].T
    heart_beat, rpeaks = ecg.extract_heartbeats(
                                 signal=signal_to_segment,
                                 rpeaks=rpeaks['signal'], sampling_rate=1000.0,
                                 before=0.2, after=0.4
                                 )
    return heart_beat, rpeaks


