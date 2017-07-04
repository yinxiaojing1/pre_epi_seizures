from pre_epi_seizures.logging_utils.formatter_logging\
import logger as _logger

from pre_epi_seizures.storage_utils.storage_utils_hdf5\
import save_signal, load_signal

from biosppy.signals import ecg

def create_rpeak_dataset(path, name_list, group_list, save_dfile=None):
    _logger.info('Detecting the R-peaks')

    X = load_signal(path=path, name=name, group=group) # 1 record per row
    signal_to_filter = X['signal']
    rpeaks = ecg.hamilton_segmenter(signal=signal_to_filter.T, sampling_rate=X['mdata']['fs'])
    save_signal(signal=rpeaks['rpeaks'], mdata='', path=path, name='rpeaks_'+name, group=group)


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


