from biosppy import storage as st_hdf5
import sys
import logging


_logger = logging.getLogger(__name__)


def setup_logging(loglevel = 'INFO'):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def load_from_file_hdf5(path):
    opened_file = open_file(path)
    list_signals_names = list_signals_hdf5(opened_file)
    _logger.debug(list_signals_names)

    for signal_name in list_signals_names:
        try: 
            header, signal = load_header_signals_hdf5(opened_file, signal_name)

        except Exception as e: 
            _logger.debug(e)

    close_file(opened_file)
    return header, signal


def save_to_file_hdf5(path):
    opened_file = open_file(path)
    save_header_signals_hdf5(opened_file)
    close_file(opened_file)


def list_signals_hdf5(file):
    list_signals = file.list_signals()
    return list_signals['signals'][0]


def save_signal_hdf5(file, signal, group, mdata, name, compress=False):
    file.add_signal(signal=signal, mdata=mdata, name='signals')


def load_header_signals_hdf5(file, name):
    header = file.get_header()
    signals = file.get_signal(name=name)
    return header, signals


def open_file(path):
    return st_hdf5.HDF(path, 'r+')


def close_file(file):
    file.close()


def main(arg):
    setup_logging('DEBUG')
    _logger.debug("Starting crazy calculations...")
    header, signals = load_from_file_hdf5(arg)
    _logger.debug(header)
    _logger.debug(signals['signal'])


def run():
    main('~/Desktop/phisionet_data/phisionet_2.h5')

run()