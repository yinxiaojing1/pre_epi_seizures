import numpy as np
import logging
import sys
import wfdb

_logger = logging.getLogger(__name__)


def setup_logging(loglevel = 'INFO'):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def load_header_signals_phisionet (nb_sz):
    record = _load_record_phisionet(nb_sz)
    return _fetch_record_header(record), _fetch_record_signals(record)


def _fetch_record_header(record):
    d = dict((k, v) for k, v in record.__dict__.items() \
        if (k != 'p_signals' ))

    return d


def _fetch_record_signals(record):
    signals = record.__dict__['p_signals']

    return signals


def _load_record_phisionet(nb_sz):
    """ Load the ECG recording number nr_sz hosted on 
    https://www.physionet.org/physiobank/database/szdb/. An internet 
    connection is required! 

        Parameters:
        -----------
        nr_sz: int
            number of the seizure.

        Returns:
        --------
        array 1D
            Raw Ecg signal.
    """
    return _load_wfdb_phisionet(nb_sz)

def _load_wfdb_phisionet(nb_sz):
    """ Load the ECG recording number nr_sz hosted on 
    https://www.physionet.org/physiobank/database/szdb/. An internet 
    connection is required! 

        Parameters:
        -----------
        nr_sz: int
            number of the seizure.

        Returns:
        --------
        array 1D
            Raw Ecg signal.
    """
    record = wfdb.rdsamp('sz0'+ str(nb_sz), pbdir = 'szdb/')
    return record


def main(arg):
    setup_logging('DEBUG')
    _logger.debug("Starting crazy calculations...")
    header, signals = load_header_signals_phisionet(arg)
    _logger.debug(header)
    _logger.debug(signals)


def run():
    main(1, 2, 3, 4, 5, 6, 7)
