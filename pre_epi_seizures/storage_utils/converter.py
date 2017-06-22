import storage_utils_phisionet as st_ph
from biosppy import storage as st_hdf5


def converter_phisionet_hdf5(*args):
    file_path = '~/Desktop/phisionet_dataset.h5'
    opened_file = open_file(file_path)

    for arg in args:
        print arg
        _converter_phisionet_hdf5(opened_file, arg)

    opened_file.close()

def _converter_phisionet_hdf5(file, nb_sz):
    header, signals = fetch_header_signals_phisionet(nb_sz)
    print header, signals
    file.add_signal(signal=signals, mdata=header, group='raw', name='sz_' + str(nb_sz), compress=False)


def open_file(path): 
    return st_hdf5.HDF(path, 'a')


def fetch_header_signals_phisionet(nb_sz):
    return st_ph.load_header_signals_phisionet(nb_sz)


converter_phisionet_hdf5(1, 2, 3, 4, 5, 6, 7)