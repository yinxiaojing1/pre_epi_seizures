from pre_epi_seizures.storage_utils.storage_utils_hdf5 import save_signal, load_signal

from Filtering import medianFIR, filterIR5to20, filter_signal, gaussian_fit

from formatter_logging import logger

def baseline_removal(path, name, group):
    create_filtered_dataset(path=path, name=name, group=group, filtmethod='medianFIR', save_dfile=path)


def create_filtered_dataset(path, name, group, filtmethod, save_dfile=None, multicolumn=False, **kwargs):
    """
    Load dataset from a hdf5 file, filter with filtmethod save it.
    Parameters:
    -----------
    dfile: str
        Path pointing to csv file containing 1 record per column.
    filtmethod: str (default: 'medianFIR')
        Name of the filter function defined in the global scope. For instance,
        'medianFIR' or 'filterIR5to20'.
    save_dfile: str (default: None)
        Path pointing to new csv file. If None, replace 'raw' with 'filtmethod'.
        If 'raw' does not exist in dfile name, append '_filtmethod'.
    multicolumn: bool (default: False)
        If the csv file to be loaded has multi-index columns. See
        read_dataset_csv and save_dataset_csv functions.
    kwargs: dict
        Additional arguments to filtmethod function (e.g. fs).
    """
    if save_dfile is None:
        save_dfile = dfile.replace('raw', filtmethod) if 'raw' in dfile else dfile.split('.')[0] + '_{}.csv'.format(filtmethod)
    X = load_signal(path=path, name=name, group=group) # 1 record per row
    X_filt = globals()[filtmethod](X['signal'].T, **kwargs)
    logger.debug(X_filt)
    stop
    save_signal(signal=X_filt.T, mdata=X['mdata'], path=path, name=name, group=filtmethod)
