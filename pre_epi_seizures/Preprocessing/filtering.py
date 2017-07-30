from pre_epi_seizures.logging_utils.formatter_logging\
    import logger as _logger

from pre_epi_seizures.storage_utils.storage_utils_hdf5\
    import save_signal, load_signal, delete_signal

from pre_epi_seizures.storage_utils.data_handlers import *

from Filtering import medianFIR, filterIR5to20,\
    filter_signal, gaussian_fit

def baseline_removal(path, name, group):
    _logger.info('Removing the baseline ...')
    create_filtered_dataset(path=path, name=name, group=group, filtmethod='medianFIR', save_dfile=path)


def create_filtered_dataset(path, name, group, filtmethod,
                            save_dfile=None, multicolumn=False,
                            **kwargs):
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
        Path pointing to new csv file. 
        If None, replace 'raw' with 'filtmethod'.
        If 'raw' does not exist in dfile name, append '_filtmethod'.
    multicolumn: bool (default: False)
        If the csv file to be loaded has multi-index columns. See
        read_dataset_csv and save_dataset_csv functions.
    kwargs: dict
        Additional arguments to filtmethod function (e.g. fs).
    """
    delete_signal(path,[name],[filtmethod])
    X = load_signal(path=path, group_name_list = zip([group], [name])) # 1 record per row
    one_signal_structure = get_one_signal_structure(X, zip([group], [name])[0])
    dataset = get_multiple_records(one_signal_structure)
    X_filt = globals()[filtmethod](dataset, **kwargs)
    _logger.debug(X_filt)
    save_signal(path=path, signal_list=[X_filt],
                mdata_list=[''], name_list=[name], group_list=[filtmethod])
