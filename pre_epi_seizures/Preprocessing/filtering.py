from pre_epi_seizures.logging_utils.formatter_logging\
    import logger as _logger

from pre_epi_seizures.storage_utils.storage_utils_hdf5\
    import save_signal, load_signal, delete_signal

from pre_epi_seizures.storage_utils.data_handlers import *

from Filtering import medianFIR, filterIR5to20,\
    filter_signal, gaussian_fit

from Filtering.eksmoothing import EKSmoothing

def baseline_removal(path, name, group, sampling_rate):
    _logger.info('Removing the baseline ...')
    create_filtered_dataset(path=path, name=name,
            group=group, filtmethod='medianFIR',
            sampling_rate=sampling_rate)


def noise_removal(path, name, group, sampling_rate):
    _logger.info('Removing noise ...')
    create_filtered_dataset(path=path, name=name, group=group, filtmethod='FIR_lowpass_40hz')

def eks_smoothing(path, name, group, sampling_rate):
    try:
        # make_new_rpeaks
        rpeaks_signal_structure = load_signal(path_to_load ,zip(group_list, rpeaks_names))
    except Exception as e:
        print e
        _logger.debug(e)
        create_rpeak_dataset(path_to_load, zip(group_list_baseline_removal, name_list), sampling_rate)
        rpeaks_signal_structure = load_signal(path_to_load ,zip(group_list, rpeaks_names))

def create_filtered_dataset(path, name, group, filtmethod,
                            save_dfile=None, multicolumn=False, sampling_rate=1000,
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
    mdata = get_mdata_dict(one_signal_structure)

    if filtmethod == 'medianFIR':
        X_filt = globals()[filtmethod](dataset, **kwargs)
        _logger.debug(X_filt)

    elif filtmethod == 'FIR_lowpass_40hz':
        X_filt = globals()['filter_signal'](dataset, ftype='FIR', band='lowpass',
                  order=10, frequency=40,
                  sampling_rate=sampling_rate)
        _logger.debug(X_filt)



    else:
        stop

    save_signal(path=path, signal_list=[X_filt],
                mdata_list=[mdata], name_list=[name], group_list=[filtmethod])
