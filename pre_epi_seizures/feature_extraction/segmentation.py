from pre_epi_seizures.storage_utils.storage_utils_hdf5 \
    import load_signal, save_signal

def compute_fixed_beats(signal, rpeaks):
    return [signal[rpeak - 200:rpeak + 400] for rpeak in rpeaks[1:-2]]

