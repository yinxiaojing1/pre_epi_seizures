
def create_seizure_dataset(path_to_save, time_before_seizure, time_after_seizure, *args):
    path = '~/Desktop/HSM_data.h5'
    name = ['FA77748S']
    group = ['/PATIENT1/crysis']

    raw = load_signal(path, name, group)

    begin_seizure_seconds = raw[0]['mdata']['crysis_time_seconds']
    begin_seizure_sample = int(1000*begin_seizure_seconds[0])

    sampling_rate_hertz = 1000

    signal = raw[0]['signal']
    no_seizure_ecg_raw = signal[0:begin_seizure_sample,0]

    ecg_10min_5min_raw = signal[begin_seizure_sample
                                 -sampling_rate_hertz*10*60:
                                 begin_seizure_sample
                                 + sampling_rate_hertz*5*60]