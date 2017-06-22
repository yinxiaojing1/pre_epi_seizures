from biosppy.signals import ecg as ecg

def rpeak_detector(signal, sampling_rate, method):
    if method == 'christov':    
        ecg.christov_segmenter(signal=signal, 
                               sampling_rate=sampling_rate)