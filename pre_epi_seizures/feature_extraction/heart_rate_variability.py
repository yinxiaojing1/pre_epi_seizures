from biosppy.signals import ecg


def inst_heart_rate(rpeaks): 
    return 60 / np.diff(rpeaks)




