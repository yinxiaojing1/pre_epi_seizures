from biosppy.signals import ecg

import numpy as np

def inst_heart_rate(rpeaks): 
    return 60.0 * 1000 / np.diff(rpeaks)




