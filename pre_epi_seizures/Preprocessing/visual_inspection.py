import numpy as np
import matplotlib.pyplot as plt


def visual_inspection(signal, rpeaks, heart_rate, time_before_seizure, start, end, sampling_rate):
    Fs = sampling_rate
    N = len(signal)
    T = (N - 1) / Fs
    time = np.linspace(0, T, N, endpoint=False)
    print time

    plt.subplot(2,1,1)
    plt.plot(time, signal)
    plt.plot(time[rpeaks], signal[rpeaks], 'o')
    plt.axvline(x=time_before_seizure*60, color = 'g')
    plt.xlim([start, end])
    plt.subplot(2, 1, 2)
    plt.plot(time[rpeaks[1]:rpeaks[-1]], heart_rate)
    plt.axvline(x=time_before_seizure*60, color = 'g')
    plt.show()