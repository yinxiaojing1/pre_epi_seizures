# 3rd party
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn


class AdaContextHRAmpWaveOutlier(object):

    def __init__(self, bpm_range, sampling_rate, weights_t, max_diff, memory_size, weights_a=[]):
        """ Class created to implement a simple decision system to decide which peaks should be taken into account when
        segmenting the signal in waves. This decision system makes use of context information in terms of
        heart rate and fiducial point amplitude, as well as physiological information. The instantaneous heart rate is
        estimated using the interbeat interval, which is defined as the interval between two consecutive fiducial
        points. This version does not perform backtracking (i.e, consider non-consecutive fiducial points).

        Parameters:
        -----------
        bpm_range: array int
            range of acceptable bpm values.
        sampling_rate: float
            sampling rate of the signal in Hz
        weights_t: array float
            coefficients for the heart rate context range
        weights_a: array float
            coefficients for the fiducial amplitude context range
        max_diff: int
            when the time interval since the last accepted peak is larger than max_diff, in seconds, the memory is
            reset
        memory_size: int
            maximum number of past peaks considered, size of the memory

        Heuristic, empirical values:
        ----------------------------
        weights_t = [0.6, 1.4]
        weights_a = [0.8, 1.2]

        !!!check histograms of high quality signal recordings to estimate the weights!!!
        """

        self.bpm_range = bpm_range
        self.sampling_rate = sampling_rate
        self.weights_t = weights_t
        self.weights_a = weights_a
        self.peaks_diff_past = []  # sample differences for the previous memory_size fiducial points
        self.peaks_amp_past = []  # amplitudes of the previous memory_size fiducial points
        self.memory_filled = False  # memory status
        self.memory_size = memory_size
        self.max_diff = max_diff
        self.peaks_diff_cum = 0  # number of samples between previous accepted fiducial point and the current discarded

    def validation(self, peaks_diff_curr, peak_amp_curr=None):
        """ Method that validates the current peak based on temporal and amplitude information of itself and last
         peaks.

            Parameters:
            -----------
            peaks_diff_curr: int
                sample difference between the next peak and the current one
            peak_amp_curr: float
                amplitude of the current peak

            Returns:
            --------
            bool
                True if the peak is considered valid, else False
        """

        past_mean_peaks_diff = np.mean(self.peaks_diff_past)
        past_mean_amp = np.mean(self.peaks_amp_past)

        # physiological acceptable heart rate condition check
        check_bpm = self.bpm_range[0] <= self.sampling_rate * 60. / peaks_diff_curr <= self.bpm_range[1]
        # print '-' * 100
        # print 'curr', peaks_diff_curr, peak_amp_curr
        # print 'cond fix', self.bpm_range[0], self.sampling_rate * 60. / peaks_diff_curr, self.bpm_range[1]

        # when the memory is not empty, use contextual information
        if len(self.peaks_diff_past) != 0:

            # print 'past', self.peaks_diff_past
            # print 'mean', past_mean_peaks_diff
            # print 'cond', self.weights_t[0] * past_mean_peaks_diff, peaks_diff_curr, \
            #     self.weights_t[1] * past_mean_peaks_diff

            # heart rate contextual interval check
            check_t = self.weights_t[0] * past_mean_peaks_diff <= peaks_diff_curr <= \
                      self.weights_t[1] * past_mean_peaks_diff

            # if the fiducial point amplitude context is used
            if len(self.weights_a) != 0:
                # fiducial point amplitude contextual interval check
                check_a = np.abs(peak_amp_curr - past_mean_amp) <= np.abs(self.weights_a[0] * past_mean_amp -
                                                                          self.weights_a[1] * past_mean_amp)

                # check the 3 conditions
                check_f = np.all([check_bpm, check_t, check_a])

            else:  # use only temporal information
                check_f = np.all([check_bpm, check_t])

        else:  # if the memory is empty, use only physiological heart rate range

            check_f = check_bpm

        if not check_f:  # reset memory if
            #print 'memory reset'
            self.memory_reset(peaks_diff_curr)

        #print check_f
        #print '-' * 100
        return check_f

    def addtomemory(self, peaks_diff_curr, amp=None):
        """ Method that adds to the memory the current validated peak and discards the oldest peak if the memory is full
        (FIFO).

            Parameters:
            ----------
            peaks_diff_curr: int
                sample difference between the current fiducial point and the previous one
            amp: float
                amplitude of the fiducial point
        """

        # reset variable that counts number of samples discarded since the last fiducial point
        # accepted
        self.peaks_diff_cum = 0

        if len(self.peaks_diff_past) == self.memory_size:
            self.peaks_diff_past = self.peaks_diff_past[1:]
            if amp is not None:
                self.peaks_amp_past = self.peaks_amp_past[1:]

        self.peaks_diff_past.append(peaks_diff_curr)
        if amp is not None:
            self.peaks_amp_past.append(amp)

    def memory_reset(self, peaks_diff_curr):
        """ Method that resets the memory.

            Parameters:
            -----------
            peaks_diff_curr: int
                sample difference between the current fiducial point not accepted and the previous one
        """

        # add number of sample points between the current fiducial point that was not accepted and the previous one
        self.peaks_diff_cum += peaks_diff_curr

        # reset memory if the amount of sample points between the current fiducial point not accepted and the last one
        # accepted is greater than max_diff
        if self.peaks_diff_cum > self.max_diff * self.sampling_rate:
            self.peaks_diff_past = []
            self.peaks_amp_past = []
            self.peaks_diff_cum = 0

    def run(self, peaks_diff_arr, peaks_amp_arr=None):

        if peaks_amp_arr is None:
            peaks_amp_arr = [None] * len(peaks_diff_arr)

        valid_peaks = []
        for i in range(len(peaks_diff_arr)):
            dec = self.validation(peaks_diff_arr[i], peaks_amp_arr[i])
            valid_peaks.append(dec)
            if dec:
                self.addtomemory(peaks_diff_arr[i], peaks_amp_arr[i])
            else:
                self.memory_reset(peaks_diff_arr[i])

        return valid_peaks


# Example -----------------

bpm_range = [40, 170]  # physiological heart range, in bpm
sampling_rate = 1e3  # in Hz
weights_t = [0.8, 1.3]  # heart rate context weights
max_diff = 3  # forgetting time interval
memory_size = 5  # number of last accepted peaks considered
weights_a = []  # fiducial point amplitude context weights

outlier_dec = AdaContextHRAmpWaveOutlier(bpm_range, sampling_rate, weights_t, max_diff, memory_size, weights_a)

rr_int = np.random.normal(700, 100, 60).astype('int')
# rr_int = np.diff(rpeaks)

valid_peaks = outlier_dec.run(rr_int)

x = np.arange(len(rr_int))
valid_idx = np.where(valid_peaks)[0]
plt.figure()
plt.plot(x, 60*sampling_rate/rr_int, label='Raw')
plt.plot(x[valid_idx], 60*sampling_rate/rr_int[valid_idx], label='Valid')
plt.legend()
plt.ylabel('Instantaneous heart rate (bpm)')

#########################
# check ratio current peak hr with average hr last n peaks

# rr_int = np.diff(rpeaks)
# num_rrs = 5
# moving_avg = np.convolve(rr_int, np.ones((num_rrs,))/num_rrs, mode='valid')
# plt.figure()
# plt.scatter(np.arange(rr_int[4:]), rr_int[4:]/moving_avg)
# plt.ylabel('Ratio last %i peaks' % num_rrs)
# plt.xlabel('Peak number')
