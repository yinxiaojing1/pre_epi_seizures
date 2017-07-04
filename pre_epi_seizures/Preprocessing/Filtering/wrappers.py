"""
Wrappers: filters / R peak detectors
@author: Afonso Eduardo
"""
from __future__ import division

import os
import sys

import numpy as np
import scipy
import scipy.signal

# import matlab
# import matlab.engine

# Local imports
from filter_signal import filter_signal as _filter_signal
# from eksmoothing import EKSmoothing
# from eksmoothing17 import EKSmoothing17
###############################################################################################
# Filtering Functions
###############################################################################################
def filter_signal(X, **kwargs):
    """
    Wrapper of filter_signal.filter_signal: input can be a matrix containing multiple signals
    (1 per row).
    """
    return np.apply_along_axis(_filter_signal, -1, X, **kwargs)

def filterIR5to20(X, fs=500., order=None, **kwargs):
    """
    Wrapper of filter_signal: Apply bandpass filter 5 to 20 Hz.
    """
    order = int(0.3*fs) if order is None else order
    return filter_signal(X, order=order, sampling_rate=fs, ftype='FIR', band='bandpass', frequency=[5., 20.], **kwargs)


def _medianFIR(signal, fs=500, lpfilter=True, **kwargs):
    """
    Filter a signal with two median filters and apply a lowpass FIR, flattop window,
    cutoff 40Hz. This procedure follows the same median filter bank structure as [1].
    Parameters:
    -----------
    signal: array 1D
        Signal to be filtered.
    fs: float (default: 500)
        Sampling frequency [Hz].
    lpfilter: bool (default: True)
        Whether to apply the lowpass filter.
    kwargs: dict
        Not used, but necessary for flexibility in function call.
    Returns:
    --------
    array 1D
        Filtered signal.
    References:
    -----------
    [1] P. de Chazal, Automatic classification of heartbeats using ECG morphology and heartbeat interval features, 2004.
    """
    # filter parameters
    order = int(0.3 * fs)
    a1 = int(0.2 * fs)
    if a1 % 2 == 0:
        a1 += 1
    a2 = int(0.6 * fs)
    if a2 % 2 == 0:
        a2 += 1

    # baseline wander
    med1 = scipy.signal.medfilt(signal, a1)
    med2 = scipy.signal.medfilt(med1, a2)
    inter = signal - med2
    # low-pass
    if lpfilter:
        filtered = filter_signal(inter, ftype='FIR', band='lowpass', order=order,
                           window='flattop', frequency=40., sampling_rate=fs)
    else:
        filtered = inter
    return filtered

def medianFIR(X, **kwargs):
    """
    Wrapper of _medianFIR: input can be a matrix containing multiple signals (1 per row).
    """
    return np.apply_along_axis(_medianFIR, -1, X, **kwargs)

###############################################################################################
# R Peak Detectors: These should take a matrix (1 signal per row) as input and return a list of
#                   arrays with the R peak locs.
###############################################################################################
def UNSW_RPeakDetector(X, fs=500., artifact_masking=False, **kwargs):
    """
    Wrapper of UNSW R peak detector (matlab function) with additional modifications
    (see kwargs).
    Parameters:
    -----------
    X: array (1D or 2D)
        Signals (1 per row if 2D).
    fs: float (default: 500)
        Sampling frequency [Hz].
    artifact_masking: bool (default: False)
        If True, runs UNSW_ArtifactMask before peak detection.
    kwargs: dict
        Additional arguments if artifact_masking is True:
        - 'railV': list of 2 floats (default: [-300, 500])
            Bottom and top rail voltages. Be sure to check if the
            default range is appropriate!
        - 'fm': float (default: 50)
            Mains frequency [Hz].
        - 'return_mask': bool (default: False)
            If True, returns the list of masks.
        Other arguments:
        - 'check_peak': bool (default: False)
            If True, modifies the Rpeak detection algorithm
            to check whether the peak is a maximum/minimum
            around a fixed neighborhood and adjust accordingly.
            This procedure ignores artifact masking!
        - 'tpoint': str (default: 'max')
            Determines which function to apply to detect the turning
            point ('max', 'min' or 'auto'). If 'auto', the function
            is determined based on the sign of the median of the detected
            R peaks on a signal-by-signal basis. Use 'auto' at your own risk!
        - 'neighbors': int (default: 30)
            Neighborhood size (1-sided) [# samples].
        - 'padding': bool (default: False)
            If True, pad the signal (on both edges) with a linear ramp of 30
            samples to 0 and then n_pad-30 0s. This procedure is applied
            after removing the baseline wander with median filters.
        - 'lp_b4_padding': bool (default: False)
            Applies a low pass filter (see medianFIR) of 40Hz before padding.
        - 'n_pad': int (default: 500)
            Number of samples when padding (>30).
        - 'check_amplitude': bool (default: False)
            If True, compute the median amplitude of R peaks (per signal) and
            discard peaks whose amplitudes are below a certain threshold.
            This is computed after removing the baseline wander with median
            filters.
        - 'amplitude_thre': float (default: -0.3)
            Amplitude threshold.
        - 'amplitude_thre_edges': float (default: -0.6)
            Amplitude threshold at the edges (within 20 samples).
    Returns:
    --------
    list of arrays
        R peak locs list.
    list of arrays (If artifact_masking==True and kwargs['return_mask']==True)
        Artifact masking list.
    """
    def check_params(**d):
        [setattr(f, key, (kwargs[key] if key in kwargs else d[key])) for key in d]
    f = getattr(sys.modules[__name__], sys._getframe().f_code.co_name)
    X = np.atleast_2d(X)
    check_params(railV=[-300, 500], fm=50, return_mask=False, check_peak=False, neighbors=30, tpoint='max',
                 padding=False, n_pad=500, lp_b4_padding=False, 
                 check_amplitude=False, amplitude_thre=-.3, amplitude_thre_edges=-.6)
    f.railV, f.fm, fs = matlab.double(f.railV), float(f.fm), float(fs)
    meng = matlab.engine.start_matlab("-nojvm")
    meng.addpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"UNSW_QRSD"), nargout=0)
    mask_list = [meng.UNSW_ArtifactMask(matlab.double(x.tolist()), f.railV, f.fm, fs,
                nargout=5)[-1] for x in X] if artifact_masking else [matlab.double([]),]*len(X)
    if f.padding:
        X = medianFIR(X, fs=fs, lpfilter=f.lp_b4_padding)
        X = np.pad(X, ((0,0),(30, 30)), 'linear_ramp')
        X = np.pad(X, ((0,0),(f.n_pad-30, f.n_pad-30)), 'constant')
        mask_list = [matlab.double((np.array(mask).ravel()+f.n_pad).tolist()) for mask in mask_list]
    R_list = [np.array(meng.UNSW_QRSDetector(matlab.double(x.tolist()), fs,
                       mask, False, nargout=5)[0], dtype=int).ravel()-1 for x, mask in zip(X, mask_list)]
    meng.quit()
    if f.check_peak:
        if f.tpoint in ['max', 'min']:
            tpoint_f_list = [np.argmax,]*len(X) if f.tpoint=='max' else [np.argmin,]*len(X)
        elif f.tpoint == 'auto':
            tpoint_f_list = [(np.argmin if np.median(x[rpeaks])<0 else np.argmax) for x, rpeaks in zip(X, R_list)]
        R_list = [np.array([rpeak+tpoint_f(x[rpeak-f.neighbors:rpeak+f.neighbors+1])-f.neighbors
                  for rpeak in rpeaks]) for x, rpeaks, tpoint_f in zip(X, R_list, tpoint_f_list)]
    mask_list = [np.array(mask, dtype=int).ravel()-1 for mask in mask_list]
    if f.padding:
        X = X[:, f.n_pad:-f.n_pad]
        R_list, mask_list = zip(*[((rpeaks-f.n_pad)[np.logical_and(rpeaks>=f.n_pad, rpeaks<X.shape[1]+f.n_pad)], mask-f.n_pad)
                                for rpeaks, mask in zip(R_list, mask_list)])
    if f.check_amplitude:
        X_ = X if f.padding else medianFIR(X, fs=fs, lpfilter=False)
        new_R_list = []
        for x, rpeaks in zip(X_, R_list):
            if len(rpeaks)==0:
                new_R_list.append(np.array([]))
                continue
            medianR = np.median(x[rpeaks])
            keep = (x[rpeaks]/medianR-1>=f.amplitude_thre)
            if rpeaks[0]<20 and (x[rpeaks[0]]/medianR-1>=f.amplitude_thre_edges):
                keep[0] = True
            if rpeaks[-1]>X.shape[1]-20 and (x[rpeaks[-1]]/medianR-1>=f.amplitude_thre_edges):
                keep[-1] = True
            new_R_list.append(rpeaks[keep])
        R_list = new_R_list
    if artifact_masking and f.return_mask:
        return R_list, mask_list
    return R_list


def PT_GIBBS_QRSDetector(X, fs=500., **kwargs):
    """
    Wrapper of QRSdetection (Pan-TompKins Algorithm) from GIBBS_PT_DEL (matlab)
    Parameters:
    -----------
    X: array (1D or 2D)
        Signals (1 per row if 2D).
    fs: float (default: 500)
        Sampling frequency [Hz].
    kwargs: dict
        Not used, but necessary for flexibility in function call.
    Returns:
    --------
    4 list of arrays
        R peak locs, QRS width, Q locs, S locs.
    array 2D
        ECG signal after low pass filter.
    """
    X = np.atleast_2d(X)
    meng = matlab.engine.start_matlab("-nojvm")
    meng.addpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"GIBBS_PT_DEL"), nargout=0)
    R_loc, QRS_width, Q_loc, S_loc, lp_ecg = zip(*[map(lambda a: np.array(a[0] if len(a) else []), 
                                             meng.QRSdetection(matlab.double(x.tolist()), 
                                             float(fs), nargout=5)) for x in X])
    meng.quit()
    R_loc, QRS_width, Q_loc, S_loc = map(lambda a: map(lambda b: b.astype(int), a), 
                                         [R_loc, QRS_width, Q_loc, S_loc])
    return R_loc, QRS_width, Q_loc, S_loc, lp_ecg

def PT_GIBBS_RPeakDetector(X, **kwargs):
    """
    Wrapper of PT_GIBBS_QRSDetector: only extracts R peak locs.
    """
    return PT_GIBBS_QRSDetector(X, **kwargs)[0]

###############################################################################################

def QSDetector(X, R_list, fs=500., wnd=0.08, QMAX=0.04, SMAX=0.04, show_plot=False):
    """
    Given the R locations, estimate Q onset and S offset locations by computing the 
    first derivative and take the 2nd zero crossing on both sides per R peak.
    Parameters:
    -----------
    X: array (1D or 2D)
        Signals (1 per row if 2D).
    R_list: list of arrays
        Each array contains the position of R peaks.R
    fs: float (default: 500)
        Sampling frequency [Hz].
    wnd: float (default: 0.08s -> 40 samples @ 500Hz)
        Window of zero crossing analysis centered on R (length of one side) [s].
    QMAX: float (default: 0.04 -> 20 samples @ 500Hz)
        Maximum allowed time frame from each Q to R peak [s].
    SMAX: float (default: 0.04 -> 20 samples @ 500Hz)
        Maximum allowed time frame from each R peak to S [s].
    show_plot: bool (default: False)
    	If True, plot the signals and the corresponding Q, R and S locations.
    Returns:
    --------
    2 list of arrays
        Q locs, S locs.
    Notes:
    ------
    This is a very crude method of detection. There might be better alternatives.
    """
    wnd = int(fs*wnd) # time to samples
    QMAX, SMAX = int(fs*QMAX), int(fs*SMAX)
    X = np.atleast_2d(X)
    Xdiff1 = np.diff(X)
    Xdiff1 = np.hstack([np.zeros((len(Xdiff1), 1)), Xdiff1])  
    #Xdiff2 = np.diff(Xdiff1)
    #Xdiff3 = np.diff(Xdiff2)
    #Xdiff1, Xdiff2, Xdiff3 = [np.hstack([np.zeros((len(s), i+1)), s]) 
    #                        for i, s in enumerate([Xdiff1, Xdiff2, Xdiff3])]
    Xdiff1sign = np.sign(Xdiff1)
    Xdiff1signchange = ((np.roll(Xdiff1sign, 1) - Xdiff1sign) != 0).astype(int)
    Q_list, S_list = [], []
    for x, xdiff1sc, rpeaks in zip(X, Xdiff1signchange, R_list):
        qloc = np.zeros_like(rpeaks)
        sloc = np.zeros_like(rpeaks)
        for i, rpeak in enumerate(rpeaks):
            a = np.array(xdiff1sc[rpeak-wnd:rpeak+wnd+1])
            a[wnd-3:wnd+3+1] = 0 # ignore the neighborhood of rpeak
            q = np.where(a[:wnd] == 1)[0]
            q = rpeak-(wnd-q[-2]) if q.size>=2 else np.maximum(rpeak-QMAX, 0)
            s = np.where(a[wnd:] == 1)[0]
            s = rpeak+s[1] if s.size>=2 else np.minimum(rpeak+SMAX, len(x)-1)
            qloc[i], sloc[i] = q, s
            #plt.plot(x[rpeak-40:rpeak+41])
            #plt.plot(xdiff1sc[rpeak-wnd:rpeak+wnd+1])
            #plt.plot(wnd-(rpeak-q), x[rpeak-40:rpeak+41][wnd-(rpeak-q)], 'ro')
            #plt.plot(wnd+s-rpeak, x[rpeak-40:rpeak+41][wnd+s-rpeak], 'go')
            #plt.show()
        Q_list.append(qloc)
        S_list.append(sloc)
    if show_plot:
	    for x, rlocs, qlocs, slocs in zip(X, R_list, Q_list, S_list):
	    	plt.figure(figsize=(19,8))
	    	plt.plot(x, 'k')
	    	plt.plot(rlocs, np.zeros(len(x))[rlocs], 'ro')
	    	plt.plot(qlocs, np.zeros(len(x))[qlocs], 'go')
	    	plt.plot(slocs, np.zeros(len(x))[slocs], 'co')
	    	ax = plt.gca()
	    	ax.axhline(0, color='k', ls='--')
	    	plt.show()
    return Q_list, S_list






if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt
    sys.path.append(os.path.abspath("../"))
    from preprocessing import read_R_csv, read_dataset_csv

    f_path = lambda datafile: os.path.abspath("..\..\{}\{}".format(datafolder, datafile))
    datafolder = 'Data'
    fs = 500. # sampling frequency [Hz]

    R_list = read_R_csv(f_path("ECGIDDB_raw_UNSW_Rpeaks.csv"))
    X, y = read_dataset_csv(f_path("ECGIDDB_medianFIR_UNSW.csv"), multicolumn=True)

    QSDetector(X, R_list, fs=fs, show_plot=True)

