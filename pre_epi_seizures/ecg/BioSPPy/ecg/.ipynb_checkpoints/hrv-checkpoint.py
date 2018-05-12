"""
.. module:: hrv
   :platform: Unix, Windows
   :synopsis: This module provides various methods to perform Heart Rate Variability analysis.

.. moduleauthor:: Carlos Carreiras
"""


# Imports
# built-in

# 3rd party
import numpy as np
import scipy.signal as ss



def hrv(R, sampleRate, npoints=100000):
    """
    Compute Heart Rate Variability (HRV) metrics from a sequence of R peak locations.
    
    Kwargs:
        R (list, array): Positions of the R peaks in samples (from a segmentation algorithm).
        
        sampleRate (float): Sampling rate (Hz).
        
        npoints (int): Number of frequency points for the Lomb-Scargle periodogram. Default=100000.
    
    Kwrvals:
        time (array): Instantaneous heart rate time points.
        
        HR (array): Instantaneous heart rate.
        
        RR (array): Instantaneous RR intervals.
        
        mHR (float): Mean heart rate.
        
        SD (float): Standard deviation of heart rate.
        
        RMSSD (float): Root mean square of successive differences.
        
        HF (float): High frequency power (0.15 to 0.4 Hz).
        
        LF (float): Low frequency power (0.04 to 0.15 Hz).
        
        L2HF (float): Ratio of LF to HF power.
    
    See Also:
        
    
    Notes:
        Maybe deal with un/mis-detected R peaks
    
    Example:
        
    
    References:
        .. [1] 
    
    """
    
    # ensure array of floats
    R = np.array(R, dtype='float')
    
    if len(R) < 4:
        raise ValueError, "Not enough R peaks to perform computations."
    
    # ensure float
    Fs = float(sampleRate)
    
    # convert samples to time units
    R /= Fs
    
    # compute RR
    RR = np.diff(R)
    t = R[:-1] + RR / 2
    
    # convert to heart rate
    HR = 60. / RR
    
    # compute RMSSD
    dRR = np.diff(RR)
    
    # compute power metrics
    f = np.linspace(0.001, 1, npoints)
    w = 2 * np.pi * f
    pgram = ss.lombscargle(t, RR, w) * 4. / RR.shape[0]
    
    HF = np.sum(pgram[(0.15 < f) * (f < 0.4)])
    LF = np.sum(pgram[(0.04 < f) * (f < 0.15)])
    
    output = {'time': t,
              'HR': HR,
              'RR': RR,
              'mHR': np.mean(HR),
              'SD': np.std(HR, ddof=1),
              'RMSSD': np.sqrt(np.mean(dRR**2)),
              'HF': HF,
              'LF': LF,
              'L2HF': LF / HF,
              }
    
    return output

