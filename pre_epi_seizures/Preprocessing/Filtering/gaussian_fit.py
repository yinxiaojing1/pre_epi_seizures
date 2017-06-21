"""
Gaussian Nonlinear Least Squares Fit to each ECG beat.
@author: Afonso Eduardo
"""
from __future__ import division

import os

from datetime import datetime

import numpy as np
import scipy
import scipy.signal as ss
from scipy.optimize import least_squares
import pandas as pd
import matplotlib.pyplot as plt

def get_phase(x, peaks):
    """
    Compute ECG phase from a given set of R-peaks.
    Parameters:
    -----------
    x: array 1D
        Signal.
    peaks: array 1D
        R peak locations.
    Returns:
    --------
    array 1D
        Phase vector.
    """
    phase = np.zeros(len(x))
    for pb, pu in zip(peaks[:-1],peaks[1:]):
        phase[pb:pu] = np.linspace(0, 2*np.pi, pu - pb, endpoint=False)
    if peaks[0] > 0:
        phase[:peaks[0]] = np.linspace(0, 2*np.pi, max(peaks[0], peaks[1] - peaks[0]), endpoint=False)[-peaks[0]:]
    if peaks[-1] < len(x)-1:
        phase[peaks[-1]:] = np.linspace(0, 2*np.pi, max(len(x)-peaks[-1], peaks[-1] - peaks[-2]), endpoint=False)[:len(x)-peaks[-1]]
    phase = np.fmod(phase, 2*np.pi)
    phase[np.where(phase>np.pi)[0]] -= 2*np.pi
    return phase

def mean_extraction(x, phase, bins=250, align_baseline=False):
    """ECG Mean extractor"""
    mnphase, mnx, sdx = np.zeros(bins), np.zeros(bins), np.full(bins, -1, dtype=float)
    idx = np.where(np.logical_or(phase>=np.pi-np.pi/bins, phase<-np.pi+np.pi/bins))[0]
    if len(idx)>0:
        mnphase[0], mnx[0], sdx[0] = -np.pi, np.mean(x[idx]), np.std(x[idx], ddof=1)
    for i in range(1, bins):
         idx = np.where(np.logical_and(phase>=2*np.pi*(i-.5)/bins-np.pi, phase<2*np.pi*(i+.5)/bins-np.pi))[0]
         if len(idx)>0:
             mnphase[i], mnx[i], sdx[i] = np.mean(phase[idx]), np.mean(x[idx]), np.std(x[idx], ddof=1)
    for i in np.where(sdx==-1)[0]:
        if i==0:
            mnphase[0], mnx[0], sdx[0] = -np.pi, mnx[1], sdx[1]
        elif i==bins-1:
            mnphase[i], mnx[i], sdx[i] = np.pi, mnx[-2], sdx[-2]
        else:
            idx = np.array([i-1, i+1], dtype=int)
            mnphase[i], mnx[i], sdx[i] = map(np.mean, mnphase[idx], mnx[idx], sdx[idx])
    if align_baseline:
        mnx -= np.mean(mnx[:np.ceil(len(mnx)/10)])
    return mnx, sdx, mnphase

def ecg_model(values, phase):
    """ECG model based on gaussian kernels in the phase space"""
    ai, bi, thetai = np.split(values, 3)
    dthetai = np.fmod(np.pi + phase - thetai[:,None], 2*np.pi) - np.pi
    return np.sum(ai[:,None] * np.exp(-dthetai**2 / (2*bi**2)[:,None]), axis=0)
# Loss function to use when performing Nonlinear Least Squares Optimization
f_loss = lambda values, mnphase, x: x-ecg_model(values, mnphase)

def beat_fitter(x, phase, max_runs=3000, values0=None, bounds=None):
    """
    Automated beat fitter: compute the gaussian kernel parameters (ai,bi,thetai)
    using Nonlinear Least Squares Optimization.
    Parameters:
    -----------
    x: array 1D
       Amplitude of ECG beat.
    phase: array 1D
        Phase of ECG beat.
    max_runs: int (default: 3000)
        Maximum number of NLS runs.
    values0: array 1D (default: None)
        Initial guesses (ai, bi, thetai). If None, take default initialization.
    bounds: array 1D
        Bounds for (ai, bi, thetai). If None, take default bounds.
    Returns:
    --------
    array 1D
        Gaussian kernel parameters, 3*L with L being the number of gaussians.
    Notes:
    ------
    Be careful using this function: make sure the initial guesses and bounds
    make sense for the signal that is being applied on!
    In the default setting, the number of gaussians is 5.
    """
    ## Define initial guess for parameters
    if values0 is None:
    	bins = len(x)
        rloc = int(bins/2) # r is assumed to be at the center
        thetai = np.zeros(5) # phase loc
        thetai[0] = phase[int(.2*bins)+np.argmax(x[int(.2*bins):int(.45*bins)])]
        idx = int(.44*bins) + np.argmin(x[int(.44*bins):int(.5*bins)])
        thetai[1] = phase[idx]
        thetai[2] = phase[rloc]
        thetai[3] = phase[2*rloc - idx]
        thetai[4] = phase[int(5/8*bins)+np.argmax(x[int(5/8*bins):int(7/8*bins)])]
        bi = np.array([.1, .05, .05, .05, .2]) # width
        ai = np.zeros(5) # amplitude
        ai[0] = np.abs(np.max(x[int(.2*bins):int(.45*bins)]))
        ai[1] = -np.abs(np.min(x))
        ai[2] = np.abs(np.max(x))
        ai[3] = -np.abs(np.min(x))
        ai[4] = np.abs(np.max(x[int(5/8*bins):int(7/8*bins)]))
        values0 = np.hstack((ai, bi, thetai))
    ## Define bounds for parameters
    if bounds is None:
        steps = phase[1] - phase[0]
        bounds_ai = [np.array([0, -np.inf, 0, -np.inf, 0]), np.array([np.inf, 0, np.inf, 0, np.inf])]
        bounds_bi = [np.array([.1, .01, .01, 0.01, .1]), np.array([.5, .1, .2, .1, .5])]
        bounds_thetai = [np.maximum(-np.pi, thetai-steps*np.array([20,5,5,5,20])), np.minimum(np.pi, thetai+steps*np.array([20,5,5,5,20]))]
        bounds = np.hstack((bounds_ai, bounds_bi, bounds_thetai))
    ## Nonlinear Least Squares Optimization using Trust Region Reflective algorithm
    values = least_squares(f_loss, x0=values0, args=(phase, x), bounds=bounds, method='trf', max_nfev=max_runs)['x']
    return values

def gaussian_fit(x, rpeaks, bins=250, plot_params=True):
	"""
	Perform Gaussian Nonlinear Least Squares to each ECG beat. 
	Parameters:
	-----------
	x: array 1D
		Amplitude of ECG beat.
	rpeaks: array 1D
		Position of the R peaks.
	bins: int (default: 250)
		Number of bins to consider when computing the phase-wrapped ECG beat.
	plot_params: bool (default: True)
		If True, draw a plot showing the ECG beats and the respective kernel parameters.
	Returns:
	--------
	Kparam: array 2D
		Kernel parameters (ai, bi, thetai) of each beat (1 per row).
	xxbin: array 2D
		Amplitude of phase-wrapped ECG beats (1 per row).
	xxbinphase: array 2D
		Phase of the phase-wrapped ECG beats (1 per row).
	Notes:
	------
	This function assumes 5 Gaussian kernels per beat.
	"""
	PLOTXXI = False
	def plotxxi(): # intermediate plotting function that shows the Gaussian Fit
		ax1 = plt.subplot(211); plt.plot(xxi, 'k'); plt.plot(xxiphase*1/4)
		ax2 = plt.subplot(212); plt.plot(xxbin[i], 'k'); plt.plot(xxbinphase[i]*1/4)
		plt.plot(ecg_model(Kparam[i], xxbinphase[i]), 'g'); plt.show()

	xphase = get_phase(x, rpeaks)
	jumps = np.nonzero(np.hstack([False, np.diff(xphase) <= -1.8*np.pi]))[0]
	xx, xxphase = zip(*[(x[s:e], xphase[s:e]) for s, e in zip(jumps[:-1], jumps[1:])])
	
	Kparam = np.zeros((len(xx), 5*3))
	xxbin, xxbinphase = np.zeros((len(xx), bins)), np.zeros((len(xx), bins))
	for i, (xxi, xxiphase) in enumerate(zip(xx, xxphase)):
		xxbin[i], _, xxbinphase[i] = mean_extraction(xxi, xxiphase, bins=bins) # phase-wrapped ECG beat
		Kparam[i] = beat_fitter(xxbin[i], xxbinphase[i])
		if PLOTXXI: plotxxi()

  	if plot_params:
		kparams =  np.array(map(np.hstack, zip(*[np.split(np.repeat(Kparam[i], e-s), 3*5) for i, (s,e)
		                    in enumerate(zip(jumps[:-1], jumps[1:]))])))
		kidx = np.arange(jumps[0], jumps[-1])
	  	plt.figure(figsize=(19, 9.5))
		ax1 = plt.subplot(411); plt.plot(x, 'k'); plt.plot(rpeaks, x[rpeaks], 'ro'); plt.plot(xphase*1/4, 'b')
		[plt.axvline(j, color='0.75', lw=1, ls='--') for j in jumps]
		ax1.legend(['signal', 'peaks', '1/4*phase'], bbox_to_anchor=(1.1, 1))
		for i, param in enumerate(['a', 'b', 'theta']):
			axi = plt.subplot(4, 1, i+2, sharex=ax1); axi.plot(kidx, kparams[5*i:5*(i+1)].T)
			[axi.axvline(j, color='0.75', lw=1, ls='--') for j in jumps]
			axi.legend([param+wave for wave in ['P', 'Q', 'R', 'S', 'T']], bbox_to_anchor=(1.08, 1))
		plt.show()
	return Kparam, xxbin, xxbinphase



if __name__ == '__main__':
    import sys
    sys.path.append(os.path.abspath("../"))
    from preprocessing import read_R_csv, read_dataset_csv

    f_path = lambda datafile: os.path.abspath("..\..\{}\{}".format(datafolder, datafile))
    datafolder = 'Data'
    fs = 500. # sampling frequency [Hz]

    R_list = read_R_csv(f_path("ECGIDDB_raw_UNSW_Rpeaks.csv"))
    X, y = read_dataset_csv(f_path("ECGIDDB_medianFIR_UNSW.csv"), multicolumn=True)

    for i, (x, rpeaks) in enumerate(zip(X, R_list)):
    	gaussian_fit(x, rpeaks)