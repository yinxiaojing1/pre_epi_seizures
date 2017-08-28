"""
ECG smoothing based on Extended Kalman Smoother.
@author: Afonso Eduardo
"""
from __future__ import division

import os

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.signal as ss
from scipy.optimize import least_squares
import pandas as pd
import matplotlib.pyplot as plt

# import matlab
# import matlab.engine

def get_phase(x, peaks):
    """
    Compute ECG phase from a given set of R-peaks (see OSET).
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
    print 'here'
    phase = np.zeros(len(x))
    # plt.plot(peaks, 'o')
    # plt.show()
    for pb, pu in zip(peaks[:-1],peaks[1:]):
        phase[pb:pu] = np.linspace(0, 2*np.pi, pu - pb, endpoint=False)


    if peaks[0] > 0:
        phase[:peaks[0]] = np.linspace(0, 2*np.pi, max(peaks[0], peaks[1] - peaks[0]), endpoint=False)[-peaks[0]:]

    if peaks[-1] < len(x)-1:
        phase[peaks[-1]:] = np.linspace(0, 2*np.pi, max(len(x)-peaks[-1], peaks[-1] - peaks[-2]), endpoint=False)[:len(x)-peaks[-1]]
    phase = np.fmod(phase, 2*np.pi)
    phase[np.where(phase>np.pi)[0]] -= 2*np.pi
    return phase

def phase_shift(phase, theta):
    """Phase shifter (see OSET)"""
    return np.fmod(phase + theta + np.pi, 2*np.pi) - np.pi

def mean_extraction(x, phase, bins=250, align_baseline=False):
    """ECG Mean extractor (see OSET)"""
    mnphase, mnx, sdx = np.zeros(bins), np.zeros(bins), np.full(bins, -1, dtype=float)
    idx = np.where(np.logical_or(phase>=np.pi-np.pi/bins, phase<-np.pi+np.pi/bins))[0]
    print 'idx'
    print idx
    stop
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
    """ECG model based on gaussian kernels in the phase space (see OSET)"""
    ai, bi, thetai = np.split(values, 3)
    dthetai = np.fmod(np.pi + phase - thetai[:,None], 2*np.pi) - np.pi
    return np.sum(ai[:,None] * np.exp(-dthetai**2 / (2*bi**2)[:,None]), axis=0)
# Loss function to use when performing Nonlinear Least Squares Optimization
f_loss = lambda values, mnphase, x: x-ecg_model(values, mnphase)

def beat_fitter(x, phase, max_runs=10, values0=None, bounds=None):
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
    values = least_squares(f_loss, x0=values0, args=(phase, x), bounds=bounds, method='trf', max_nfev=max_runs, verbose=0)['x']
 
    return values

# def EKSmoother(Y, X0, P0, Q, R, Wmean, Vmean, Inits, InovWlen=250, tau=None, gamma=1., RadaptWlen=250):
#     """
#     Wrapper of EKSmoother (matlab function - OSET).
#     Parameters:
#     -----------
#     Y: array 2D [2 x samples]
#         Observation signals with first row being the phase and the second the amplitude of noisy ECG.
#     X0: array 1D [2]
#         Initial state vector.
#     P0: array 2D [2 x 2]
#         Covariance matrix of the initial state vector.
#     Q: array 2D [(3L+2) x (3L+2)]
#         Covariance matrix of the process noise vector, with L being the number of gaussian kernels.
#     R: array 2D [2 x 2]
#         Covariance matrix of the observation noise vector.
#     Wmean: array 1D [3L+2]
#         Mean process noise vector.
#     Vmean: array 1D [2]
#         Mean observation noise vector.
#     Inits: array 1D [3L+2]
#         Filter initialization parameters: [ai, bi, thetai, w, fs].
#     InovWlen: int (default: 250, assuming fs of 500Hz)
#         Innovations monitoring window length.
#     tau: float (default: None)
#         Kalman filter forgetting time. No forgetting factor when tau is None.
#     gamma: float (default: 1.)
#         Observation covariance adaptation rate. 0.<gamma<1. and gamma=1. for no adaptation.
#     RadaptWlen: int (default: 250, assuming fs of 500Hz)
#         Window length for observation covariance adaptation.
#     Returns:
#     --------
#     Xekf: array 2D [2 x samples]
#         State vectors estimated by EKF with first row being the estimated phase and the second the
#         denoised ECG.
#     Pekf: array 3D [2 x 2 x samples]
#         EKF state vector covariance matrices.
#     Xeks: array 2D [2 x samples]
#         State vectors estimated by EKS with first row being the estimated phase and the second the
#         denoised ECG.
#     Peks: array 3D [2 x 2 x samples]
#         EKS state vector covariance matrices.
#     a: array 2D [2 x samples]
#         Measure of innovations signal whiteness.
#     Notes:
#     ------
#     The original documentation of this function in matlab (OSET) seems to contain errors regarding input
#     and output parameters specifications such as the Y or Xekf dimensions being [samples x 2] when it
#     should read [2 x samples].
#     References:
#     -----------
#     R. Sameni, A Nonlinear Bayesian Filtering Framework for ECG Denoising, 2007.
#     """
#     meng = matlab.engine.start_matlab("-nojvm")
#     meng.addpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"OSET"), nargout=0)
#     Xekf, Pekf, Xeks, Peks, a = map(lambda x: np.array(x),
#                                 meng.EKSmoother(*map(lambda x: x if not isinstance(x, np.ndarray)\
#                                 else matlab.double(x.tolist()), [Y, X0[:,None], P0, Q, R, Wmean,
#                                 Vmean[:,None], Inits, InovWlen, (matlab.double([]) if tau is None\
#                                 else float(tau)), float(gamma), RadaptWlen, 0]), nargout=5))
#     meng.quit()
#     return Xekf, Pekf, Xeks, Peks, a


def _EKSmoother(Y, X0, P0, Q, R, Wmean, Vmean, Inits, InovWlen=250, tau=None, gamma=1., RadaptWlen=250):
    """
    Python implmentation of EKSmoother.
    Parameters:
    -----------
    Y: array 2D [2 x samples]
        Observation signals with first row being the phase and the second the amplitude of noisy ECG.
    X0: array 1D [2]
        Initial state vector.
    P0: array 2D [2 x 2]
        Covariance matrix of the initial state vector.
    Q: array 2D [(3L+2) x (3L+2)]
        Covariance matrix of the process noise vector, with L being the number of gaussian kernels.
    R: array 2D [2 x 2]
        Covariance matrix of the observation noise vector.
    Wmean: array 1D [3L+2]
        Mean process noise vector.
    Vmean: array 1D [2]
        Mean observation noise vector.
    Inits: array 1D [3L+2]
        Filter initialization parameters: [ai, bi, thetai, w, fs].
    InovWlen: int (default: 250, assuming fs of 500Hz)
        Innovations monitoring window length.
    tau: float (default: None)
        Kalman filter forgetting time. No forgetting factor when tau is None.
    gamma: float (default: 1.)
        Observation covariance adaptation rate. 0.<gamma<1. and gamma=1. for no adaptation.
    RadaptWlen: int (default: 250, assuming fs of 500Hz)
        Window length for observation covariance adaptation.
    Returns:
    --------
    Xekf: array 2D [2 x samples]
        State vectors estimated by EKF with first row being the estimated phase and the second the
        denoised ECG.
    Pekf: array 3D [2 x 2 x samples]
        EKF state vector covariance matrices.
    Xeks: array 2D [2 x samples]
        State vectors estimated by EKS with first row being the estimated phase and the second the
        denoised ECG.
    Peks: array 3D [2 x 2 x samples]
        EKS state vector covariance matrices.
    a: array 2D [2 x samples]
        Measure of innovations signal whiteness.
    References:
    -----------
    R. Sameni, A Nonlinear Bayesian Filtering Framework for ECG Denoising, 2007.
    """
    ## Auxiliary functions
    Observation = lambda x, v: x+v
    def State(x):
        phase = x[0] + w*dt
        if phase>np.pi:
            phase -= 2*np.pi
        dthetai = np.fmod(phase - thetai, 2*np.pi)
        z = x[1] - dt*np.sum(w*alphai/(bi**2)*dthetai*np.exp(-.5*(dthetai/bi)**2))
        return np.array([phase, z])
    Linearize_Obs = lambda: (np.eye(2),)*2
    def Linearize_State(x):
        dthetai = np.fmod(x[0] - thetai, 2*np.pi)
        M = np.eye(2)
        M[1,0] = -dt*np.sum(w*alphai/(bi**2)*(1-(dthetai/bi)**2)*np.exp(-.5*(dthetai/bi)**2)) # dF2/dphase
        N = np.zeros((2, 3*kernels+2))
        N[0, 3*kernels] = dt
        N[1, :kernels] = -dt*w/(bi**2)*dthetai*np.exp(-.5*(dthetai/bi)**2)
        N[1, kernels:2*kernels] = 2*dt*alphai*w*dthetai/bi**3*(1-.5*(dthetai/bi)**2)*np.exp(-.5*(dthetai/bi)**2)
        N[1, 2*kernels:3*kernels] = dt*w*alphai/(bi**2)*(1-(dthetai/bi)**2)*np.exp(-.5*(dthetai/bi)**2)
        N[1, 3*kernels] = -np.sum(dt*alphai*dthetai/(bi**2)*np.exp(-.5*(dthetai/bi)**2))
        N[1, -1] = 1
        return M, N
    ## Initializations
    (alphai, bi, thetai), (w, fs) = np.split(Inits[:-2],3), Inits[-2:]
    dt, kernels = 1/fs, len(alphai)
    samples, states = Y.shape[1], len(X0)
    Pminus, Xminus = P0, X0
    Xpred, Ppred = np.zeros((states, samples)), np.zeros((states, states, samples))
    Xekf, Pekf = np.zeros((states, samples)), np.zeros((states, states, samples))
    # Innovation monitoring
    mem1, mem2 = np.ones((Y.shape[0], InovWlen)), np.zeros((Y.shape[0], RadaptWlen)) + R[1,1]
    a = np.zeros((Y.shape[0],samples))
    # Forgetting factor - based on tau
    alpha = 1 if tau is None else np.exp(-dt/tau)
    ## Filtering
    for k in range(samples):
        # Prevent 'Xminus' miscalculations on phase jumps
        if np.abs(Xminus[0]-Y[0,k]) > np.pi:
            Xminus[0] = Y[0,k]
        # Store predicted mean and cov
        Xpred[:, k] = Xminus
        Ppred[:,:,k] = Pminus
        # Measurement update
        XX = Xminus
        PP = Pminus
        for jj in range(Y.shape[0]):
            Yminus = Observation(XX, Vmean)
            CC, GG = Linearize_Obs()
            YY, C, G = Yminus[jj], np.atleast_2d(CC[jj, :]), np.atleast_2d(GG[jj, :])
            K = PP.dot(C.T)/(C.dot(PP.dot(C.T)) + alpha*np.dot(G*R[jj,jj], G.T)) # Kalman Gain
            temp = np.eye(states)-np.dot(K,C)
            PP = (temp.dot(PP).dot(temp.T) + (np.dot(K,G)*R[jj,jj]).dot(np.dot(G.T, K.T)))/alpha # cov update (Joseph form)
            XX = XX + (K*(Y[jj,k]-YY)).ravel() # mean update
        Xplus = XX
        Pplus = (PP + PP.T)/2
        # Store filtered mean and cov
        Xekf[:, k] = Xplus
        Pekf[:,:,k] = Pplus
        # Monitoring the innovation variance
        inovk = Y[:, k] - Yminus
        Yk = np.dot(C, Pminus).dot(C.T) + np.dot(G, R).dot(G.T)
        mem1[:,1:], mem2[:,1:] = mem1[:, :-1], mem2[:, :-1]
        mem1[:,0] = inovk**2/Yk
        mem2[:,0] = inovk**2
        a[:, k] = np.mean(mem1, 1)
        R[1,1] = gamma*R[1,1] + (1-gamma)*np.mean(mem2[:,1])
        # Prediction update
        Xminus = State(Xplus)
        A, F = Linearize_State(Xplus)
        Pminus = A.dot(Pplus).dot(A.T) + F.dot(Q).dot(F.T)
    ## Smoothing
    Xeks, Peks = np.zeros_like(Xekf), np.zeros_like(Pekf)
    Xeks[:,-1], Peks[:,:,-1] = Xekf[:,-1], Pekf[:,:,-1]
    for k in reversed(range(samples-1)):
        A = Linearize_State(Xekf[:,k])[0]
        S = np.dot(Pekf[:,:,k], A.T).dot(np.linalg.inv(Ppred[:,:,k+1]))
        Xeks[:,k] = Xekf[:,k] + np.dot(S, (Xeks[:,k+1] - Xpred[:,k+1])[:,None]).ravel()
        Peks[:,:,k] = Pekf[:,:,k] + np.dot(S, (Peks[:,:,k+1] - Ppred[:,:,k+1])).dot(S.T)
    return Xekf, Pekf, Xeks, Peks, a


## Main function
def EKSmoothing(X, R_list, fs=500., bins=250, verbose=False, oset=False, savefolder=None):
    """
    ECG smoothing based on Extended Kalman Smoother (see OSET).
    Parameters:
    -----------
    X: array 1D or 2D
        Signals (1 per row).
    R_list: list of arrays
        Each array contains the position of R peaks
        (make sure to correctly identify these peaks beforehand).
    fs: float (default: 500)
        Sampling frequency [Hz].
    bins: int (default: 250)
        Number of bins to consider when computing the mean ECG
        in phase space.
    verbose: bool (default: False)
        If True, indicate the progress via stdout.
    oset: bool (default: False)
        If True, call EKSmoother from matlab (OSET). Otherwise, use
        python implementation.
    savefolder: str (default: None)
        Provide a folder to save plots with the input signals, 
        corresponding gaussian least squares fit and smoothed signals.
        If None, this feature is disabled.
    Returns:
    --------
    array 2D
        Smoothed signals.
    References:
    -----------
    R. Sameni, A Nonlinear Bayesian Filtering Framework for ECG Denoising, 2007.
    """
    X = np.atleast_2d(X)
    # print 'X'
    # print X
    Xeks_out = np.full(X.shape, -1, dtype=float)
    # print 'Xeks_out'#***
    # print Xeks_out#***
    # # stop
    for i, (x, rpeaks) in enumerate(zip(X, R_list)):
        if verbose: print ("[{}] - {} / {} ...".format(str(datetime.now())[:-7], i+1, len(X))),
        # print x #***
        # plt.plot(x) #***
        # # stop
        phase = get_phase(x, rpeaks)
        # print 'phase' #***
        # plt.plot(phase) #***
        # plt.show() #****
        mnx, sdx, mnphase = mean_extraction(x, phase, bins=bins)
        # plt.plot(mnx) #***
        # plt.show() #****
        values = beat_fitter(mnx, mnphase)
        N = int(len(values)/3) # number of gaussian kernels
        fm = fs / np.diff(rpeaks) # heart rate
        w, wsd = np.mean(2*np.pi*fm), np.std(2*np.pi*fm, ddof=0) # heart rate [rads] - mean and std (normalized by N-ddof)
        X0 = np.array([-np.pi, 0])
        P0 = np.array([[(2*np.pi)**2, 0],[0, (10*np.max(np.abs(x)))**2]])
        Q  = np.diag(np.hstack([(.1*values[:N])**2, (.05*np.ones(N))**2, (.05*np.ones(N))**2, wsd**2, (.05*np.mean(sdx[:int(len(sdx)/10)]))**2]))
        R = np.array([[(w/fs)**2/12, 0], [0, np.mean(sdx[:int(len(sdx)/10)])**2]])
        Wmean = np.hstack([values, w, 0])
        Vmean = np.zeros(2)
        Inits = np.hstack([values, w, fs])
        InovWlen = int(np.ceil(.5*fs)) # innovations monitoring window length
        tau = None # Kalman filter forgetting time. tau=None for no forgetting factor
        gamma = 1. # observation covariance adaptation rate. 0<gamma<1 and gamma=1 for no adaptation
        RadaptWlen = int(np.ceil(.5*fs)) # window length for observation covariance adaptation
        Y = np.stack([phase, x])
        if not oset:
            Xekf, Pekf, Xeks, Peks, ak = _EKSmoother(Y,X0,P0,Q,R,Wmean,Vmean,Inits,InovWlen,tau,gamma,RadaptWlen)
        else:
            Xekf, Pekf, Xeks, Peks, ak = EKSmoother(Y,X0,P0,Q,R,Wmean,Vmean,Inits,InovWlen,tau,gamma,RadaptWlen)
        Xeks_out[i,:] = Xeks[1,:]
        if savefolder is not None:
            if not os.path.exists(savefolder):
                os.makedirs(savefolder)
            plt.figure(figsize=(20, 8))
            ax1 = plt.subplot(3, 1, 1); ax1.plot(x, 'k', linewidth=2, label='Raw'); ax1.plot(rpeaks, x[rpeaks], 'ro', label='Peaks')
            ax1.plot(phase/4, 'b', linewidth=2, label='Phase*1/4'); ax1.legend(bbox_to_anchor=(1.12, 1.))
            ax2 = plt.subplot(3, 1, 2); ax2.plot(mnx, 'k', linewidth=2, label='Mean')
            ax2.fill_between(np.arange(bins), mnx-3*sdx, mnx+3*sdx, alpha=.2, edgecolor='k', facecolor='k', label='Mean+-3*SD')
            ax2.plot(ecg_model(values, mnphase), 'g', linewidth=2, label='Gaussian LS fit'); ax2.legend(bbox_to_anchor=(1.13, 1.))
            ax3 = plt.subplot(3, 1, 3, sharex=ax1); ax3.plot(Xeks_out[i, :], 'g', linewidth=1, label='Smoothed'); ax3.legend(bbox_to_anchor=(1.12, 1.))
            ax3.fill_between(np.arange(len(Xeks_out[i,:])), Xeks_out[i,:]-3*np.sqrt(Peks[1,1,:]), Xeks_out[i,:]+3*np.sqrt(Peks[1,1,:]), alpha=.2, edgecolor='g',
            facecolor='g')
            plt.savefig(os.path.join(savefolder, '{}.png'.format(i)), dpi=350)
            plt.close()
        if verbose: print "Done!"
    return Xeks_out
