"""
ECG smoothing based on Extended Kalman Smoother. In this version the dynamical model does not
depend on the amplitude of the Gaussian kernels.
IMPLEMENTATION IN PROGRESS !!!!
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

from eksmoothing import get_phase, phase_shift, mean_extraction, ecg_model, f_loss
from eksmoothing import beat_fitter # redefine beat_fitter in this module if necessary (e.g. initializations, bounds or number of kernels)


def EKSmoother6(Y, X0, P0, Vmean, R, Wmean, Q, Inits, InovWlen=250, tau=None, gamma=1., RadaptWlen=250):
    """
    Extended Kalman Smoother (6 state variant for gaussian kernels=5, i.e. L=5). In general, 2+3L states
    with L being the number of gaussian kernels. In this version the dynamical model does not
    depend on the amplitude of the Gaussian kernels.
    Parameters:
    -----------
    Y: array 2D [2 x samples]
        Observation signals with first row being the phase and the second the amplitude (z) of noisy ECG.
    X0: array 1D [1+L]
        Mean initial state vector, with L being the number of gaussian kernels: phase, fis (basis functions).
    P0: array 2D [(1+L) x (1+L)]
        Initial state covariance matrix.
    Vmean: array 1D [2]
        Mean observation noise vector: phase, z.
    R: array 2D [2 x 2]
        Observation noise covariance matrix.
    Wmean: array 1D [3L+1]
        Mean process noise vector: betas, thetas, etas, w.
    Q: array 2D [(3L+1) x (3L+1)]
        Process noise covariance matrix.
    Inits: array 1D [2]
        Filter initialization parameters: w, fs.
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
    Xekf: array 2D [(1+L) x samples]
        EKF mean state vectors.
    Pekf: array 3D [(1+L) x (1+L) x samples]
        EKF state vector covariance matrices.
    Xeks: array 2D [(1+L) x samples]
        EKS mean state vectors.
    Peks: array 3D [(1+L) x (1+L) x samples]
        EKS state vector covariance matrices.
    a: array 2D [2 x samples]
        Measure of innovations signal whiteness.
    References:
    -----------
    E. Roonizi, A Signal Decomposition Model-Based Bayesian Framework for ECG Components Separation, 2016.
    R. Sameni, A Nonlinear Bayesian Filtering Framework for ECG Denoising, 2007.
    Notes:
    ------
    I believe the implementation of the propagation and linearization equations is correct, but it is not a guarantee.
    """
    SHOW_EK = True # internal plotting
    ## Auxiliary internal plot function
    def plot_signal(X, name=''):
        plt.figure(figsize=(20, 8))
        ax1 = plt.subplot(2, 1, 1); ax1.plot(Y[1,:], 'k', linewidth=2, label='Raw'); ax1.legend(bbox_to_anchor=(1.11, 1.))
        ax2 = plt.subplot(2, 1, 2, sharex=ax1); ax2.plot(np.sum(X[1:,:], axis=0), 'k', linewidth=2, label=name); ax2.legend(bbox_to_anchor=(1.11, 1.))
        #ax3 = plt.subplot(5, 1, 3, sharex=ax1); ax3.plot(X[2:2+5,:].T, linewidth=2); ax3.legend(['aP', 'aQ', 'aR', 'aS', 'aT'], bbox_to_anchor=(1.11, 1.))
        #ax4 = plt.subplot(5, 1, 4, sharex=ax1); ax4.plot(X[2+5:2+10,:].T, linewidth=2); ax4.legend(['bP', 'bQ', 'bR', 'bS', 'bT'], bbox_to_anchor=(1.11, 1.))
        #ax5 = plt.subplot(5, 1, 5, sharex=ax1); ax5.plot(X[12:,:].T, linewidth=2); ax5.legend(['tP', 'tQ', 'tR', 'tS', 'tT'], bbox_to_anchor=(1.11, 1.))
        plt.show()
    ## Auxiliary variables and functions - propagation and linearization equations
    def Observation(x, Vmean):
        return np.array([x[0], np.sum(x[1:])]) + Vmean
    def State(x):
        phase = x[0] + w*dt
        if phase>np.pi:
            phase -= 2*np.pi
        dthetai = np.fmod(phase - thetai, 2*np.pi)
        return np.hstack([phase, (1-w*dt*dthetai/bi**2)*x[1:]])
    def Linearize_Obs():
        M = np.zeros((2, 1+kernels)); M[0,0] = 1.; M[1,1:] = 1.
        N = np.eye(2)
        return M, N
    def Linearize_State(x):
        dthetai = np.fmod(x[0] - thetai, 2*np.pi)
        M = np.eye(1+kernels)
        M[1:,0] = -w*dt/bi**2*x[1:]
        M[1:,1:] = np.diag(1-w*dt*dthetai/bi**2)
        N = np.zeros((1+kernels, 3*kernels+1))
        N[0,-1] = dt 
        N[1:,:kernels] = np.diag(2*w*dt*dthetai/bi**3*x[1:])
        N[1:,kernels:2*kernels] = np.diag(w*dt/bi**2*x[1:])
        N[1:, 2*kernels:3*kernels] = np.eye(kernels)
        N[1:, -1] = -dt*dthetai/bi**2*x[1:]
        return M, N
    ## Initializations
    w, fs = Inits
    bi, thetai, etai = np.split(Wmean[:-1], 3)
    dt, kernels = 1/fs, len(X0)-1
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
        S = np.dot(Pekf[:,:,k], A.T).dot(np.linalg.pinv(Ppred[:,:,k+1])) # changed to pseudoinverse
        Xeks[:,k] = Xekf[:,k] + np.dot(S, (Xeks[:,k+1] - Xpred[:,k+1])[:,None]).ravel()
        Peks[:,:,k] = Pekf[:,:,k] + np.dot(S, (Peks[:,:,k+1] - Ppred[:,:,k+1])).dot(S.T)
    if SHOW_EK:
        plot_signal(Xekf, 'Filtered')
        plot_signal(Xeks, 'Smoothed')
    return Xekf, Pekf, Xeks, Peks, a

## Main function
def EKSmoothing6(X, R_list, fs=500., bins=250, verbose=False, savefolder=None):
    """
    ECG smoothing based on Extended Kalman Smoother: 6 state variant for gaussian kernel=5.
    In general, 2+3L states with L being the number of gaussian kernels.  In this version 
    the dynamical model does not depend on the amplitude of the Gaussian kernels.
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
    E. Roonizi, A Signal Decomposition Model-Based Bayesian Framework for ECG Components Separation, 2016.
    Notes:
    ------
    Currently, some initializations are not the same as in the paper, namely P0 and Q. They are proposals of my own
    based on tests done on signals from ECGIDDB.
    """ 
    X = np.atleast_2d(X)
    Zeks = np.full(X.shape, -1, dtype=float)
    for i, (x, rpeaks) in enumerate(zip(X, R_list)):
        if verbose: print ("[{}] - {} / {} ...".format(str(datetime.now())[:-7], i+1, len(X))),
        phase = get_phase(x, rpeaks)
        mnx, sdx, mnphase = mean_extraction(x, phase, bins=bins)
        values = beat_fitter(mnx, mnphase)
        #values2 = beat_fitter(mnx+np.sign(mnx)*sdx, mnphase)
        N = int(len(values)/3) # number of gaussian kernels
        fm = fs / np.diff(rpeaks) # heart rate
        s_b = np.mean(np.diff(rpeaks)) # average samples per beat
        w, wsd = np.mean(2*np.pi*fm), np.std(2*np.pi*fm, ddof=0) # heart rate [rads] - mean and std (normalized by N-ddof)
        Y = np.stack([phase, x]) # (noisy) observations: phase, z
        X0 = np.hstack([-np.pi, np.ones(N)]) # init state mean: phase, fis (basis functions)
        P0 = np.diag(np.hstack([2*np.pi, [10*np.max(np.abs(x))/N,]*N ])**2) # init covariance
        Vmean = np.zeros(2) # observation noise mean: phase, z
        R = np.diag(np.hstack([w/(np.sqrt(12)*fs), np.mean(sdx[:int(len(sdx)/10)])])**2) # observation noise covariance: phase, z
        Wmean = np.hstack([values[N:], np.zeros(N), w]) # process noise mean: betas, thetas, etas, w
        Q  = np.diag(np.hstack([1e-2*values[N:2*N], 1e-2*values[2*N:], [.4*np.mean(sdx[:int(len(sdx)/10)]),]*N, wsd])**2) # experiment
        #Q = np.diag(np.hstack([wsd, .05*np.mean(sdx[:int(len(sdx)/10)]), np.abs(values - values2)])) # my interpretation of what's in the paper (doesn't seem to work well, but implementation of model seems correct?)
            # process noise covariance
        Inits = np.hstack([w, fs]) # w, fs
        InovWlen = int(np.ceil(.5*fs)) # innovations monitoring window length
        tau = None # Kalman filter forgetting time. tau=None for no forgetting factor
        gamma = 1. # observation covariance adaptation rate. 0<gamma<1 and gamma=1 for no adaptation
        RadaptWlen = int(np.ceil(.5*fs)) # window length for observation covariance adaptation
        Xekf, Pekf, Xeks, Peks, ak = EKSmoother6(Y,X0,P0,Vmean,R,Wmean,Q,Inits,InovWlen,tau,gamma,RadaptWlen)
        Zeks[i,:] = np.sum(Xeks[1:,:], axis=0) # z extraction
        if savefolder is not None:
            if not os.path.exists(savefolder):
                os.makedirs(savefolder)
            plt.figure(figsize=(20, 8))
            ax1 = plt.subplot(3, 1, 1); ax1.plot(x, 'k', linewidth=2, label='Raw'); ax1.plot(rpeaks, x[rpeaks], 'ro', label='Peaks')
            ax1.plot(phase/4, 'b', linewidth=2, label='Phase*1/4'); ax1.legend(bbox_to_anchor=(1.12, 1.))
            ax2 = plt.subplot(3, 1, 2); ax2.plot(mnx, 'k', linewidth=2, label='Mean')
            ax2.fill_between(np.arange(bins), mnx-3*sdx, mnx+3*sdx, alpha=.2, edgecolor='k', facecolor='k', label='Mean+-3*SD')
            ax2.plot(ecg_model(values, mnphase), 'g', linewidth=2, label='Gaussian LS fit'); ax2.legend(bbox_to_anchor=(1.13, 1.))
            ax3 = plt.subplot(3, 1, 3, sharex=ax1); ax3.plot(Zeks[i, :], 'g', linewidth=1, label='Smoothed'); ax3.legend(bbox_to_anchor=(1.12, 1.))
            plt.savefig(os.path.join(savefolder, '{}.png'.format(i)), dpi=350)
            plt.close()
        if verbose: print "Done!"
    return Zeks


if __name__ == '__main__':
    import sys
    sys.path.append(os.path.abspath("../"))
    from preprocessing import read_R_csv, read_dataset_csv

    f_path = lambda datafile: os.path.abspath("..\..\{}\{}".format(datafolder, datafile))
    datafolder = 'Data'
    fs = 500. # sampling frequency [Hz]

    R_list = read_R_csv(f_path("ECGIDDB_raw_UNSW_Rpeaks.csv"))
    X, y = read_dataset_csv(f_path("ECGIDDB_medianFIR_UNSW.csv"), multicolumn=True)
    n_rec = 0
    x, r, yy = X[n_rec], R_list[n_rec], y[n_rec]
    xeks = EKSmoothing6(x, [r], fs=500., bins=250, verbose=True)
    #EKSmoothing6(X, R_list, fs=500., bins=250, verbose=True, savefolder=f_path("ECGIDDB_EKSmoothing17_images"))
