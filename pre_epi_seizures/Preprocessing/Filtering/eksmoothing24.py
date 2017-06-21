"""
ECG smoothing based on Extended Kalman Smoother (24 state variant).
INCOMPLETE DO NOT USE - IMPLEMENTATION IN PROGRESS
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

from eksmoothing import get_phase, phase_shift, mean_extraction,
                        ecg_model, f_loss
from eksmoothing import beat_fitter # redefine beat_fitter in this module if necessary

def EKSmoother24(Y, X0, P0, Q, R, Wmean, Vmean, Inits, InovWlen=250, tau=None, gamma=1., RadaptWlen=250):
    """
    Extended Kalman Smoother (24 state variant).
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
    M. Akhbari, ECG denoising and fiducial point extraction using an extended Kalman filtering framework with
    linear and nonlinear phase observations, 2016.
    R. Sameni, A Nonlinear Bayesian Filtering Framework for ECG Denoising, 2007.
    """
    #######################################################################################
    # TO BE MODIFIED
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
    #######################################################################################
    ## Initializations
    (alphai, bi, thetai), (w, fs) = np.split(Inits[:-2],3), Inits[-2:]
    dt, kernels = 1/fs, len(alphai)
    samples, states = Y.shape[1], len(X0)
    Pminus, Xminus = P0, X0
    Xpred, Ppred = np.zeros((states, samples)), np.zeros((states, states, samples))
    Xekf, Pekf = np.zeros((states, samples)), np.zeros((states, states, samples))
    # Innovation monitoring
    mem1, mem2 = np.ones((Y.shape[0], InovWlen)), np.zeros((Y.shape[0], RadaptWlen)) + R[1,1]
    a = np.zeros((states,samples))
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
        for jj in range(states):
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


def wv_windows(phase, bP=[-np.pi, -np.pi/6], bQRS=[-np.pi/6, np.pi/6], bT=[np.pi/6, np.pi], gamma=30):
    wnd_f = lambda phase, b: 1/(1+np.exp(-gamma*(phase-b[0]))) - 1/(1+np.exp(-gamma*(phase-b[1])))
    return wnd_f(phase, bP), wnd_f(phase, bQRS), wnd_f(phase, bT)


## Main function
def EKSmoothing24(X, R_list, fs=500., bins=250, verbose=False, savefolder=None):
    """
    ECG smoothing based on Extended Kalman Smoother (24 state variant).
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
    M. Akhbari, ECG denoising and fiducial point extraction using an extended Kalman filtering framework with
    linear and nonlinear phase observations, 2016.
    M. Akhbari, Fiducial Points Extraction and Characteristic Waves Detection in ECG Signal using a model-based
    Bayesian Framework, 2013.
    """
    X = np.atleast_2d(X)
    Xeks_out = np.full(X.shape, -1, dtype=float)
    for i, (x, rpeaks) in enumerate(zip(X, R_list)):
        if verbose: print ("[{}] - {} / {} ...".format(str(datetime.now())[:-7], i+1, len(X))),
        phase = get_phase(x, rpeaks)
        ###############################
        print i
        plt.figure(figsize=(20, 8))
        ax1 = plt.subplot(4, 1, 1); ax1.plot(x, 'k', linewidth=2, label='Raw'); ax1.plot(rpeaks, x[rpeaks], 'ro', label='Peaks')
        wnd_P, wnd_QRS, wnd_T = wv_windows(phase)
        ax2 = plt.subplot(4, 1, 2, sharex=ax1, sharey=ax1); ax2.plot(wnd_P*x, 'k')
        ax3 = plt.subplot(4, 1, 3, sharex=ax1, sharey=ax1); ax3.plot(wnd_QRS*x, 'k')
        ax4 = plt.subplot(4, 1, 4, sharex=ax1, sharey=ax1); ax4.plot(wnd_T*x, 'k')
        plt.savefig(os.path.join(os.path.abspath("../../../Data/ECGIDDB_windowing_test"), '{}.png'.format(i)), dpi=350)
        plt.close()
        continue
        print "===================="
        asdasdsa
        ###############################
        mnx, sdx, mnphase = mean_extraction(x, phase, bins=bins)
        values = beat_fitter(mnx, mnphase)
        N = int(len(values)/3) # number of gaussian kernels
        fm = fs / np.diff(rpeaks) # heart rate
        w, wsd = np.mean(2*np.pi*fm), np.std(2*np.pi*fm, ddof=0) # heart rate [rads] - mean and std (normalized by N-ddof)
        #######################################################################################
        # TO BE MODIFIED
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
        #######################################################################################
        Xekf, Pekf, Xeks, Peks, ak = EKSmoother24(Y,X0,P0,Q,R,Wmean,Vmean,Inits,InovWlen,tau,gamma,RadaptWlen)
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


if __name__ == '__main__':
    import sys
    sys.path.append(os.path.abspath("../"))
    from preprocessing import read_R_csv, read_dataset_csv

    f_path = lambda datafile: os.path.abspath("..\..\{}\{}".format(datafolder, datafile))
    datafolder = 'Data'
    fs = 500. # sampling frequency [Hz]

    R_list = read_R_csv(f_path("ECGIDDB_raw_UNSW_Rpeaks.csv"))
    X, y = read_dataset_csv(f_path("ECGIDDB_medianFIR_UNSW.csv"), multicolumn=True)
    #n_rec = 15
    #x, r, yy = X[n_rec], R_list[n_rec], y[n_rec]
    #xeks = EKSmoothing(x, [r], fs=500., bins=250, verbose=True)
    #plt.plot(x, 'k')
    #plt.plot(r, x[r], 'ro')
    #plt.show()

    EKSmoothing24(X, R_list, fs=500., bins=250, verbose=True)