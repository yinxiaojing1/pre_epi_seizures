"""
EEG functions
"""

import numpy as np
import pylab as pl
import scipy.signal as ss
from scipy.stats import stats
from scipy import unwrap
from scipy import interpolate
from sklearn.decomposition import FastICA
from collections import deque
import pandas
# import sys
# sys.path.append("../")
from filt import windowfcn



def analyticSignal(signal=None, axis=-1):
	# to compute analytic signal and return amplitude and phase
	
	# check inputs
	if signal is None:
		raise TypeError, "A signal must be provided."
	
	# hilbert transform
	asig = ss.hilbert(signal, axis=axis)
	
	# amplitude envelope
	amp = np.absolute(asig)
	
	# instanteneous phase
	phase = np.angle(asig)
	
	# kwrvals
	kwrvals = {}
	kwrvals['aSignal'] = asig
	kwrvals['amplitude'] = amp
	kwrvals['phase'] = phase
	
	return kwrvals


def plf(signal=None, pairList=None):
	# to compute PLF between the specified pairs
	
	# check inputs
	if signal is None:
		raise TypeError, "A signal must be provided."
	if pairList is None:
		raise TypeError, "A list of pairs (tuples) must be provided."
	
	# compute analytic signal
	phase = analyticSignal(signal)['phase']
	
	# compute PLF
	plf = []
	for item in pairList:
		plf.append(np.absolute(np.mean(np.exp(1j * (phase[item[0], :] - phase[item[1], :])))))
	
	return np.array(plf)


def syncLikelihood(signal=None):
	
	# don't know how...
	
	return None


def power(signal=None, NFFT=4096, axis=-1):
	# compute power spectrum
	
	# check inputs
	if signal is None:
		raise TypeError, "A signal must be provided."
		
	# parameters
	length = signal.shape[axis]
	
	# fft
	pwr = np.fft.fft(signal, NFFT, axis)/NFFT
	pwr = 2 * np.absolute(pwr[:, 0:NFFT/2])
	
	return pwr


def bandPower(signal=None, Fs=1, lower=0, upper=1, NFFT=4096, axis=-1):
	# compute mean signal power in given band
	
	# check inputs
	if signal is None:
		raise TypeError, "A signal must be provided."
	
	# get power
	pwr = power(signal, NFFT, axis)
	
	# frequencies
	W = np.linspace(0, Fs/2, NFFT/2)
	
	# select
	aux = (W >= lower) & (W < upper)
	mpwr = np.mean(pwr[aux, :], axis=0)
	
	return mpwr


def laplacian(signal=None, indList=None):
	# change reference to Laplacian
	
	# check inputs
	if signal is None:
		raise TypeError, "A signal must be provided."
	if indList is None:
		raise TypeError, "An index list must be provided."
	
	length = signal.shape[-1]
	lap = []
	# get the laplacian
	for n in range(len(indList)):
		if not (indList[n] is ()):
			avg = np.mean(signal[indList[n], :], axis=0).reshape((1, length))
			lap.append(signal[n, :] - avg)
	
	# kwrvals
	kwrvals = {}
	kwrvals['signal'] = np.array(lap)
	
	return kwrvals


def car(signal=None):
	# change reference to Common Average Reference
	
	# check inputs
	if signal is None:
		raise TypeError, "A signal must be provided."
	
	N = signal.shape[0]
	length = signal.shape[1]
	avg = np.dot(np.ones((N, 1)), np.mean(signal, axis=0).reshape((1, length)))
	
	# kwrvals
	kwrvals = {}
	kwrvals['signal'] = signal - avg
	
	return kwrvals


def windower(signal=None, Fs=None, length=128, shift=64, fcn=None, fcnArgs={}):
	# to apply fcn to squential windows
	
	# check inputs
	if signal is None:
		raise TypeError, "A signal must be provided."
	if Fs is None:
		raise TypeError, "A sampling frequency must be provided."
	if fcn is None:
		raise TypeError, "A function must be specified."
	
	# number of windows
	size = signal.shape
	N = int((size[-1] - length) / shift) + 1
	
	# time
	T = np.arange(length/2., (N) * shift + (length/2.), shift)/Fs
	
	# window
	win = windowfcn('hamming')(length).reshape((1, length))
	
	# apply fcn to windows
	val = []
	for n in range(N):
		# select part
		f = length + n * shift
		i = f - length
		part = signal[:, i:f].copy()
		
		# normalize
		# part = ss.detrend(part, type='constant')
		# apply window
		# part = part * np.dot(np.ones((size[0], 1)), win)
		
		# apply fcn
		val.append(fcn(part, **fcnArgs))
		
	
	# kwrvals
	kwrvals = {}
	kwrvals['time'] = T
	kwrvals['values'] = np.array(val)
	
	return kwrvals


def emd(data, extrapolation='mirror', nimfs=12, sifting_distance=0.2):
    """
    Perform a Empirical Mode Decomposition on a data set.
    
    This function will return an array of all the Emperical Mode Functions as 
    defined in [1]_, which can be used for further Hilbert Spectral Analysis.
    
    The EMD uses a spline interpolation function to approximate the upper and 
    lower envelopes of the signal, this routine implements a extrapolation
    routine as described in [2]_ as well as the standard spline routine.
    The extrapolation method removes the artifacts introduced by the spline fit
    at the ends of the data set, by making the dataset a continuious circle.
	
	Reproduced from github.com/jaidevd/pyhht
    
    Parameters
    ----------
    data : array_like
            Signal Data
    extrapolation : str, optional
            Sets the extrapolation method for edge effects. 
            Options: None
                     'mirror'
            Default: 'mirror'
    nimfs : int, optional
            Sets the maximum number of IMFs to be found
            Default : 12
    sifiting_distance : float, optional
            Sets the minimum variance between IMF iterations.
            Default : 0.2
    
    Returns
    -------
    IMFs : ndarray
            An array of shape (len(data),N) where N is the number of found IMFs
    
    Notes
    -----
    
    References
    ----------
    .. [1] Huang H. et al. 1998 'The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis.'
    Procedings of the Royal Society 454, 903-995
    
    .. [2] Zhao J., Huang D. 2001 'Mirror extending and circular spline function for empirical mode decomposition method'
    Journal of Zhejiang University (Science) V.2, No.3,P247-252
    
    .. [3] Rato R.T., Ortigueira M.D., Batista A.G 2008 'On the HHT, its problems, and some solutions.' 
    Mechanical Systems and Signal Processing 22 1374-1394
    

    """
    
    #Set up signals array and IMFs array based on type of extrapolation
    # No extrapolation and 'extend' use signals array which is len(data)
    # Mirror extrapolation (Zhao 2001) uses a signal array len(2*data)
    if not(extrapolation):
        base = len(data) 
        signals = np.zeros([base, 2])
        nimfs = range(nimfs) # Max number of IMFs
        IMFs = np.zeros([base, len(nimfs)])    
        ncomp = 0
        residual = data
        signals[:, 0] = data
        #DON'T do spline fitting with periodic bounds
        inter_per = 0 
        
    elif extrapolation == 'mirror':
        #Set up base
        base = len(data) 
        nimfs = range(nimfs) # Max number of IMFs
        IMFs = np.zeros([base, len(nimfs)])    
        ncomp = 0
        residual = data
        #Signals is 2*base
        signals = np.zeros([base*2, 2])
        #Mirror Dataset
        half = base / 2
        odd = base % 2
        signals[0:half, 0] = data[::-1][half:2*half]
        signals[half:base + half, 0] = data
        signals[base + half:base * 2, 0] = data[::-1][0:half + odd]
        # Redfine base as len(signals) for IMFs
        base = len(signals) 
        data_length = len(data) # Data length is used in recovering input data
        #DO spline fitting with periodic bounds
        inter_per = 1 
        
    else:
        raise Exception(
        "Please Specifiy extrapolation keyword as None or 'mirror'")
            
    for j in nimfs:
#       Extract at most nimfs IMFs no more IMFs to be found when Finish is True
        k = 0
        sd = 1.
        finish = False
                    
        while sd > sifting_distance and not(finish):   
            min_env = np.zeros(base)
            max_env = min_env.copy()
            
            min_env = np.logical_and(
                                np.r_[True, signals[1:,0] > signals[:-1,0]],
                                np.r_[signals[:-1,0] > signals[1:,0], True])
            max_env = np.logical_and(
                                np.r_[True, signals[1:,0] < signals[:-1,0]],
                                np.r_[signals[:-1,0] < signals[1:,0], True])
            max_env[0] = max_env[-1] = False
            min_env = min_env.nonzero()[0]
            max_env = max_env.nonzero()[0] 

            #Cubic Spline by default
            order_max = 3
            order_min = 3
            
            if len(min_env) < 2 or len(max_env) < 2:
                #If this IMF has become a straight line
                finish = True
            else:
                if len(min_env) < 4:
                    order_min = 1 #Do linear interpolation if not enough points
                    
                if len(max_env) < 4:
                    order_max = 1 #Do linear interpolation if not enough points
                
#==============================================================================
# Mirror Method requires per flag = 1 No extrapolation requires per flag = 0
# This is set in intial setup at top of function.
#==============================================================================
                t = interpolate.splrep(min_env, signals[min_env,0],
                                       k=order_min, per=inter_per)
                top = interpolate.splev(
                                    np.arange(len(signals[:,0])), t)
                
                b = interpolate.splrep(max_env, signals[max_env,0],
                                       k=order_max, per=inter_per)
                bot = interpolate.splev(
                                    np.arange(len(signals[:,0])), b)
                
            #Calculate the Mean and remove from the data set.
            mean = (top + bot)/2
            signals[:,1] = signals[:,0] - mean
        
            #Calculate the sifting distance which is a measure of 
            #simulartity to previous IMF
            if k > 0:
                sd = (np.sum((np.abs(signals[:,0] - signals[:,1])**2))
                             / (np.sum(signals[:,0]**2)))
            
            #Set new iteration as previous and loop
            signals = signals[:,::-1]
            k += 1

        if finish:
            #If IMF is a straight line we are done here.
            IMFs[:,j]= residual
            ncomp += 1
            break
        
        if not(extrapolation):
            IMFs[:,j] = signals[:,0]
            residual = residual - IMFs[:,j]#For j==0 residual is initially data
            signals[:,0] = residual
            ncomp += 1
                
        elif extrapolation == 'mirror':
            IMFs[:,j] = signals[data_length / 2:data_length 
                                                           + data_length / 2,0]
            residual = residual - IMFs[:,j]#For j==0 residual is initially data
                
            #Mirror case requires IMF subtraction from data range then
            # re-mirroring for each IMF 
            half = data_length / 2
            odd = data_length % 2
            signals[0:half,0] = residual[::-1][half:2*half]
            signals[half:data_length + half,0] = residual
            signals[data_length + half:,0] = residual[::-1][0:half + odd]
            ncomp += 1
                
        else:
            raise Exception(
                "Please Specifiy extrapolation keyword as None or 'mirror'")
            
    return IMFs[:,0:ncomp]


def HilbertSpectrum(IMF, Fs, NF=100):
	# to get the Hilbert Energy Spectrum
	
	# Hilbert Transform
	out = analyticSignal(IMF, axis=0)
	amp = out['amplitude']
	amp = amp**2
	phase = unwrap(out['phase'], axis=0)
	omega = (Fs * np.diff(phase, axis=0)) / (2 * np.pi)
	
	# frequencies
	W = np.linspace(0, Fs/2, NF)
	
	# times
	NT = omega.shape[0]
	T = np.linspace(0, (NT-1)/Fs, NT)
	
	# organize into matrix
	matrix = np.zeros([NF, NT], dtype=float)
	for i in range(IMF.shape[1]):
		for t in range(NT):
			k = 1
			while (k < NF) & (omega[t, i] > W[k]):
				k = k + 1
			
			if k < NF:
				matrix[k, t] = matrix[k, t] + amp[t, i]
	
	return T, W, matrix


def HSPlot(T, W, matrix, NTicks=8):
	# to plot HHT
	
	NT = len(T)
	NF = len(W)
	
	W_ticks = np.array(range(0, NF, NF/NTicks))
	W_labels = ['%2.2f' % W[i] for i in W_ticks]
	W_ticks = NF - W_ticks
	
	T_ticks = range(0, NT, NT/NTicks)
	T_labels = ['%2.2f' % T[i] for i in T_ticks]
	
	plot = pl.imshow(matrix[::-1, :], cmap=pl.cm.jet, interpolation='nearest')
	pl.xlabel('Time')
	pl.ylabel('Frequency')
	pl.colorbar()
	pl.xticks(T_ticks, T_labels)
	pl.yticks(W_ticks, W_labels)
	
	return plot


def spectPlot(X, Y, Z, **kwarg):
	# plot spectrogram
	from matplotlib import cm
	
	xmin = np.amin(X)
	xmax = np.amax(X)
	ymin = np.amin(Y)
	ymax = np.amax(Y)
	
	im = pl.imshow(Z[::-1, :], cm.jet, extent=(xmin, xmax, ymin, ymax), **kwarg)
	pl.colorbar(im, shrink=0.5, aspect=5)
	pl.axis('auto')
	
	
	# from mpl_toolkits.mplot3d import Axes3D
	# from matplotlib import cm
	# from matplotlib.ticker import LinearLocator, FormatStrFormatter
	
	# fig = pl.figure()
	# ax = Axes3D(fig)
	# X, Y = np.meshgrid(X, Y)
	# surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
	# ax.set_zlim((np.min(Z), np.max(Z)))

	# # ax.zaxis.set_major_locator(LinearLocator(10))
	# # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	# fig.colorbar(surf, shrink=0.5, aspect=5)
	
	return im


def intervalThresholding(IMF=None, thr=None):
	# Interval-Thresholding
	
	IMF_n = np.zeros(IMF.shape)
	
	for i in range(IMF.shape[1]):
		# find zero-crossings
		zeros = np.diff(np.sign(IMF[:, i])).nonzero()[0] + 1 # index after crossing
		# apply threshold
		for k in range(1, len(zeros)):
			if np.max(np.abs(IMF[zeros[k-1:k], i])) > thr[i]:
				IMF_n[zeros[k-1:k], i] = IMF[zeros[k-1:k], i]
	tSignal = np.sum(IMF_n, axis=1)
	return tSignal


def EMDiit(Signal=None, niter=20, coef=0.6):
	# EMD-based Iterative Interval-Threshold denoising
	
	# perform EMD on signal
	IMF = emd(Signal, extrapolation='mirror', nimfs=20, sifting_distance=0.01)
	
	# partial reconstruction
	h1 = IMF[:, 0]
	mx = len(h1)
	
	# iterate niter times
	dSignal = np.zeros(Signal.shape)
	for i in range(niter):
		# circular shift
		ind = deque(range(mx))
		ind.rotate(np.random.random_integers(0, mx))
		h1a = h1[list(ind)]
		# reconstruct signal
		rec = h1a + np.sum(IMF[:, 1:], axis=1)
		# perform EMD on reconstructed signal
		IMF_n = emd(rec, extrapolation='mirror', nimfs=20, sifting_distance=0.01)
		# thresholds
		sigma2 = np.var(IMF_n[:, 0], ddof=1)
		aux = np.array(range(IMF_n.shape[1])) + 1
		E = (sigma2 / 0.719) * (2.01**(-aux))
		E[0] = sigma2
		thr = coef * pl.sqrt(E*2*pl.log(IMF_n.shape[0]))
		# IT denoising
		dSignal += intervalThresholding(IMF_n, thr)
	
	# average
	dSignal = dSignal / niter
	
	return dSignal


def noiseMeter(data=None, Fs=None, norm=False):
	# get signal statistics to assess noise
	
	cols = ['MaxAmp', 'StdAmp', 'KurtAmp', 'SkewAmp']
	features = ['MaxAmp', 'StdAmp', 'KurtAmp', 'SkewAmp', 'AvgPwr', 'StdPwr']
	tails = ['-alpha', '-beta']
	cols.extend([f+t for f in features for t in tails])
	table = pandas.DataFrame(index=range(data.shape[1]), columns=cols, dtype='float64')
	
	# band pass filter the data
	# alpha band
	[b, a] = ss.butter(3, [2*8/Fs, 2*13/Fs], 'bandpass')
	dataFa = ss.filtfilt(b, a, data.transpose()).transpose()
	# alpha band
	[b, a] = ss.butter(3, [2*13/Fs, 2*25/Fs], 'bandpass')
	dataFb = ss.filtfilt(b, a, data.transpose()).transpose()
	
	# maximum amplitude / variance
	sigma2 = data.std(ddof=1, axis=0)**2
	table['MaxAmp'] = np.abs(data).max(axis=0) #/sigma2
	
	# standard deviation
	table['StdAmp'] = np.sqrt(sigma2)
	
	# kurtosis
	table['KurtAmp'] = stats.kurtosis(data, bias=False, axis=0)
	
	# skewness
	table['SkewAmp'] = np.abs(stats.skew(data, bias=False, axis=0))
	
	# mean and standard deviation of power
	pwr = dataFa**2
	table['AvgPwr-alpha'] = pwr.mean(axis=0)
	table['StdPwr-alpha'] = pwr.std(ddof=1, axis=0)
	pwr = dataFb**2
	table['AvgPwr-beta'] = pwr.mean(axis=0)
	table['StdPwr-beta'] = pwr.std(ddof=1, axis=0)
	
	# maximum amplitude (bands)
	sigma2Fa = dataFa.std(ddof=1, axis=0)**2
	sigma2Fb = dataFb.std(ddof=1, axis=0)**2
	table['MaxAmp-alpha'] = np.abs(dataFa).max(axis=0) #/ sigma2Fa
	table['MaxAmp-beta'] = np.abs(dataFb).max(axis=0) #/ sigma2Fb
	
	# standard deviation (bands)
	table['StdAmp-alpha'] = np.sqrt(sigma2Fa)
	table['StdAmp-beta'] = np.sqrt(sigma2Fb)
	
	# kurtosis (bands)
	table['KurtAmp-alpha'] = stats.kurtosis(dataFa, bias=False, axis=0)
	table['KurtAmp-beta'] = stats.kurtosis(dataFb, bias=False, axis=0)
	
	# skewness (bands)
	table['SkewAmp-alpha'] = np.abs(stats.skew(dataFa, bias=False, axis=0))
	table['SkewAmp-beta'] = np.abs(stats.skew(dataFb, bias=False, axis=0))
	
	if norm:
		aux = table.as_matrix().copy()
		aux /= aux.max(axis=0)
		nrm = np.sqrt(np.dot(aux, aux.transpose())).diagonal()
		return table, nrm
	else:	
		return table


def ICAFilter(signal=None):
	# EEG filtering based on Independent Component Analysis
	
	# ICA decomposition
	ica = FastICA(whiten=True)
	IC = ica.fit(signal).transform(signal)
	A = ica.get_mixing_matrix() # signal = np.dot(IC, A.T)
	
	# noise metrics
	sigma2 = IC.std(ddof=1, axis=0)**2
	f1 = np.abs(IC).max(axis=0) / sigma2
	f2 = np.abs(stats.skew(IC, bias=False, axis=0))
	f = np.hstack((f1.reshape((len(f1), 1)), f2.reshape((len(f2), 1))))
	fr = f.copy()
	f /= f.max(axis=0)
	norm = np.sqrt(np.dot(f, f.T)).diagonal()
	
	# remove noisy IC
	ind = norm.argmax()
	IC_ = IC.copy()
	IC_[:, ind] = 0
	
	# recompute signal
	signalF = np.dot(IC_, A.T)
	
	return signalF, IC, fr
	



if __name__ == '__main__':
	
	# Example
	Fs = 1024
	duration = 1
	t = np.arange(0, duration, 1./Fs)
	signal = pl.cos(2*np.pi*50*t)
	
	asig = analyticSignal(signal)
	
	# fig1 = pl.figure(1)
	# ax1 = fig1.add_subplot(211)
	# ax1.plot(t, signal, t, asig['amplitude'])
	# ax2 = fig1.add_subplot(212)
	# ax2.plot(t, np.unwrap(asig['phase']))
	
	signalM = np.array([signal, signal+2])
	out = windower(signalM, Fs, 128, 64, plf, {'pairList':[(0, 1)]})
	
	# fig2 = pl.figure(2)
	# ax3 = fig2.add_subplot(111)
	# ax3.plot(out['time'], out['values'])
	# ax3.legend(('1 vs 2', ''), loc='best')
	
	NFFT = 4096
	out = windower(signalM, Fs, 128, 64, power, {'NFFT':NFFT})
	W = (Fs/2.)*np.linspace(0, 1, NFFT/2)
	
	# pl.show()
	

