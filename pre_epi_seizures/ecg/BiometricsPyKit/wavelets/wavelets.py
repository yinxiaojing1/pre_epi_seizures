"""
.. module:: wavelets
   :platform: Unix, Windows
   :synopsis: This module provides various functions to ...

.. moduleauthor:: Carlos Carreiras


"""

# imports
import numpy as np
import pylab as pl
import scipy.signal as ss
#import scipy.stats as st
from itertools import cycle
import pywt


def conv(x, h):
    """
    
    Convolution operator. To avoid boundary effects, the input signal is padded (length of the filter) at both ends.
    
    Kwargs:
        x (array): The input signal.
        
        h (list): The FIR filter.
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1]
       
    """
    
    nb = 2 * len(h)
    
    a = x[:nb][::-1]
    b = x[-nb:][::-1]
    
    # s = np.r_[x[0] * np.ones(nb), x, x[-1] * np.ones(nb)]
    s = np.r_[a, x, b]
    y = np.convolve(s, h, mode='same')
    
    # remove mean
    y = y - y.mean()
    
    return y[nb:-nb]


def trous(h):
    """
    
    Add zeros (holes - trous) between the filter coefficients.
    
    Kwargs:
        h (list): The FIR filter.
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1]
       
    """
    
    new = (2 * len(h)) * [0]
    new[::2] = h

    return new


def reverse(x):
    """
    
    Method to reverse a signal.
    y[n] = x[-n]
    
    Kwargs:
        x (array): The input signal.
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1]
       
    """
    
    return x[::-1]


def maxLevel(signal, wavelet):
    """
    
    Determine the maximum decomposition level that is possible to apply the
    RDWT to a given signal, with the specified mother wavelet.
    
    Kwargs:
        signal (int, array): The input signal, or the length of the signal to decompose.
        
        wavelet (string, pywt.Wavelet): The mother wavelet.
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1] Fowler, The Redundant Discrete Wavelet Transform and Additive Noise, 2005
       
    """
    # get the filters from the wavelet
    if isinstance(wavelet, basestring):
        wavelet = pywt.Wavelet(wavelet)
    
    fl = float(max([len(wavelet.dec_lo), len(wavelet.rec_lo), len(wavelet.dec_hi), len(wavelet.rec_hi)]))
    
    # signal length
    if not isinstance(signal, int):
        signal = len(signal)
    
    return int(np.log2(signal / fl) + 1)


def RDWT(signal, wavelet, level, matrix=True, **kwargs):
    """
    
    Perform the Redundant Discrete Wavelet Transform (RDWT). This transform is different from the
    Discrete Wavelet Transform (DWT) by skipping the decimation step at each decomposition level.
    One advantage of RDWT over DWT is that it is time-invariant.
    This transform is also known as:
        Stationary wavelet transform,
        Algorithme a trous,
        Quasi-continuous wavelet transform,
        Translation invariant wavelet transform,
        Shift invariant wavelet transform,
        Undecimated wavelet transform.
    
    Kwargs:
        signal (array): The input signal.
        
        wavelet (string, pywt.Wavelet): The mother wavelet.
        
        level (int): The number of decomposition levels.
        
        matrix (bool): Flag to return the output in matrix form. Default: True.
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        The approximation and detail coefficients are arranged by level: [(cAn, cDn), ..., (cA1, cD1)].
        In matrix form the output is [cD1, cD2, ..., cDn, cAn].
    
    Example:
        signal = ones(1000)
        wavelet = pywt.Wavelet('rbio3.3')
        coefficients, H, G = RDWT(signal, wavelet, 3)
    
    References:
        .. [1] Fowler, The Redundant Discrete Wavelet Transform and Additive Noise, 2005
       
    """
    # Redundant Discrete Wavelet Transform
    
    coefficients = range(level+1)
    coefficients[-1] = (signal, None)
    
    # get the filters from the wavelet
    if isinstance(wavelet, basestring):
        wavelet = pywt.Wavelet(wavelet)
    
    h_dec = list(wavelet.dec_lo)
    h_rec = list(wavelet.rec_lo)
    g_dec = list(wavelet.dec_hi)
    g_rec = list(wavelet.rec_hi)
    
    # to store inverse filters
    H = range(level)
    G = range(level)
    
    for j in xrange(level):
        prev = coefficients[-j-1][0]
        # convolve
        cA = conv(prev, reverse(h_dec))
        cD = conv(prev, reverse(g_dec))
        coefficients[-j-2] = (cA, cD)
        
        # strore filters
        H[-j-1] = h_rec
        G[-j-1] = g_rec
        
        # update filters
        h_dec = trous(h_dec)
        g_dec = trous(g_dec)
        h_rec = trous(h_rec)
        g_rec = trous(g_rec)
    
    if matrix:
        return coeffs2Matrix(coefficients[:-1])
    else:
        return coefficients[:-1], H, G


def iRDWT(coefficients, wavelet, zeros=False):
    """
    
    The inverse Redundant Discrete Wavelet Transform.
    
    Kwargs:
        coefficients (array): The coefficients as produced by the RDWT function (matrix form).
        
        wavelet (string, pywt.Wavelet): The mother wavelet.
        
        zeros (bool): Flag to replace approximation coefficients with zeros.
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        The approximation and detail coefficients are arranged by level: [(cAn, cDn), ..., (cA1, cD1)].
    
    Example:
        signal = ones(1000)
        wavelet = pywt.Wavelet('rbio3.3')
        coefficients, H, G = RDWT(signal, wavelet, 3)
        back = iRDWT(coefficients, H, G)
    
    References:
        .. [1]
       
    """
    
    # get the filters from the wavelet
    if isinstance(wavelet, basestring):
        wavelet = pywt.Wavelet(wavelet)
    
    h_rec = list(wavelet.rec_lo)[::-1]
    g_rec = list(wavelet.rec_hi)[::-1]
    
    level = coefficients.shape[1] - 1
    
    H = range(level)
    G = range(level)
    for j in xrange(level):
        # store filters
        H[-j-1] = h_rec
        G[-j-1] = g_rec
        # update filters
        h_rec = trous(h_rec)
        g_rec = trous(g_rec)
    
    if zeros:
        signal = np.zeros(coefficients.shape[0])
    else:
        signal = coefficients[:, -1].copy()
    
    for j in xrange(1, level+1):
        a = conv(signal, H[j-1])
        b = conv(coefficients[:, -j-1], G[j-1])
        signal = 0.5 * (a + b)
    
    return signal


def coeffs2Matrix(coefficients):
    """
    
    Convert the coefficients list to a matrix format. Note that only the approximation coefficients
    from the last level are kept.
    
    Kwargs:
        coefficents (list): The coefficients from RDWT.
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1]
       
    """
    # convert the coefficients to matrix form
    
    level = len(coefficients)
    
    matrix = [coefficients[-lv][1] for lv in xrange(1, level+1)]
    matrix.append(coefficients[0][0])
    matrix = np.array(matrix).T # to give time in first axis
    
    return matrix


def universalThr(coefficients):
    """
    
    Compute the Universal Threshold for each level of the RDWT.
    
    Kwargs:
        coefficents (array): The coefficients from RDWT in matrix form.
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1] Donoho, De-Noising by Soft Thresholding, 1995
       
    """
    
    N = coefficients.shape[0]
    
    thrV = []
    for i in xrange(coefficients.shape[1]-1):
        cD = coefficients[:, i]
        # noise
        nSigma = np.median(np.abs(cD - np.median(cD))) / 0.6745
        
        thr = nSigma * np.sqrt(2 * np.log(N))
        
        thrV.append(thr)
    
    return thrV


def muniversalThr(coefficients):
    """
    
    Compute the modifief Universal Threshold for each level of the RDWT.
    
    Kwargs:
        coefficents (array): The coefficients from RDWT in matrix form.
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1] Donoho, De-Noising by Soft Thresholding, 1995
       
    """
    
    thrV = []
    for i in xrange(coefficients.shape[1]-1):
        cD = coefficients[:, i]
        thr = np.median(np.abs(cD - np.median(cD))) / 0.6745
        thrV.append(thr)
    
    return thrV


def mchangThr(coefficients):
    """
    
    Compute the modified Chang Threshold for each level of the RDWT. Noise is estimated for each level.
    
    Kwargs:
        coefficents (array): The coefficients from RDWT in matrix form.
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1] Chang, Adaptive Wavelet Thresholding for Image Denoising and Compression, 2000
       
    """
    
    thrV = []
    for i in xrange(coefficients.shape[1]-1):
        cD = coefficients[:, i]
        # noise
        nSigma = np.median(np.abs(cD - np.median(cD))) / 0.6745
        # coefficients
        cSigma = np.std(cD, ddof=1)
        # signal
        sSigma = np.sqrt(np.max([0, (cSigma**2) - (nSigma**2)]))
        
        if sSigma == 0.:
            thr = cD.max()
        else:
            thr = (nSigma**2) / sSigma
        
        thrV.append(thr)
    
    return thrV


def changThr(coefficients):
    """
    
    Compute the Chang Threshold for each level of the RDWT. Noise is estimated from the first level.
    
    Kwargs:
        coefficents (array): The coefficients from RDWT in matrix form.
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1] Chang, Adaptive Wavelet Thresholding for Image Denoising and Compression, 2000
       
    """
    
    thrV = []
    cD = coefficients[:, 0]
    # noise
    nSigma = np.median(np.abs(cD - np.median(cD))) / 0.6745
    
    for i in xrange(coefficients.shape[1]-1):
        cD = coefficients[:, i]
        
        # coefficients
        cSigma = np.std(cD, ddof=1)
        # signal
        sSigma = np.sqrt(np.max([0, (cSigma**2) - (nSigma**2)]))
        
        if sSigma == 0.:
            thr = cD.max()
        else:
            thr = (nSigma**2) / sSigma
        
        thrV.append(thr)
    
    return thrV


def applyThreshold(coefficients, threshold, hard=False, last=False):
    """
    
    Function to apply a threshold to each of the decomposition levels.
    
    Kwargs:
        coefficients (array): List of the wavelet decomposition coefficients in matrix form.
        
        threshold (list): List of length L with the threshold to apply to each level.
        
        hard (bool): Flag to perform hard thresholding. Default: False.
        
        last (bool): Flag to also apply the threshold to the last level. Default: False.
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1]
        
    """
    
    # copy the coefficients
    new = coefficients.copy()
    
    if last:
        stop = coefficients.shape[1] - 1
    else:
        stop = coefficients.shape[1] - 2
    
    for i in xrange(stop):
        cD = new[:, i]
        if hard:
            cD = pywt.thresholding.hard(cD, threshold[i])
        else:
            cD = pywt.thresholding.soft(cD, threshold[i])
        new[:, i] = cD
    
    return new


def cutLevel(coefficients, level):
    """
    
    Set detail coefficients to zero up to (excluding) a given level.
    
    Kwargs:
        coefficients (array): List of the wavelet decomposition coefficients in matrix form.
        
        level (int): The level at which to cut.
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1]
        
    """
    
    # copy the coefficients
    new = coefficients.copy()
    
    new[:, :level] = 0
    
    return new


def meanWave(data, axis=2):
    # compute the mean wave
    
    return data.mean(axis=axis)


def medianWave(data, axis=2):
    # compute the median wave
    
    return np.median(data, axis=axis)


def waveDist(coefficients1, coefficients2, thr=400., levels=None, indexes=None):
    """
    
    Compute distance between a pair of wavelet coefficients (in matrix form).
    
    Kwargs:
        coefficients1 (array): First set of coefficients.
        
        coefficients2 (array): Second set of coefficients.
        
        levels (list, slice): Levels to use in the distance (e.g [0, 1, 2]). Default: all levels.
        
        indexes (list, slice): Time indexes to use in the distance. Default: all.
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1] Chan, Wavelet disntance measure for person identification using ECG, 2008
       
    """
    
    # dimensions (levels, nb of coefficients)
    nb, L = coefficients1.shape
    
    # normalization factor
    aux = np.empty(shape=(nb, L, 2))
    aux[:, :, 0] = np.abs(coefficients1)
    aux[:, :, 1] = np.abs(coefficients2)
    nrm = np.max(aux, axis=2)
    nrm[nrm < thr] = thr
    
    # absolute difference weighted by the normalization
    dist = np.abs(coefficients1 - coefficients2) / nrm
    # dist = np.abs(coefficients1 - coefficients2)
    
    # sums
    if indexes is None:
        # indexes = xrange(nb)
        indexes = xrange(0, 600)
    dist = dist[indexes, :]
    
    if levels is None:
        levels = xrange(L)
        # levels = xrange(4, 7)
    dist = dist[:, levels]
    
    m = np.prod(dist.shape)
    
    dist = np.sum(np.sum(dist)) / m
    
    return dist


def rcosdmean(data=None, th_min=1.5, th_max=1.5, th_dist=0.5):
    """
    
    Outlier detection based on the mean segment and the medians of maxima and minima (wavelet version).
    
    Kwargs:
        data (array): The segments of wavelet coefficients.
        
        th_min (float): Parameter for median of minima. Default: 1.5
        
        th_max (float): Parameter for median of maxima. Default: 1.5
        
        th_dist (float): Parameter for distances. Default: 0.5
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1] 
       
    """
    
    # check inputs
    if data is None:
        raise TypeError, "No data provided."
    
    # results
    good = []
    maybe = []
    bad = []
    dists = np.zeros(data.shape[0], dtype='float')
    
    # minima
    data_min = data.min(axis=1)
    min_med = np.median(data_min, axis=0)
    # maxima
    data_max = data.max(axis=1)
    max_med = np.median(data_max, axis=0)
    
    # distances to mean
    mean_seg = data.mean(axis=0)
    
    # first check and compute distance to mean
    k = data.shape[2] / 2
    for i in xrange(len(data)):
        # check
        cm = (data_min[i, :] < th_min * min_med) & (data_max[i, :] > th_max * max_med)
        if np.sum(cm) > k: # majority voting
            # bad segment
            bad.append(i)
        else:
            # save for second check
            maybe.append(i)
        
        # distance
        dists[i] = waveDist(data[i, :, :], mean_seg)
    
    # compute mean and std distance
    dist_mean = dists.mean()
    dist_std = dists.std(ddof=1)
    
    # second check
    for i in maybe:
        if dists[i] > dist_mean + th_dist * dist_std:
            # bad segment
            bad.append(i)
        else:
            # good segment
            good.append(i)
    
    good.sort()
    good = np.array(good)
    bad.sort()
    bad = np.array(bad)
    
    return {'0': good, '-1': bad}
    


def RPeaks(matrix, sampleRate):
    """
    
    Determine locations of R peaks in the wavelet domain.
    
    Kwargs:
        matrix (array): Set of coefficients.
        
        sampleRate (float): The sample rate.
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1] Rooijakkers, Low-complexity R-peak Detection in ECG Signals, 2011
       
    """
    
    # output
    R = []
    
    # apply universal threshold
    thrV = 0.6 * np.array(universalThr(matrix))
    # used = 'modified'
#    # check if too much is being cut
#    nb = np.sum(np.abs(matrix[:, 4]) > thrV[4])
#    if nb < (2 * matrix.shape[0] / float(sampleRate)):
#        # use modified threshold
#        thrV = muniversalThr(matrix)
#        used = 'modified'
    
    # apply the threshold
    coeffs = applyThreshold(matrix, thrV)
    
    # use the 5th level
    scale = coeffs[:, 4]
    S = np.abs(scale)
    
    # statistics
#    mx = scale.max()
    mxS = S.max()
#    mean = scale.mean()
    meanS = S.mean()
#    std = scale.std(ddof=1)
    stdS = S.std(ddof=1)
#    kurt = st.kurtosis(scale, fisher=False, bias=False)
#    kurtS = st.kurtosis(S, fisher=False, bias=False)
#    skew = st.skew(scale, bias=False)
#    skewS = st.skew(S, bias=False)
    
    # parameters
    N = len(scale)
    loRR = int((60. / 200.) * sampleRate) # 200 bpm
#    hiRR = int((60. / 20.) * sampleRate) # 20 bpm
    avgRR = int(sampleRate) # 60 bpm
#    alpha = 1. / 3.
    v1 = int(0.05 * sampleRate) # 50 ms
#    v2 = int(0.25 * sampleRate) # 250 ms
    v3 = int(0.02 * sampleRate) # 20 ms
    v4 = int(0.125 * sampleRate) # 125 ms
    
    thrI = mxS / 2
    minThr = 0.2 * stdS
    noiseThrUp = meanS + 4. * stdS
    
    # cycle
    start = 0
    stop = loRR
    thr = thrI
    thrP = thr
    while stop < N:
        window = S[start:stop]
        candy = window > thr
        
        if np.all(candy == False):
            # no peak found (increase segment, reduce the threshold)
            thr = 1 * thr / 3
            if thr < minThr:
                # this area is sterile
                start = stop
                stop = start + avgRR
                thr = thrI
            else:
                stop = stop + loRR
            
            continue
                
        # get the max within loRR
        ind = np.argmax(candy)
        subwindow = S[start+ind:start+ind+loRR]
        sind = np.argmax(subwindow)
        r = sind + start + ind
        
        # check sign
        if scale[r] < 0:
            # it's a negative minimum, search for the next positive maximum
            a = r + v3
            b = r + v1
            miniwindow = scale[a:b]
            try:
                sind = np.argmax(miniwindow)
            except ValueError:
                # not enough data
                break
            if np.abs(miniwindow[sind]) < (5 * thr / 6):
                # this is not a R
                start = r + v3
                stop = start + loRR
                continue
            
            r = a + sind
#        else:
#            # it's a positive maximum, search for the previous negative minimum
#            a = r - v3
#            if a < 0:
#                a = 0
#            b = r - v2
#            if b < 0:
#                b = 0
#            miniwindow = scale[b:a]
#            try:
#                sind = np.argmin(miniwindow)
#            except ValueError:
#                # not enough data
#                start = r + v3
#                stop = start + loRR
#                continue
#            if np.abs(miniwindow[sind]) < (5 * thr / 6):
#                # this is not a R
#                start = r + v3
#                stop = start + loRR
#                continue
        
        # check local noise
        a = r - v4
        if a < 0:
            a = 0
        b = r + v4
        miniwindow = S[a:b]
        locNoise = miniwindow.mean() + miniwindow.std(ddof=1)
        if (locNoise > noiseThrUp):
            # disregard segment
            start = r + 4 * v4
            stop = start + loRR
            continue
        
        
        R.append(r)
        
        # update threshold
        # a1 = np.max([0, r-v1])
        # a2 = np.max([0, r-v2])
        # if a1 == a2:
            # Imax = np.max(S[r+v1:r+v2])
        # else:
            # Imax = np.max(np.hstack((S[a2:a1], S[r+v1:r+v2])))
        
        # snr = np.log2(S[r]) - np.log2(Imax)
        # noise = 0.5 * (noise + np.median([1./3, (5 - snr)/6, 1.]))
        # thr_e = noise * np.max(window)
        # thr = alpha * thr_e + (1 - alpha) * thr
        
        
        aux = thr
        thr = np.mean([S[r]/2., thr, thrP])
        thrP = aux
        
        # next window
        if len(R) > 2:
            avgRR = int(np.mean(np.diff(R)))
        start = r + loRR
        stop = start + avgRR + loRR
    
    
    return R


def extractSegments(matrix, R, sampleRate):
    """
    
    Extract the wavelet segments.
    
    Kwargs:
        matrix (array): Set of coefficients.
        
        R (list): List of indexes locating the R peaks.
        
        sampleRate (float): The sample rate.
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1]
       
    """
    
    length = matrix.shape[0]
    
    segs = []
    for r in R:
        a = r - int(0.2 * sampleRate) # 200 ms before R
        if a < 0:
            continue
        b = r + int(0.4 * sampleRate) # 400 ms after R
        if b > length:
            continue
        segs.append(matrix[a:b, :])
    segs = np.array(segs)
    
    return segs


def waveletSegments(matrix, sampleRate, **kwargs):
    """
    
    Determine R peak locations and extract the wavelet segments.
    
    Kwargs:
        matrix (array): Set of coefficients.
        
        sampleRate (float): The sample rate.
    
    Kwrvals:
        segments (array): The segmented wavelet coefficients (time, levels, segments).
        
        R (list): List of indexes locating the R peaks.
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1]
       
    """
    
    # R peaks
    R = RPeaks(matrix, sampleRate)
    
    # segments
    segs = extractSegments(matrix, R, sampleRate)
    
    return {'segments': segs, 'R': R}


def plotWaveletSpectrum(wavelet, level, sampleRate, npoints=1024, start=None, stop=None):
    """
    
    Plot the transfer function (amplitude) of the wavelet decomposition filters.
    
    Kwargs:
        wavelet (string, pywt.Wavelet): The mother wavelet.
        
        level (int): The number of decomposition levels.
        
        sampleRate (float): The sample rate.
        
        npoints (int): Number of points in which to compute the transfer function. Default: 1024.
        
        start (int): Starting frequency to plot. Default: None.
        
        stop (int): Ending frequency to plot. Default: None.
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1]
       
    """
    
    # get the filters from the wavelet
    if isinstance(wavelet, basestring):
        wavelet = pywt.Wavelet(wavelet)
    
    # get the filter coefficients
    h_dec = list(wavelet.dec_lo)
    g_dec = list(wavelet.dec_hi)
    
    transFunc = np.zeros((npoints, level+1), dtype='complex128')
    labels = range(level+1)
    prev = np.ones(npoints, dtype='complex128')
#    w, th = ss.freqz(reverse(h_dec), worN=npoints)
#    w, tg = ss.freqz(reverse(g_dec), worN=npoints)
#    transFunc[:, 0] = tg
#    prev = th
    for i in xrange(level):
        # labels
        labels[i] = 'Level %d' % (i+1)
        
        # compute individual transfer functions
        w, th = ss.freqz(reverse(h_dec), worN=npoints)
        w, tg = ss.freqz(reverse(g_dec), worN=npoints)
        # details
        transFunc[:, i] = prev * tg
        # approximations
        prev = prev * th
        # update filters
        h_dec = trous(h_dec)
        g_dec = trous(g_dec)
    
    # plug the spectrum
    labels[-1] = 'Cork'
    transFunc[:, -1] = prev
    
    # normalize
    transFunc = np.abs(transFunc)
    transFunc /= transFunc.max(axis=0)
    if start is not None:
        start = int(npoints * 2 * start / sampleRate)
    if stop is not None:
        stop = int(npoints * 2 * stop / sampleRate)
    sel = slice(start, stop)
    
    # plot
    font = {'family': 'Bitstream Vera Sans',
            'weight': 'normal',
            'size': 20}
    pl.rc('font', **font)
    lines = ["-.", ":", "-", "--"]
    linecycler = cycle(lines)
    
    fig = pl.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)
    w = np.linspace(0, sampleRate/2, npoints)
    for i in xrange(level+1):
        ax.semilogx(w[sel], transFunc[sel, i], next(linecycler), linewidth=4)
    # ax.axhline(linewidth=3, color='k')
    # ax.axvline(linewidth=3, color='k')
    # ax.tick_params(axis='both', which='major', labelsize=16)
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude (normalized)')
    ax.legend(labels, loc='center left')
    ax.grid(which='both', axis='x')
    
    return fig


def plotCoeffs(signal, coefficients, start=None, stop=None, step=None):
    """
    
    Plot the decomposition.
    
    Kwargs:
        signal (array): The original signal.
        
        coefficients (array): Set of coefficients in matrix form.
        
        start (int): Starting time index to plot. Default: None.
        
        stop (int): Ending time index to plot. Default: None.
        
        step (int): The time step to plot. Default: None.
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1]
       
    """
    # plot the decomposition
    
    sel = slice(start, stop, step)
    level = coefficients.shape[1]
    
    fig = pl.figure(figsize=(16, 9))
    ax = fig.add_subplot(level+1, 1, 1)
    ax.plot(signal[sel])
    ax.set_ylabel('Original Signal')
    pl.setp(ax.get_yticklabels(), visible=False)
    
    # details
    for k in xrange(level-1):
        ax = fig.add_subplot(level+1, 1, k+2)
        ax.plot(coefficients[sel, k])
        ax.set_ylabel('Level %d' % (k+1))
        pl.setp(ax.get_yticklabels(), visible=False)
    
    # approximation
    ax = fig.add_subplot(level+1, 1, level+1)
    ax.plot(coefficients[sel, -1])
    ax.set_ylabel('Approximation')
    pl.setp(ax.get_yticklabels(), visible=False)
    
    return fig


def plotOutliers(data=None, outliersResult=None, offset=0):
    """
    
    Plot the results of the outlier removal procedure.
    
    Kwargs:
        data (array): The segments of wavelet coefficients.
        
        outliersResult (dict): Result of the outlier removal procedure;
                               has keys '0' (good) and '-1' (bad).
        
        offset (int): Offset to correct the display of the levels. Default: 0
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1]
       
    """
    
    # check inputs
    if data is None:
        raise TypeError, "No data provided."
    if outliersResult is None:
        raise TypeError, "No outliers provided."
    
    nb = data.shape[2]
    good = outliersResult['0']
    bad = outliersResult['-1']
    acc = float(len(good)) / (len(good) + len(bad))
    rej = 1. - acc
    
    fig = pl.figure(figsize=(12, 9))
    
    for i in xrange(nb):
        ax1 = fig.add_subplot(nb, 2, 2*(i+1)-1)
        ax1.plot(data[good, :, i].T, 'b', alpha=0.7)
        ax1.set_ylabel('Level %d' % (i + offset))
        
        ax2 = fig.add_subplot(nb, 2, 2*(i+1))
        ax2.plot(data[bad, :, i].T, 'r', alpha=0.7)
        
        ax1.grid()
        ax2.grid()
    
    ax1.set_xlabel('Retained: %2.2f %% (%d)' % (100*acc, len(good)))
    ax2.set_xlabel('Removed: %2.2f %% (%d)' % (100*rej, len(bad)))
    
    return fig



if __name__ == '__main__':
    # mother wavelet
    wave = 'rbio3.3'
    
    # parameters
    level = 10
    
    # get RDWT
    Fs = 1000.
    time = np.arange(0, 10, 1/Fs)
    signal = np.cos(2*pl.pi*10*time) + np.cos(2*pl.pi*5*time) + 0.1 * np.random.random(len(time))
    signal = signal - signal.mean()
    
    level = maxLevel(signal, wave)
    
    coeffs = RDWT(signal, wave, level)
    # coeffs, H, G = RDWT(signal, wave, level, matrix=False)
    
    # inverse RDWT
    back = iRDWT(coeffs, wave, zeros=True)
    
    fig = plotCoeffs(signal, coeffs)
    pl.show()
    
    pl.subplot(111)
    pl.plot(time, signal, label='original')
    pl.plot(time, np.roll(back, -5), label='reconstructed')
    pl.legend()
    pl.show()
    
