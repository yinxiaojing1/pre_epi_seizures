'''
Created on 15 de Out de 2012

@author: Carlos
'''

# imports
import numpy as np
import pylab as pl
from scipy.stats import stats
import pandas
import itertools
import pywt
import json
from database.biomesh import dataContainer
from datamanager import datamanager as dm
import wavelets
import scipy.ndimage.filters as sf



def saveWaves(config, inputs):
    # save the wavelet coefficients
    
    # mother wavelets
    waves = [pywt.Wavelet('coif5'), pywt.Wavelet('rbio3.3'), pywt.Wavelet('rbio3.5'),
             pywt.Wavelet('rbio3.9'), pywt.Wavelet('rbio5.5'), pywt.Wavelet('db3'),
             pywt.Wavelet('db8')]
    
    # decomposition level
    level = 10
    
    # update the config
    for wave in waves:
        waveName = wave.name
        for var in inputs:
            name = var + '+' + waveName
            ty = config['mapper'][var].split('/')
            ty[-1] = 'Wavelets'
            ty.append(waveName.replace('.', '-'))
            ty = '/'.join(ty)
            config['mapper'][name] = ty
    
    # cycle the subjects
    for subject in dm.Store(config):
        print subject
        
        # cycle the records
        for record in subject:
            print record
            
            # cycle the inputs
            for var in inputs:
                ty = config['mapper'][var]
                print var
                try:
                    data = record[var]
                except KeyError:
                    print "Skipped", var
                    continue
                
                # cycle the datasets
                for item in data:
                    print item.metadata['name']
                    weg = ty + '/' + item.metadata['name']
                    
                    # cycle the wavelets
                    for wave in waves:
                        waveName = wave.name
                        print waveName
                        
                        # compute the RDWT
                        coeffs = wavelets.RDWT(item.signal, wave, level)[0]
                        
                        # save
                        name = var + '+' + waveName
                        print name
                        metadata = {'units': item.metadata['units'], 'sampleRate': item.metadata['sampleRate'],
                                    'wavelet': {'name': waveName, 'level': level}, 'source': weg, 'labels': [waveName]}
                        record[name] = dataContainer('signals', signal=coeffs, mdata=metadata)
    
    return config


def segWaves(config, inputs):
    # extract the wavelet segments
    
    # update the config
    for var in inputs:
        # to save the R positions
        name = var + '+R'
        ty = config['mapper'][var].replace('signals', 'events') + '/R'
        config['mapper'][name] = ty
        # to save the segments
        name = var + '+Segs'
        ty = config['mapper'][var] + '/Segments'
        config['mapper'][name] = ty
    
    # cycle the subjects
    for subject in dm.Store(config):
        print subject
        
        # cycle the records
        for record in subject:
            print record
            
            # cycle the inputs
            for var in inputs:
                print var
                
                try:
                    data = record[var]
                except KeyError:
                    print "Skipped", var
                    continue
                
                ty = config['mapper'][var]
                
                # cycle the datasets
                for item in data:
                    print item.metadata['name']
                    
                    # convert wavelet coefficients to matrix
                    matrix = wavelets.coeffs2Matrix(item.signal)
                    
                    # extract segments
                    R = wavelets.RPeaks(item.signal, item.metadata['sampleRate'])
                    segs = wavelets.extractSegments(matrix, R, item.metadata['sampleRate'])
                    
                    # store
                    weg = ty + '/' + item.metadata['name']
                    
                    # store R locations
                    name = var + '+R'
                    print name
                    
                    metadata = {'source': weg, 'eventSync': 0, 'dictionary': {}}
                    record[name] = dataContainer('events', timeStamps=R, values=[])
                    
                    # store segments
                    name = var + '+Segs'
                    print name
                    
                    metadata = {'units': item.metadata['units'], 'sampleRate': item.metadata['sampleRate'],
                                'source': weg, 'labels': ['Segments']}
                    record[name] = dataContainer('signals', signal=segs, mdata=metadata)
    
    return config


def corrDist(seg1, seg2):
    
    # correlation at lag 0
    corr = np.correlate(seg1, seg2, mode='valid')[0]
    
    # energies
    E1 = np.sum(np.array(seg1) ** 2)
    if E1 == 0:
        E1 = 1
    E2 = np.sum(np.array(seg2) ** 2)
    if E2 == 0:
        E2 = 1
    
    return 1 - (corr / (np.sqrt(E1 * E2)))


def SIFT(signal, sigma, sampleRate):
    # try 1D-SIFT
    
    # make sure signal is numpy
    signal = np.array(signal)
    
    # compute scale-space
    N = len(signal)
    k = 2 ** (1./4)
    sigma = float(sigma)
    # nb = int(np.log2(N))
    nb = int(np.log(N / sigma) / np.log(k))
    S = np.zeros((N, nb))
    for i in xrange(nb):
        # print sigma * (k ** i)
        S[:, i] = sf.gaussian_filter1d(signal, sigma * (k ** i))
    
    # difference of gaussians
    D = np.zeros((N, nb-1))
    for i in xrange(1, nb):
        D[:, i-1] = S[:, i] - S[:, i-1]
    
    # extrema detection
    D_abs = np.abs(D)
    E = []
    for i in xrange(1, N-1):
        for j in xrange(1, nb-2):
            aux = D_abs[i-1:i+2, j-1:j+2].max()
            if D_abs[i, j] >= aux:
                E.append([i, j])
    E = np.array(E)
    kPoints = set(E[:, 0])
    
    # descriptors
    descriptor = []
    newE = []
    M = 8 # number of segments
    N = 20 # number of extrema
    
    
    for keyP in kPoints:
        # select N closest key points
        neighbors = np.array(list(kPoints - set([keyP])))
        aux = np.argsort(np.abs(neighbors - keyP))[:N]
        neighbors = neighbors[aux]
        
#        # find extrema
#        ext, extType = findExtrema(scale)
#        
#        if keyP in ext:
#            ind = pl.find(ext == keyP)[0]
#            extrema = np.delete(ext, ind)
#            d2x = np.delete(extType, ind)
#        else:
#            extrema = ext
#            d2x = extType
#        
#        aux = np.argsort(np.abs(keyP - extrema))[:N]
#        extrema = extrema[aux]
#        d2x = d2x[aux]
        
        # loc_desc = buildDescriptor(signal, sampleRate, M, extrema, d2x)
        # loc_desc = buildDescriptor(scale, sampleRate, M, extrema, d2x)
        # loc_desc = buildDescriptor2(signal, sampleRate, M, keyP)
        loc_desc = buildDescriptor3(signal, sampleRate, keyP, neighbors)
        
        if loc_desc is None:
            print keyP
            continue
        
        descriptor.append(loc_desc)
        newE.append(keyP)
        
    descriptor = np.array(descriptor)
    newE = np.array(newE)
    
    return S, D, newE, descriptor

def buildDescriptor(signal, sampleRate, M, extrema, d2x):
    # build the descriptor for each keypoint
    loc_desc = []
    for neighbor, maxi in itertools.izip(extrema, d2x):
        # get M nearest points on each side of the extrema
        tt = np.array([(i+1) / sampleRate for i in range(M)])
        try:
            left = (signal[neighbor] - signal[neighbor-M:neighbor]) / tt[::-1]
            right = (signal[neighbor+1:neighbor+M+1] - signal[neighbor]) / tt
        except ValueError:
            return None
        if maxi == -1:
            right = np.abs(right)
        loc_desc.extend(right / left)
    
    return loc_desc


def buildDescriptor2(signal, sampleRate, M, kp):
    # build the descriptor for each keypoint
    loc_desc = []
    # get M nearest points on each side of the extrema
    tt = np.array([(i+1) / sampleRate for i in range(M)])
    try:
        left = (signal[kp] - signal[kp-M:kp]) / tt[::-1]
        right = (signal[kp+1:kp+M+1] - signal[kp]) / tt
    except ValueError:
        return None
#    if (left[-1] > 0) and (right[0] < 0):
#        right = np.abs(right)
    
    loc_desc.extend(right / left)
    
    return loc_desc


def buildDescriptor3(signal, sampleRate, keyP, neighbors):
    # build descriptor for each key point
    
    tt = (neighbors - keyP) / sampleRate
    loc_desc = (signal[neighbors] - signal[keyP]) / tt
    
    return loc_desc


def tmpSIFT(s1, s2, sigma, sampleRate):
    # compare
    
    out1 = SIFT(s1, 5., sampleRate)
    kp1 = out1[2]
    ds1 = out1[3]
    
    out2 = SIFT(s2, 5., sampleRate)
    kp2 = out2[2]
    ds2 = out2[3]
    s2 = s2 + 50 + s1.max() - s2.min()
    
    matrix = np.zeros((len(kp1), len(kp2)))
    for i in xrange(len(kp1)):
        for j in xrange(len(kp2)):
            matrix[i, j] = np.sqrt(np.sum((ds1[i]-ds2[j])**2))
    
    m1 = np.argmin(matrix, axis=1)
    m2 = np.argmin(matrix, axis=0)
    
    fig = pl.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    
    ax1.plot(s1)
    ax1.plot(s2)
    ax1.plot(kp1, s1[kp1], 'ro', kp2, s2[kp2], 'ro')
    for i in xrange(len(m1)):
        ax1.plot([kp1[i], kp2[m1[i]]], [s1[kp1[i]], s2[kp2[m1[i]]]])
    
    ax2.plot(s1)
    ax2.plot(s2)
    ax2.plot(kp1, s1[kp1], 'ro', kp2, s2[kp2], 'ro')
    for i in xrange(len(m2)):
        ax2.plot([kp1[m2[i]], kp2[i]], [s1[kp1[m2[i]]], s2[kp2[i]]])
    
    return fig


def findExtrema(x, strict = False, withend = False):
    """
    This function will index the extrema of a given array x.
    
    Options:
        strict        If true, will not index changes to zero gradient
        withend    If true, always include x[0] and x[-1]
    
    This function will return a tuple of extrema indexes and extrema type
    """
    
    # This is the gradient
    dx = np.zeros(len(x))
    dx[1:] = np.diff(x)
    dx[0] = dx[1]
    
    # Clean up the gradient in order to pick out any change of sign
    dx = np.sign(dx)
    
    # define the threshold for whether to pick out changes to zero gradient
    threshold = 0
    if strict:
        threshold = 1
        
    # Second order diff to pick out the spikes
    d2x = np.diff(dx)
    
    # find maxima and minima
    old = d2x.copy()
    d2x = np.abs(d2x)
    
    # Take care of the two ends
    if withend:
        d2x[0] = 2
        d2x[-1] = 2
    
    # Sift out the list of extremas
    ind = np.nonzero(d2x > threshold)[0]
    extremaType = np.sign(old[ind])
    
    return ind, extremaType


def noiseMeter(data=None):
    # get signal statistics to assess noise
    
    cols = ['Max', 'Std', 'Max/Std', 'Kurt', 'Skew']
    table = pandas.DataFrame(index=range(data.shape[1]), columns=cols, dtype='float64')
    
    # maximum amplitude
    table['Max'] = np.abs(data).max(axis=0) #/sigma2
    
    # standard deviation
    table['Std'] = data.std(ddof=1, axis=0)
    
    # max/std
    table['Max/Std'] = table['Max'] / table['Std']
    
    # kurtosis
    table['Kurt'] = stats.kurtosis(data, bias=False, axis=0)
    
    # skewness
    table['Skew'] = np.abs(stats.skew(data, bias=False, axis=0))
    
    return table


def landmarkSmoothing(landmarks, D, P):
    # minimal distance, percentage principal (MDPP)
    
    # max
    mx = landmarks[:, 1].max() - landmarks[:, 1].min()
    
    # distances on x
    xd = np.diff(landmarks[:, 0]) # out[n] = a[n+1] - a[n]
    candx = pl.find(xd < D)
    
    # distances on y
    # yd = 2. * np.abs(landmarks[candx-1, 1] - landmarks[candx, 1]) / (np.abs(landmarks[candx-1, 1]) + np.abs(landmarks[candx, 1]))
    # yd = np.abs(landmarks[candx-1, 1] - landmarks[candx, 1])
    yd = np.abs(landmarks[candx, 1] - landmarks[candx-1, 1]) / mx
    candy = pl.find(yd < P)
    
    remove = set(candx[candy]) | set(candx[candy] + 1)
    Lset = set(range(len(landmarks)))
    ind = list(Lset - remove)
    ind.sort()
    
    return landmarks[ind, :]


def landmarkReconstruction(landmarks, x):
    # for order 1 landmarks
    
    P = []
    newy = np.zeros(len(x))
    for i in xrange(1, len(landmarks)):
        x1, y1 = landmarks[i-1, :]
        x2, y2 = landmarks[i, :]
        
        X = np.array(
                     [[x1**3, x1**2, x1, 1],
                      [x2**3, x2**2, x2, 1],
                      [3 * x1**2, 2 * x1, 1, 0],
                      [3 * x2**2, 2 * x2, 1, 0]]
                     )
        Y = np.array([y1, y2, 0, 0])
        
        v = np.linalg.solve(X, Y)
        P.append(v)
        
        sel = (x >= x1) & (x <= x2)
        xsel = x[sel]
        y = v[0] * (xsel ** 3) + v[1] * (xsel ** 2) + v[2] * xsel + v[3]
        newy[sel] = y
    
    return P, newy


def landmarkFeatures(landmarks):
    
    f = []
    
    for i in xrange(1, len(landmarks)-1):
        # horizontal distance
        h = landmarks[i, 0] - landmarks[i-1, 0]
        # vertical distance
        v = landmarks[i, 1] - landmarks[i-1, 1]
        # derivative
        vhr = v / h
        # relative vertical distance
        pv = v / landmarks[i, 1]
        # y value
        y = landmarks[i, 1]
        
        f.append([h, v, vhr, pv, y])
    
    return np.array(f)


def findAllExtrema(x):
    # find all extrema (Fink, Important Extrema of Time Series, 2007)
    
    N = len(x)
    output = []
    
    i = 1
    while (i < N-1) and (x[i] == x[0]):
        i += 1
    
    if (i < N-1) and (x[i] < x[0]):
        i = findMin(x, i, output)
    
    while i < N:
        i = findMin(x, i, output)
        i = findMax(x, i, output)
    
    return output

def findMin(x, i, output):
    # find the first minimum after the i-th point (Fink, Important Extrema of Time Series, 2007)
    
    N = len(x)
    left = i
    
    while (i < N-1) and (x[i] >= x[i+1]):
        i += 1
        if x[left] > x[i]:
            left = i
    
    if i < N-1:
        writeOutput(x, left, i, 'min', output)
    
    return i + 1

def findMax(x, i, output):
    # find the first maximum after the i-th point (Fink, Important Extrema of Time Series, 2007)
    
    N = len(x)
    left = i
    
    while (i < N-1) and (x[i] <= x[i+1]):
        i += 1
        if x[left] < x[i]:
            left = i
    
    if i < N-1:
        writeOutput(x, left, i, 'max', output)
    
    return i + 1

def writeOutput(x, left, right, tpe, output):
    # output extrema (Fink, Important Extrema of Time Series, 2007)
    
    if left == right:
        output.append((x[right], right, tpe, 'strict'))
    else:
        output.append((x[left], left, tpe, 'left'))
        print left, right
        for k in range(left+1, right):
            output.append((x[k], k, tpe, 'flat'))
        output((x[right], right, tpe, 'right'))
    
    return None


def importantExtrema(x, R, fdist):
    # find the important extrema (Fink, Important Extrema of Time Series, 2007)
    
    N = len(x)
    output = []
    
    i = findFirstI(x, R, fdist, output)
    
    if (i < N-1) and (x[i] < x[0]):
        i = findMinI(x, i, R, fdist, output)
    
    while i < N-1:
        i = findMaxI(x, i, R, fdist, output)
        i = findMinI(x, i, R, fdist, output)
    
    return output

def findFirstI(x, R, fdist, output):
    # find first important extremum (Fink, Important Extrema of Time Series, 2007)
    
    N = len(x)
    i = 0
    leftMin = 0
    rightMin = 0
    leftMax = 0
    rightMax = 0
    
    while (i < N-1) and (fdist(x[i+1], x[leftMax]) < R) and (fdist(x[i+1], x[leftMin]) < R):
        i += 1
        if x[leftMin] > x[i]: leftMin = i
        if x[rightMin] >= x[i]: rightMin = i
        if x[leftMax] < x[i]: leftMax = i
        if x[rightMax] <= x[i]: rightMax = i
    
    i += 1
    if (i < N-1) and (x[i] > x[0]):
        writeOutput(x, leftMin, rightMin, 'min', output)
    if (i < N-1) and (x[i] < x[0]):
        writeOutput(x, leftMax, rightMax, 'max', output)
    
    return i

def findMinI(x, i, R, fdist, output):
    # find the first important min after i-th point (Fink, Important Extrema of Time Series, 2007)
    
    N = len(x)
    left = i
    right = i
    
    while (i < N-1) and ((x[i+1] < x[left]) or (fdist(x[i+1], x[left]) < R)):
        i += 1
        if x[left] > x[i]: left = i
        if x[right] >= x[i]: right = i
    
    if (left < N-1) and (right < N-1):
        writeOutput(x, left, right, 'min', output)
    
    return i + 1

def findMaxI(x, i, R, fdist, output):
    # find the first important max after i-th point (Fink, Important Extrema of Time Series, 2007)
    
    N = len(x)
    left = i
    right = i
    
    while (i < N-1) and ((x[i+1] > x[left]) or (fdist(x[i+1], x[left]) < R)):
        i += 1
        if x[left] < x[i]: left = i
        if x[right] <= x[i]: right = i
    
    if (left < N-1) and (right < N-1):
        writeOutput(x, left, right, 'max', output)
    
    return i + 1


def correctR(signal, R, sampleRate, left=0, right=0.25, absFlag=True):
    # corret R position to next closest max
    
    if absFlag:
        f = lambda v: np.abs(v)
    else:
        f = lambda v: v
    
    left = int(left * sampleRate)
    right = int(right * sampleRate)
    
    newR = []
    for r in R:
        a = r - left
        if a < 0:
            a = 0
        b = r + right
        miniwindow = f(signal[a:b])
        ind = np.argmax(miniwindow)
        newR.append(a + ind)
    
    return newR


def segment(signal, R, sampleRate):
    # segment a signal
    
    length = len(signal)
    segs = []
    for r in R:
        a = r - int(0.2 * sampleRate) # 200 ms before R
        if a < 0:
            continue
        b = r + int(0.4 * sampleRate) # 400 ms after R
        if b > length:
            continue
        segs.append(signal[a:b])
    segs = np.array(segs)
    
    return segs




if __name__ == '__main__':
    import os
    import cPickle
    import gzip
    from database import biomesh
    
#    path = 'C:\\Users\\Carlos\\testWaveletDist\\T_P_outliers_meanWave'
#    
#    # connect to db
#    db = biomesh.biomesh('CVP', host='193.136.222.234', dstPath='D:/BioMESH', sync=False)
#    
#    records = db.records.listAndTags(['T1', 'Enfermagem', 'Sitting'])['idList']
#    
#    for rec in records:
#        print rec
#        
#        # get wavelet segments
#        data = db.records.getSignal(rec, '/ECG/hand/Wavelets/rbio3-3/Segments/4to6', 0)
#        waveSegs = data['signal']
#        Fs = float(data['mdata']['sampleRate'])
#        
#        # get reconstructed wavelet segments
#        data = db.records.getSignal(rec, '/ECG/hand/Wavelets/rbio3-3/Reconstruction/Segments', 0)
#        segs = data['signal']
#        
#        # outliers
#        meanWave = np.zeros((waveSegs.shape[0], waveSegs.shape[1], 1))
#        meanWave[:, :, 0] = wavelets.meanWave(waveSegs)
#        stdWave = np.zeros(meanWave.shape)
#        stdWave[:, :, 0] = waveSegs.std(axis=2, ddof=1)
#        
#        diff = np.abs(waveSegs - meanWave)
#        lim = diff < stdWave
#        sums = lim.sum(axis=0).sum(axis=0)
#        
#        thr = int(0.8 * waveSegs.shape[0] * waveSegs.shape[1])
#        
#        good = pl.find(sums > thr)
        
#        # compute distances to mean wave
#        dists = np.zeros((waveSegs.shape[2], ))
#        for i in xrange(waveSegs.shape[2]):
#            dists[i] = wavelets.waveDist(waveSegs[:, :, i], meanWave, 400.)
#        # rejection threshold
#        lim = dists.mean() + 0.1 * dists.std(ddof=1)
#        good = pl.find(dists < lim)
#        bad = pl.find(dists >= lim)
#        
#        # T wave location from wavelets
#        T = np.zeros((waveSegs.shape[2]))
#        Vt = np.zeros((waveSegs.shape[2]))
#        P = np.zeros((waveSegs.shape[2]))
#        Vp = np.zeros((waveSegs.shape[2]))
#        a = int(0.3 * Fs) # 300 ms
#        b = int(0.15 * Fs) # 150 ms
#        for i in xrange(waveSegs.shape[2]):
#            # P wave
#            ext, ty = findExtrema(waveSegs[:b, 1, i])
#            ext = ext[ty == -1] # maxima
#            aux = waveSegs[ext, 1, i]
#            try:
#                P[i] = ext[np.argmax(aux)]
#            except ValueError:
#                P[i] = np.argmax(waveSegs[:b, 1, i])
#            Vp[i] = waveSegs[P[i], 1, i]
#            # T wave
#            ext, ty = findExtrema(waveSegs[a:, 1, i])
#            ext = a + ext[ty == -1] # maxima
#            aux = waveSegs[ext, 1, i]
#            try:
#                T[i] = ext[np.argmax(aux)]
#            except ValueError:
#                T[i] = a + np.argmax(waveSegs[a:, 1, i])
#            Vt[i] = waveSegs[T[i], 1, i]
#        
#        # plot
#        fig = pl.figure(figsize=(16, 9))
#        ax = fig.add_subplot(211)
#        
#        ax.plot(waveSegs[:, 1, :], 'b', alpha=0.6)
#        ax.set_title('All')
#        
#        markerline, stemlines, baseline = ax.stem(P, Vp, 'm-')
#        pl.setp(markerline, 'markerfacecolor', 'g')
#        pl.setp(stemlines, 'alpha', 0.9)
#        
#        markerline, stemlines, baseline = ax.stem(T, Vt, 'r-')
#        pl.setp(markerline, 'markerfacecolor', 'g')
#        pl.setp(stemlines, 'alpha', 0.9)
#        
#        ax.axhline(0, color='g', linewidth=2)
#        
#        
#        ax = fig.add_subplot(212)
#        
#        ax.plot(waveSegs[:, 1, good], 'b', alpha=0.6)
#        ax.set_title('Without outliers (kept %2.2f %%)' % (100 * float(len(good)) / waveSegs.shape[2]))
#        
#        markerline, stemlines, baseline = ax.stem(P[good], Vp[good], 'm-')
#        pl.setp(markerline, 'markerfacecolor', 'g')
#        pl.setp(stemlines, 'alpha', 0.9)
#        
#        markerline, stemlines, baseline = ax.stem(T[good], Vt[good], 'r-')
#        pl.setp(markerline, 'markerfacecolor', 'g')
#        pl.setp(stemlines, 'alpha', 0.9)
#        
#        ax.axhline(0, color='g', linewidth=2)
#        
#        
#        fig.savefig(os.path.join(path, 'rec_%d.png' % rec), dpi=250, bbox_inches='tight')
#        pl.close(fig)
#    
#    
#    # close db
#    db.close()
    
    
    
#    records = [82, 84, 86, 88, 90, 92, 94, 96, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 122, 124, 126, 128, 130]
#    thrV = np.arange(200., 1001., 100.)
#    
#    for rec in records:
#        print rec
#        
#        # get wavelet segments
#        with h5py.File('D:/BioMESH/Databases/CVP/rec_%d.hdf5' % rec, 'r') as fid:
#            data = fid['signals/ECG/hand/Wavelets/rbio3-3/Segments/4to6/signal0']
#            segs = data[...]
#            mdata = json.loads(data.attrs['json'])
#            recSegs = fid['signals/ECG/hand/Wavelets/rbio3-3/Reconstruction/Segments/signal0'][...]
#        
#        # mean wave
#        meanWave = wavelets.meanWave(segs)
#        
#        for thr in thrV:
#            print thr
#            
#            # compute distances to mean wave
#            dists = np.zeros((segs.shape[2], ))
#            for i in xrange(segs.shape[2]):
#                dists[i] = wavelets.waveDist(segs[:, :, i], meanWave, thr)
#            
#            # rejection threshold
#            lim = dists.mean() + dists.std(ddof=1)
#            
#            good = pl.find(dists < lim)
#            bad = pl.find(dists >= lim)
#            accepted = len(good) / float(len(good) + len(bad))
#            rejected = 1 - accepted
#            
#            # plot
#            fig = pl.figure(figsize=(12, 12))
#            
#            ax = fig.add_subplot(4, 2, 1)
#            ax.plot(recSegs[:, good], 'b', alpha=0.7)
#            ax.set_title('Retained: %2.2f %%' % (100*accepted))
#            ax.set_ylabel('Reconstruction')
#            
#            ax = fig.add_subplot(4, 2, 2)
#            ax.plot(recSegs[:, bad], 'r', alpha=0.7)
#            ax.set_title('Removed: %2.2f %%' % (100*rejected))
#            
#            for i in xrange(3):
#                ax = fig.add_subplot(4, 2, 2*(i+1)+1)
#                ax.plot(segs[:, i, good], 'b', alpha=0.7)
#                ax.set_ylabel('Level %d' % i)
#                
#                ax = fig.add_subplot(4, 2, 2*(i+1)+2)
#                ax.plot(segs[:, i, bad], 'r', alpha=0.7)
#            
#            # save fig
#            fig.savefig(os.path.join(path, 'outliers_%d_%3.0f.png' % (rec, thr)), dpi=250, bbox_inches='tight')
#            pl.close(fig)
    
    
    
#    Fs = float(mdata['sampleRate'])
#    
#    s1 = segs[:, 1]
#    # s1 = signal
#    out1 = SIFT(s1, 5., Fs)
#    kp1 = out1[2]
#    ds1 = out1[3]
#    
#    s2 = segs[:, 2]
#    out2 = SIFT(s2, 5., Fs)
#    kp2 = out2[2]
#    ds2 = out2[3]
#    s2 = s2 + 50 + s1.max() - s2.min()
#    
#    matrix = np.zeros((len(kp1), len(kp2)))
#    for i in xrange(len(kp1)):
#        for j in xrange(len(kp2)):
#            matrix[i, j] = np.sqrt(np.sum((ds1[i]-ds2[j])**2))
#    
#    m1 = np.argmin(matrix, axis=1)
#    m2 = np.argmin(matrix, axis=0)
#    
#    fig = pl.figure(figsize=(16, 9))
#    ax1 = fig.add_subplot(2, 1, 1)
#    ax2 = fig.add_subplot(2, 1, 2)
#    
#    ax1.plot(s1)
#    ax1.plot(s2)
#    ax1.plot(kp1, s1[kp1], 'ro', kp2, s2[kp2], 'ro')
#    for i in xrange(len(m1)):
#        ax1.plot([kp1[i], kp2[m1[i]]], [s1[kp1[i]], s2[kp2[m1[i]]]])
#    
#    ax2.plot(s1)
#    ax2.plot(s2)
#    ax2.plot(kp1, s1[kp1], 'ro', kp2, s2[kp2], 'ro')
#    for i in xrange(len(m2)):
#        ax2.plot([kp1[m2[i]], kp2[i]], [s1[kp1[m2[i]]], s2[kp2[i]]])
#        
#    pl.show()
    
    
    # connect to db
    db = biomesh.biomesh('CVP', host='193.136.222.234', dstPath='D:/BioMESH', sync=True)
    
    allrecs = set(db.records.getAll()['idList'])
    # records = [0]
    done = set([0])
    records = list(allrecs - done)
    
    # wavelts
    waves = ['rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio5.5', 'coif5', 'db3', 'db8']
    # waves = ['rbio3.9', 'rbio3.5']
    L = 8
    
    for rec in records:
        print rec
        
        # load raw signal
        raw = db.records.getSignal(rec, '/ECG/hand/raw', 0)
        signal = raw['signal']
        Fs = float(raw['mdata']['sampleRate'])
        
        # load filtered signal
        fid = gzip.open('Y:\\CVP-Results\\data\\filtered\\zee5to20\\output-%d' % rec, 'rb')
        filtered = cPickle.load(fid)
        fid.close()
        # save
        mdata = {'labels': ['filtered'],
                 'units': raw['mdata']['units'],
                 'sampleRate': Fs,
                 'source': 'signals/ECG/hand/raw/signal0',
                 'filter': 'zee4to5'}
        out1 = db.records.addSignal(rec, '/ECG/hand/zee5to20', filtered, mdata)
        
        # load R positions and segments
        fid = gzip.open('Y:\\CVP-Results\\data\\segmentation\\engzee\\output-%d' % rec, 'rb')
        data = cPickle.load(fid)
        fid.close()
        R = data['R']
        segs = data['segments']
        # save
        mdata = {'source': 'signals' + out1['signalType'] + '/' + out1['signalRef'],
                 'algorithm': 'engzee'}
        out2 = db.records.addEvent(rec, '/ECG/hand/zee5to20/R/engzee', R, [], mdata)
        mdata = {'labels': ['Segments'],
                 'units': raw['mdata']['units'],
                 'sampleRate': Fs,
                 'source': {'signal': 'signals' + out1['signalType'] + '/' + out1['signalRef'],
                            'R': 'events' + out2['eventType'] + '/' + out2['eventRef']}}
        out3 = db.records.addSignal(rec, '/ECG/hand/zee5to20/Segments/engzee', segs, mdata)
        
        for wave in waves:
            print wave
            wave_ty = wave.replace('.', '-')
            
            # wavelet decomposition
            coeffs = wavelets.RDWT(signal, wave, L)
        
            # reconstruct [4, 5, 6, 7]
            coeffsR = coeffs.copy()
            coeffsR[:, :4] = 0
            back1 = wavelets.iRDWT(coeffsR, wave, zeros=True)
            # reconstruct [4, 5, 6]
            coeffsR[:, 7] = 0
            back2 = wavelets.iRDWT(coeffsR, wave, zeros=True)
            
            # segment ECG
            # R from engzee
            R_a = correctR(coeffs[:, 4], R, Fs, left=0, right=0.25, absFlag=False)
            segs_a = wavelets.extractSegments(coeffs, R_a, Fs)
            
            R_b = correctR(back1, R, Fs, left=0, right=0.25, absFlag=True)
            segs_b = segment(back1, R_b, Fs)
            
            R_c = correctR(back2, R, Fs, left=0, right=0.25, absFlag=True)
            segs_c = segment(back2, R_c, Fs)
            
            # R from wavelet
            res = wavelets.waveletSegments(coeffs, Fs)
            R_d = res['R']
            segs_d = res['segments']
            
            R_e = correctR(back1, R_d, Fs, left=0.1, right=0.1, absFlag=True)
            segs_e = segment(back1, R_e, Fs)
            
            R_f = correctR(back2, R_d, Fs, left=0.1, right=0.1, absFlag=True)
            segs_f = segment(back2, R_f, Fs)
            
            
            
            # save to db
            base_ty = '/ECG/hand/Wavelets/' + wave_ty
            
            # wavelet coefficients
            ty = base_ty
            mdata = {'labels': [wave_ty],
                     'units': raw['mdata']['units'],
                     'sampleRate': Fs,
                     'source': 'signals/ECG/hand/raw/signal0',
                     'wavelet': {'name': wave, 'level': L}}
#            print ty
#            print mdata
            out4 = db.records.addSignal(rec, ty, coeffs, mdata)
    #        out1 = {'signalType': '/ECG/hand/Wavelets/rbio3-5', 'signalRef': 'signal0'}
            
            # reconstruction 1
            mdata = {'labels': ['Segments'],
                     'units': raw['mdata']['units'],
                     'sampleRate': Fs,
                     'source': 'signals' + out4['signalType'] + '/' + out4['signalRef'],
                     'wavelet': {'name': wave, 'level': L, 'reconstruction': [4, 5, 6, 7]}}
            ty = base_ty + '/Reconstruction1'
#            print ty
#            print mdata
            out5 = db.records.addSignal(rec, ty, back1, mdata)
            
            # reconstruction 2
            mdata = {'labels': ['Segments'],
                     'units': raw['mdata']['units'],
                     'sampleRate': raw['mdata']['sampleRate'],
                     'source': 'signals' + out4['signalType'] + '/' + out4['signalRef'],
                     'wavelet': {'name': wave, 'level': L, 'reconstruction': [4, 5, 6]}}
            ty = base_ty + '/Reconstruction2'
#            print ty
#            print mdata
            out6 = db.records.addSignal(rec, ty, back2, mdata)
            
            # ECG segmentation
            # R from engzee
            mdata = {'source': 'signals' + out1['signalType'] + '/' + out1['signalRef'],
                     'algorithm': 'engzee',
                     'notes': 'With time correction.'}
            ty = base_ty + '/R/engzee'
            out7 = db.records.addEvent(rec, ty, R_a, [], mdata)
            mdata = {'labels': ['Segments'],
                     'units': raw['mdata']['units'],
                     'sampleRate': Fs,
                     'source': {'signal': 'signals' + out4['signalType'] + '/' + out4['signalRef'],
                                'R': 'events' + out7['eventType'] + '/' + out7['eventRef']}}
            ty = base_ty + '/Segments/engzee'
            out8 = db.records.addSignal(rec, ty, segs_a, mdata)
            
            
            mdata = {'source': 'signals' + out1['signalType'] + '/' + out1['signalRef'],
                     'algorithm': 'engzee',
                     'notes': 'With time correction.'}
            ty = base_ty + '/Reconstruction1/R/engzee'
            out9 = db.records.addEvent(rec, ty, R_b, [], mdata)
            mdata = {'labels': ['Segments'],
                     'units': raw['mdata']['units'],
                     'sampleRate': Fs,
                     'source': {'signal': 'signals' + out5['signalType'] + '/' + out5['signalRef'],
                                'R': 'events' + out9['eventType'] + '/' + out9['eventRef']}}
            ty = base_ty + '/Reconstruction1/Segments/engzee'
            out10 = db.records.addSignal(rec, ty, segs_b, mdata)
            
            
            mdata = {'source': 'signals' + out1['signalType'] + '/' + out1['signalRef'],
                     'algorithm': 'engzee',
                     'notes': 'With time correction.'}
            ty = base_ty + '/Reconstruction2/R/engzee'
            out11 = db.records.addEvent(rec, ty, R_c, [], mdata)
            mdata = {'labels': ['Segments'],
                     'units': raw['mdata']['units'],
                     'sampleRate': Fs,
                     'source': {'signal': 'signals' + out6['signalType'] + '/' + out6['signalRef'],
                                'R': 'events' + out11['eventType'] + '/' + out11['eventRef']}}
            ty = base_ty + '/Reconstruction2/Segments/engzee'
            out12 = db.records.addSignal(rec, ty, segs_c, mdata)
            
            # R from wavelet
            mdata = {'source': 'signals' + out4['signalType'] + '/' + out4['signalRef'],
                     'algorithm': 'wavelet'}
            ty = base_ty + '/R/waveletSegmentation'
            out13 = db.records.addEvent(rec, ty, R_d, [], mdata)
            mdata = {'labels': ['Segments'],
                     'units': raw['mdata']['units'],
                     'sampleRate': Fs,
                     'source': {'signal': 'signals' + out4['signalType'] + '/' + out4['signalRef'],
                                'R': 'events' + out13['eventType'] + '/' + out13['eventRef']}}
            ty = base_ty + '/Segments/waveletSegmentation'
            out14 = db.records.addSignal(rec, ty, segs_d, mdata)
            
            
            mdata = {'source': 'signals' + out4['signalType'] + '/' + out4['signalRef'],
                     'algorithm': 'wavelet',
                     'notes': 'With time correction.'}
            ty = base_ty + '/Reconstruction1/R/waveletSegmentation'
            out15 = db.records.addEvent(rec, ty, R_e, [], mdata)
            mdata = {'labels': ['Segments'],
                     'units': raw['mdata']['units'],
                     'sampleRate': Fs,
                     'source': {'signal': 'signals' + out5['signalType'] + '/' + out5['signalRef'],
                                'R': 'events' + out15['eventType'] + '/' + out15['eventRef']}}
            ty = base_ty + '/Reconstruction1/Segments/waveletSegmentation'
            out16 = db.records.addSignal(rec, ty, segs_e, mdata)
            
            
            mdata = {'source': 'signals' + out4['signalType'] + '/' + out4['signalRef'],
                     'algorithm': 'wavelet',
                     'notes': 'With time correction.'}
            ty = base_ty + '/Reconstruction2/R/waveletSegmentation'
            out17 = db.records.addEvent(rec, ty, R_f, [], mdata)
            mdata = {'labels': ['Segments'],
                     'units': raw['mdata']['units'],
                     'sampleRate': Fs,
                     'source': {'signal': 'signals' + out6['signalType'] + '/' + out6['signalRef'],
                                'R': 'events' + out17['eventType'] + '/' + out17['eventRef']}}
            ty = base_ty + '/Reconstruction2/Segments/waveletSegmentation'
            out18 = db.records.addSignal(rec, ty, segs_f, mdata)
        
#        # plot
#        fig = pl.figure(figsize=(16, 9))
#        ax = fig.add_subplot(221)
#        ax.plot(back1)
#        ax.plot(newR1, back1[newR1], 'go')
#        ax.set_title('[4, 5, 6]')
#        ax = fig.add_subplot(223)
#        ax.plot(segs1, 'b', alpha=0.7)
#        ax = fig.add_subplot(222)
#        ax.plot(back2)
#        ax.plot(newR2, back2[newR2], 'go')
#        ax.set_title('[4, 5, 6, 7]')
#        ax = fig.add_subplot(224)
#        ax.plot(segs2, 'b', alpha=0.7)
#        # pl.show()
#        fig.savefig('C:\\Users\\Carlos\\testWaveletDist\\waveletReconstruction\\rbio3-5\\waveletRec_%d.png' % rec, dpi=250, bbox_inches='tight')
#        pl.close(fig)
    
    # close db
    db.close()
    
    
    
#            # save to db
#        base_ty = '/ECG/hand/Wavelets/' + wave_ty
#        
#        # wavelet coefficients
#        ty = base_ty
#        mdata = {'labels': [wave_ty],
#                 'units': data['mdata']['units'],
#                 'sampleRate': data['mdata']['sampleRate'],
#                 'source': 'signals/ECG/hand/raw/signal0',
#                 'wavelet': {'name': wave, 'level': L}}
#        print ty
##        print mdata
#        out1 = db.records.addSignal(rec, ty, coeffs, mdata)
##        out1 = {'signalType': '/ECG/hand/Wavelets/rbio3-5', 'signalRef': 'signal0'}
#        
#        # wavelet segmentation
#        ty = base_ty + '/R'
#        mdata = {'source': 'signals' + out1['signalType'] + '/' + out1['signalRef'],
#                 'wavelet': {'name': wave, 'level': L}}
#        print ty
##        print mdata
#        out2 = db.records.addEvent(rec, ty, R, [], mdata)
#        
#        # wavelet segments (all)
#        ty = base_ty + '/Segments'
#        mdata = {'labels': ['Segments'],
#                 'units': data['mdata']['units'],
#                 'sampleRate': data['mdata']['sampleRate'],
#                 'source': 'signals' + out1['signalType'] + '/' + out1['signalRef'],
#                 'wavelet': {'name': wave, 'level': L}}
#        print ty
##        print mdata
#        out3 = db.records.addSignal(rec, ty, waveSegs, mdata)
#        
#        # wavelet segments (4 to 6)
#        ty = base_ty + '/Segments/4to6'
#        mdata = {'labels': ['Segments'],
#                 'units': data['mdata']['units'],
#                 'sampleRate': data['mdata']['sampleRate'],
#                 'source': 'signals' + out1['signalType'] + '/' + out1['signalRef'],
#                 'wavelet': {'name': wave, 'level': L}}
#        print ty
##        print mdata
#        out4 = db.records.addSignal(rec, ty, waveSegs[:, 4:7], mdata)
#        
#        # reconstruction 1
#        mdata = {'labels': ['Segments'],
#                 'units': data['mdata']['units'],
#                 'sampleRate': data['mdata']['sampleRate'],
#                 'source': 'signals' + out1['signalType'] + '/' + out1['signalRef'],
#                 'wavelet': {'name': wave, 'level': L, 'reconstruction': [4, 5, 6]}}
#        ty = base_ty + '/Reconstruction1'
#        print ty
##        print mdata
#        out5 = db.records.addSignal(rec, ty, back1, mdata)
##        out5 = {'signalType': '/ECG/hand/Wavelets/rbio3-5/Reconstruction1', 'signalRef': 'signal0'}
#        
#        # segments 1
#        mdata = {'labels': ['Segments'],
#                 'units': data['mdata']['units'],
#                 'sampleRate': data['mdata']['sampleRate'],
#                 'source': 'signals' + out5['signalType'] + '/' + out5['signalRef'],
#                 'wavelet': {'name': wave, 'level': L, 'reconstruction': [4, 5, 6]}}
#        ty = base_ty + '/Reconstruction1/Segments'
#        print ty
##        print mdata
#        out6 = db.records.addSignal(rec, ty, segs1, mdata)
#        
#        # new R 1
#        mdata = {'source': 'signals' + out5['signalType'] + '/' + out5['signalRef'],
#                 'wavelet': {'name': wave, 'level': L, 'reconstruction': [4, 5, 6]}}
#        ty = base_ty + '/R/Reconstruction1'
#        print ty
##        print mdata
#        out7 = db.records.addEvent(rec, ty, newR1, [], mdata)
#        
#        # reconstruction 2
#        mdata = {'labels': ['Segments'],
#                 'units': data['mdata']['units'],
#                 'sampleRate': data['mdata']['sampleRate'],
#                 'source': 'signals' + out1['signalType'] + '/' + out1['signalRef'],
#                 'wavelet': {'name': wave, 'level': L, 'reconstruction': [4, 5, 6, 7]}}
#        ty = base_ty + '/Reconstruction2'
#        print ty
##        print mdata
#        out8 = db.records.addSignal(rec, ty, back2, mdata)
##        out8 = {'signalType': '/ECG/hand/Wavelets/rbio3-5/Reconstruction2', 'signalRef': 'signal0'}
#        
#        # segments 2
#        mdata = {'labels': ['Segments'],
#                 'units': data['mdata']['units'],
#                 'sampleRate': data['mdata']['sampleRate'],
#                 'source': 'signals' + out8['signalType'] + '/' + out8['signalRef'],
#                 'wavelet': {'name': wave, 'level': L, 'reconstruction': [4, 5, 6, 7]}}
#        ty = base_ty + '/Reconstruction2/Segments'
#        print ty
##        print mdata
#        out9 = db.records.addSignal(rec, ty, segs2, mdata)
#        
#        # new R 1
#        mdata = {'source': 'signals' + out8['signalType'] + '/' + out8['signalRef'],
#                 'wavelet': {'name': wave, 'level': L, 'reconstruction': [4, 5, 6, 7]}}
#        ty = base_ty + '/R/Reconstruction2'
#        print ty
##        print mdata
#        out10 = db.records.addEvent(rec, ty, newR2, [], mdata)
    
    
    
    
    
    
#        
#        
#        pl.plot(segs)
#        pl.show()
#
#        nb = segs.shape[1]
#        matrix = np.zeros((nb, nb))
#        vector = []
#        for i in xrange(nb):
#            for j in xrange(i+1, nb):
#                dist = corrDist(segs[:, i], segs[:, j])
#                matrix[i, j] = matrix[j, i] = dist
#                vector.append(dist)
#        
#        out = pl.hist(vector, 200)
#        pl.show()
#        
#        pl.matshow(matrix)
#        pl.show()
        
        
    
    
#    recs = db.records.getAll()['idList']
#    for rec in recs:
#        data = db.records.getSignal(rec, '/ECG/hand/Wavelets/rbio3-3/Segments', 0)
#        mdata = data['mdata']
#        mdata.pop('name')
#        mdata.pop('type')
#        db.records.addSignal(rec, '/ECG/hand/Wavelets/rbio3-3/Segments/4to6', data['signal'][:, 4:7, :], mdata, compress=True)
        
    
#    wave = 'rbio3-3'
#    
#    # config to old HDF5
#    config = {'source': 'HDF5',
#              'path': 'D:\\BioMESH\\Databases\\CVP_old',
#              'experiments': ['T1', 'T2'],
#              'mapper': {'coeffsRec': 'signals/ECG/hand/Recumbent/Wavelets/' + wave,
#                         'coeffsSit': 'signals/ECG/hand/Sitting/Wavelets/' + wave,
#                         'segsRec': 'signals/ECG/hand/Recumbent/Wavelets/'  + wave + '/Segments',
#                         'segsSit': 'signals/ECG/hand/Sitting/Wavelets/' + wave + '/Segments',
#                         'RRec': 'events/ECG/hand/Recumbent/Wavelets/'  + wave + '/R',
#                         'RSit': 'events/ECG/hand/Sitting/Wavelets/'  + wave + '/R'}
#    }
#    
#    Map = {'coeffs': '/ECG/hand/Wavelets/' + wave,
#           'segs': '/ECG/hand/Wavelets/' + wave + '/Segments',
#           'R': '/ECG/hand/Wavelets/' + wave + '/R',
#    }
#    store = dm.Store(config)
#    recs = store.dbmetada()
#    
#    for ID in recs:
#        print "Old ID", ID
#        doc = recs[ID]
#        exp = doc['experiment']
#        sub = doc['subject']
#        src = doc['source']
#        
#        # recumbent
#        recId = db.records.get(refine={'experiment': exp + '-Recumbent', 'subject': sub}, restrict={'_id': 1})['docList'][0]['_id']
#        print "New ID - Recumbent", recId
#        # coeffs
#        data = store.db2data('coeffsRec', refine={'_id': [ID]})[ID][0]
#        matrix = wavelets.coeffs2Matrix(data['signal'])
#        data['mdata']['source'] = 'signals/ECG/hand/raw/signal0'
#        data['mdata'].pop('type')
#        data['mdata'].pop('name')
#        db.records.addSignal(recId, Map['coeffs'], matrix, data['mdata'], compress=True)
#        # segs
#        data = store.db2data('segsRec', refine={'_id': [ID]})[ID][0]
#        segs = data['signal'].swapaxes(0, 1)
#        data['mdata']['source'] = Map['coeffs'] + '/signal0'
#        data['mdata'].pop('type')
#        data['mdata'].pop('name')
#        db.records.addSignal(recId, Map['segs'], segs, data['mdata'])
#        # R
#        data = store.db2data('RRec', refine={'_id': [ID]})[ID][0]
#        data['mdata']['source'] = Map['segs'] + '/signal0'
#        data['mdata'].pop('type')
#        data['mdata'].pop('name')
#        db.records.addEvent(recId, Map['R'], data['timeStamps'], [], data['mdata'])
#        
#        # sitting
#        recId = db.records.get(refine={'experiment': exp + '-Sitting', 'subject': sub}, restrict={'_id': 1})['docList'][0]['_id']
#        print "New ID - Sitting", recId
#        # coeffs
#        data = store.db2data('coeffsSit', refine={'_id': [ID]})[ID][0]
#        matrix = wavelets.coeffs2Matrix(data['signal'])
#        data['mdata']['source'] = 'signals/ECG/hand/raw/signal0'
#        data['mdata'].pop('type')
#        data['mdata'].pop('name')
#        db.records.addSignal(recId, Map['coeffs'], matrix, data['mdata'], compress=True)
#        # segs
#        data = store.db2data('segsSit', refine={'_id': [ID]})[ID][0]
#        segs = data['signal'].swapaxes(0, 1)
#        data['mdata']['source'] = Map['coeffs'] + '/signal0'
#        data['mdata'].pop('type')
#        data['mdata'].pop('name')
#        db.records.addSignal(recId, Map['segs'], segs, data['mdata'])
#        # R
#        data = store.db2data('RSit', refine={'_id': [ID]})[ID][0]
#        data['mdata']['source'] = Map['segs'] + '/signal0'
#        data['mdata'].pop('type')
#        data['mdata'].pop('name')
#        db.records.addEvent(recId, Map['R'], data['timeStamps'], [], data['mdata'])
        
        
    
    
    
#    # close db
#    db.close()
#    store.close()
        
        
        
        
        
        
        
