"""
.. module:: preprocessing
   :platform: Unix, Windows
   :synopsis: This module provides various functions to...

.. moduleauthor:: Filipe Canento, Carlos Carreiras


"""


# Imports
# built-in
from multiprocessing import Process, Queue, Manager
from Queue import Empty

# 3rd party
import numpy
import scipy
import scipy.signal as ss
import scipy.stats as st

# BiometricsPyKit
from wavelets import wavelets

# BioSPPy
import filt
from ecg import tools as ecgtools

# Global Variables
global NUMBER_PROCESSES
NUMBER_PROCESSES = 5


def patterns2strings(patterns):
    res = []
    for i, si in enumerate(patterns):
        line = ''.join('%d'%i for i in si)
        res.append(line)
    return res

def normalization(data, ntype):
    if ntype == 'mean_maxs':
        data /= scipy.mean(numpy.max(data, 1))
    elif ntype == 'median_maxs':
        data /= scipy.median(numpy.max(data, 1))
    else:
        raise TypeError, "Normalization type %s not implemented."


def normSelector(method):
    # select a normalization method
    
    if method == 'norm1':
        fcn = norm1
    elif method == 'norm2':
        fcn = norm2
    elif method == 'norm3':
        fcn = norm3
    elif method == 'norm4':
        fcn = norm4
    elif method == 'norm5':
        fcn = norm5
    elif method == 'norm6':
        fcn = norm6
    elif method == 'norm7':
        fcn = norm7
    elif method == 'norm8':
        fcn = norm8
    elif method == 'norm9':
        fcn = norm9
    elif method == 'arcLength':
        fcn = normArcLength
    else:
        raise TypeError, "Method %s not implemented." % method
    
    return fcn


def norm1(data):
    # for 2D data, nb_samples * nb_features
    
    # ensure numpy
    res = numpy.array(data).T
    
    res -= res.mean(axis=0)
    res /= res.std(axis=0, ddof=1)
    
    return res.T

def norm2(data):
    # for 2D data, nb_samples * nb_features
    
    # ensure numpy
    res = numpy.array(data).T
    
    res -= res.mean(axis=0)
    res /= res.max(axis=0)
    
    return res.T

def norm3(data):
    # for 2D data, nb_samples * nb_features
    
    # ensure numpy
    res = numpy.array(data).T
    
    res -= res.mean(axis=0)
    aux = numpy.sum(res ** 2, axis=0)
    res /= numpy.sqrt(aux)
    
    return res.T

def norm4(data, R=200):
    # for 2D data, nb_samples * nb_features
    
    # ensure numpy
    res = numpy.array(data).T
    
    res -= res.mean(axis=0)
    aux = numpy.array(res[R, :])
    aux[numpy.nonzero(aux == 0)] = 1.
    res /= aux
    
    return res.T

def norm5(data):
    # for 2D data, nb_samples * nb_features
    
    # ensure numpy
    res = numpy.array(data).T
    
    res -= res.mean(axis=0)
    res /= (res.max(axis=0) - res.min(axis=0))
    
    return res.T

def norm6(data):
    # for 2D data, nb_samples * nb_features
    
    # ensure numpy
    res = numpy.array(data).T
    
    res -= res.min(axis=0)
    aux = res.max(axis=0)
    aux[numpy.nonzero(aux == 0)] = 1.
    res /= aux
    
    return res.T

def norm7(data):
    # for 2D data, nb_samples * nb_features
    
    # ensure numpy
    res = numpy.array(data).T
    
    res -= res.min(axis=0)
    aux = numpy.sum(res ** 2, axis=0)
    res /= numpy.sqrt(aux)
    
    return res.T

def norm8(data, R=200):
    # for 2D data, nb_samples * nb_features
    
    # ensure numpy
    res = numpy.array(data).T
    
    res -= res.min(axis=0)
    aux = numpy.array(res[R, :])
    aux[numpy.nonzero(aux == 0)] = 1.
    res /= aux
    
    return res.T

def norm9(data, R=200, sigma=1.5):
    # for 2D data, nb_samples * nb_features
    
    # ensure numpy
    res = numpy.array(data)
    
    # get gaussian window
    win = normalWindow(res.shape[1], R, sigma)
    res = win[numpy.newaxis, :] * res
    
    res = norm6(res)
    
    return res

def sparseNorm(data):
    # for 2D data, nb_samples * nb_features
    
    # ensure numpy
    res = numpy.array(data)
    
    # normalize features
    ftr = numpy.linalg.norm(res, axis=0)[numpy.newaxis, :]
    ftr[numpy.nonzero(ftr == 0)] = 1.
    res /= ftr
    
    # normalize samples
    spl = numpy.linalg.norm(res, axis=1)[:, numpy.newaxis]
    spl[numpy.nonzero(spl == 0)] = 1.
    res /= spl
    
    return res

def normArcLength(data):
    # for 2D data, nb_samples * nb_features
    
    dy = numpy.diff(data, n=1, axis=1)
    a = numpy.sqrt(2 / numpy.sum(dy**2, axis=1))
    a = a.reshape((len(data), 1))
    
    res = a * data
    
    return res

def normalWindow(nb, mu, sigma):
    # adjustable gaussian window
    
    rv = st.norm()
    
    # get 0.999 percentile
    x = numpy.linspace(rv.ppf(0.001), rv.ppf(0.999), nb)
    
    # adjust to mu and sigma
    mu = x[mu]
    x = (x - mu) / sigma
    
    # get pdf
    y = rv.pdf(x)
    
    return y
    

def heartRateSifter(signal, R, Fs=1000., edges=None):
    # separate heartbeats based on heart rate
    
    # compute heart rate
    hr = (60. * Fs) / numpy.diff(R)
    
    # filter out acute "dysrhythmias"
    fhr = ss.medfilt(hr, 3)
    th = 2 * fhr.std(ddof=1)
    dhr = numpy.abs(hr - fhr)
    good = numpy.nonzero(dhr < th)[0]
    length = len(good)
    
    # sieve edges
    if edges is None:
        # default edges
        aux = numpy.arange(40., 200., 5.)
        edges = numpy.array([aux, aux + 10.]).T
        edges = numpy.vstack(([0., 40.], edges, [200., numpy.inf]))
    else:
        # ensure numpy
        edges = numpy.array(edges)
    
    # extract segments
    gR = R[good + 1]
    hrD = dict([(gR[i], hr[good][i]) for i in xrange(length)])
    segs, nR = ecgtools.extractHeartbeats(signal, gR, Fs)
    
    # sift heart rate
    indx = [[] for i in xrange(len(edges))]
    for i in xrange(len(nR)):
        left = hrD[nR[i]] >= edges[:, 0]
        right = hrD[nR[i]] < edges[:, 1]
        levels = numpy.nonzero(left * right)[0]
        for item in levels:
            indx[item].append(i)
#    for i in good:
#        left = hr[i] >= edges[:, 0]
#        right = hr[i] < edges[:, 1]
#        levels = numpy.nonzero(left * right)[0]
#        for item in levels:
#            indx[item].append(i)
#    
#    # sift R indices
#    RS = []
#    for item in indx:
#        RS.append(R[numpy.array(item, dtype='int') + 1])
#    
#    # sift heartbeat waveforms
#    segs = []
#    for item in RS:
#        segs.append(ecgtools.extractHeartbeats(signal, item, Fs)[0])
    
    return edges, segs, indx


def heartRateExtractor(data, sift, selection=None):
    # separate the segments in data according to the sift
    
    if selection is None:
        selection = numpy.arange(len(data), dtype='int')
    
    segs = []
    for item in sift:
        aux = numpy.intersect1d(item, selection, assume_unique=True)
        if len(aux) > 0:
            segs.append(numpy.array(data[aux]))
        else:
            segs.append(numpy.array([]))
    
    return segs

    
def do_filter_work(work_queue, output):
    while 1:
        try:
            q = work_queue.get(block=False)
            function = q['function']
            data = q['data']
            parameters = q['parameters']
            recid = q['recid']
            output[recid] = function(data, **parameters)
            
        except Empty:
            break    
    
def filter_data(data, parameters):
    """
    Filters input data according to specified parameters.

    Input:
    
        data (dict): input data. 
                     example: data = {1: record 1 signal, 2: record 2 signal, ...}
        
        parameters (dict): filter parameters.
                           Example: parameters = {'filter': 'butter', 
                                                  'order': 4,
                                                  'low_cutoff': 100.,
                                                  'sampling_rate': 1000.}
                                                  
    Output:
        
        output (dict): output filtered data where keys correspond to record id numbers.
                       example: output = {1: record 1 filtered signal, 2: record 2 filtered signal}

    Configurable fields:{"name": "??.??", "config": {}, "inputs": ["data", "parameters"], "outputs": ["output"]}

    See Also:

    Notes:

    Example:
       
    """   
    if parameters['filter'] == 'butter':
        filter2use = butter
    elif parameters['filter'] == 'fir':
        filter2use = firfilt
    elif parameters['filter'] == 'wavelet':
        filter2use = wavelets.RDWT
    else:
        raise TypeError, "Filter %s not implemented."%parameters['filter']
    # create work queue
    work_queue = Queue()
    manager = Manager()
    output = manager.dict()
    output['info'] = parameters
    parameters.pop('filter')
    # fill queue
    for recid in data:
        if recid == 'info': continue
        work_queue.put({'function': filter2use, 'data': data[recid], 'parameters': parameters, 'recid': recid})
    # create N processes and associate them with the work_queue and do_work function
    processes = [Process(target=do_filter_work, args=(work_queue,output,)) for _ in range(NUMBER_PROCESSES)]
    # lauch processes
    for p in processes: p.start()
    # wait for processes to finish
    print "waiting ..."
    for p in processes: p.join()
    print "wait is over ..."
    for p in processes: p.terminate()
        
    return output    

def butter(signal=None, low_cutoff=None, high_cutoff=None, zi=None, order=4, sampling_rate=1000.):
    """

    Butterworth filtering.

    Input:
        signal (array): input signal.
        
        'order' (int): filter order.
        
        'low_cutoff' (float): low cutoff frequency.
        
        'high_cutoff' (float): high cutoff frequency.
        
        'sampling rate' (float): sampling rate.
        
        'zi' (array): initial conditions

    Output:
        output_signal (array): output FIR-filtered signal. Output signal length = length(signal) - filter order (n) 

    Configurable fields:{"name": "preprocessing.butter", "config": {"order": 4, "sampling_rate": 1000., "zi": None}, "inputs": ["signal", "low_cutoff", "high_cutoff"], "outputs": ["output_signal"]}

    See Also:
        
    Notes:
    

    Example:


    References:
        .. [1]
        
    """
    if signal is None:
        raise TypeError, "An input signal is needed."
    if not low_cutoff and not high_cutoff:
        raise TypeError, "Cutoff frequencies are needed.\n Specify at least one of the following: low_cutoff, high_cutoff."
    if low_cutoff and not high_cutoff: # low-pass filter
        [b, a] = ss.butter(order, 2.*float(low_cutoff)/sampling_rate, 'low')
    if not low_cutoff and high_cutoff: # high-pass filter
        [b, a] = ss.butter(order, 2.*float(high_cutoff)/sampling_rate, 'high')
    else: # band-pass filter        
        [b, a] = ss.butter(order, [2.*float(low_cutoff)/sampling_rate, 2.*float(high_cutoff)/sampling_rate], 'band')
    zf = zi if zi else ss.lfilter_zi(b, a) 
    output_signal, zf = ss.lfilter(b=b, a=a, x=signal, zi=zf)
    return output_signal, zf


# Band-pass FIR Filtering
def firfilt(signal=None, low_cutoff=None, high_cutoff=None, order=300, sampling_rate=1000.):
    """

    FIR Filtering. Uses firwin and lfilter from scipy.signal.

    Input:
        signal (array): input signal.
        
        low_cutoff (float): low cutoff frequency.
        
        high_cutoff (float): high cutoff frequency.

        order (int): filter order.
        
        sampling_rate (float): sampling rate.

    Output:
        output_signal (array): output FIR-filtered signal. Output signal length = length(signal) - filter order (n) 

    Configurable fields:{"name": "preprocessing.firfilt", "config": {"order" : 300, "sampling_rate": 1000.}, "inputs": ["signal", "low_cutoff", "high_cutoff"], "outputs": ["output_signal"]}

    See Also:
        scipy.signal.firwin
        scipy.signal.lfilter
        
    Notes:
    

    Example:
    

    References:
        .. [1]
        
    """
    if signal is None:
        raise TypeError, "An input signal is needed."
    if not low_cutoff and not high_cutoff:
        raise TypeError, "Cutoff frequencies are needed.\n Specify at least one of the following: low_cutoff, high_cutoff."
    if low_cutoff and not high_cutoff: # low-pass filter
        bfir1 = ss.firwin(int(order)+1, 2.*float(low_cutoff)/float(sampling_rate))
    if not low_cutoff and high_cutoff: # high-pass filter
        bfir1 = ss.firwin(int(order)+1, 2.*float(high_cutoff)/float(sampling_rate), pass_zero=False)
    else: # band-pass filter        
        bfir1 = ss.firwin(int(order)+1, [2.*float(low_cutoff)/float(sampling_rate), 2.*float(high_cutoff)/float(sampling_rate)], pass_zero=False)
        
    output_signal = ss.lfilter(bfir1, [1], signal)[order:]
    
    return output_signal


def butterFilt(signal=None, low_cutoff=None, high_cutoff=None, order=1, sampling_rate=1000.):
    """
    
    Butterworth IIR filtering. Uses filtfilt to avoid phase distortions.

    Input:
        signal (array): input signal.
        
        low_cutoff (float): low cutoff frequency.
        
        high_cutoff (float): high cutoff frequency.

        order (int): filter order.
        
        sampling_rate (float): sampling rate.

    Output:
        output_signal (array): output filtered signal.

    Configurable fields:{"name": "preprocessing.butterFilt", "config": {"order" : 300, "sampling_rate": 1000.}, "inputs": ["signal", "low_cutoff", "high_cutoff"], "outputs": ["output_signal"]}

    See Also:
        scipy.signal.filtfilt
        scipy.signal.butter
        
    Notes:
    

    Example:
    

    References:
        .. [1]
        
    """
    if signal is None:
        raise TypeError, "An input signal is needed."
    
    if not low_cutoff and not high_cutoff:
        raise TypeError, "Cutoff frequencies are needed.\n Specify at least one of the following: low_cutoff, high_cutoff."
    
    if low_cutoff and not high_cutoff:
        # low-pass
        b, a = ss.butter(order, 2.*float(low_cutoff)/float(sampling_rate), btype='lowpass')
    if not low_cutoff and high_cutoff:
        # high-pass
        b, a = ss.butter(order, 2.*float(high_cutoff)/float(sampling_rate), btype='highpass')
    else:
        # band-pass     
        b, a = ss.butter(order, [2.*float(low_cutoff)/float(sampling_rate), 2.*float(high_cutoff)/float(sampling_rate)], btype='bandpass')
        
    output_signal = ss.filtfilt(b, a, signal)
    
    return output_signal


def notchFilter(signal=None, center=None, width=None, order=300, sampling_rate=1000., harmonics=False):
    """

    FIR notch Filtering. Uses firwin and lfilter from scipy.signal.

    Input:
        signal (array): input signal.
        
        low_cutoff (float): low cutoff frequency.
        
        high_cutoff (float): high cutoff frequency.

        order (int): filter order.
        
        sampling_rate (float): sampling rate.
        
        harmonics (bool): Flag to also remove harmonics; the default is False.

    Output:
        output_signal (array): output FIR-filtered signal. Output signal length = length(signal) - filter order (n) 

    Configurable fields:{"name": "preprocessing.firfilt", "config": {"order" : 300, "sampling_rate": 1000.}, "inputs": ["signal", "low_cutoff", "high_cutoff"], "outputs": ["output_signal"]}

    See Also:
        scipy.signal.firwin
        scipy.signal.lfilter
        
    Notes:
    

    Example:
    

    References:
        .. [1]
        
    """
    
    # check inputs
    if signal is None:
        raise TypeError, "An input signal is needed."
    
    if center is None:
        raise TypeError, "A center frequency is needed."
    
    if width is None:
        raise TypeError, "A width is needed."
    
    # ensure floats
    center = float(center)
    width = float(width)
    nyq = float(sampling_rate) / 2.
    
    # freqs
    lc = center - width / 2.
    if lc <= 0:
        raise ValueError, "Left notch edge is 0 or lower."
    hc = center + width / 2.
    if hc >= nyq:
        raise ValueError, "Right notch edge is Fs/2 or higher."
    
    if harmonics:
        nb = int(nyq / center)
        
        low = []
        high = []
        for i in xrange(nb):
            low.append(i * center + lc)
            high.append(i * center + hc)
        
        low = numpy.array(low)
        high = numpy.array(high)
        
        if numpy.isclose(nyq % center, 0):
            prune = True
            nb -= 1
        else:
            prune = False
    else:
        nb = 1
        low = numpy.array([lc])
        high = numpy.array([hc])
        prune = False
    
    # merge
    cutoff = numpy.zeros(2 * nb)
    cutoff[::2] = low
    cutoff[1::2] = high
    cutoff /= nyq
    
    if prune:
        # removes last item if center frequency is multiple of Nyquist frequency (harmonics)
        cutoff = cutoff[:-1]
    
    # band-stop
    bfir = ss.firwin(int(order)+1, cutoff, pass_zero=True)
    
    # filter
    output_signal = ss.lfilter(bfir, [1], signal)[order:]
    
    return output_signal


def butterNotch(signal=None, center=None, width=None, order=1, sampling_rate=1000., harmonics=False):
    """
    
    Butterworth IIR notch filtering. Uses filtfilt to avoid phase distortions.

    Input:
        signal (array): input signal.
        
        low_cutoff (float): low cutoff frequency.
        
        high_cutoff (float): high cutoff frequency.

        order (int): filter order.
        
        sampling_rate (float): sampling rate.
        
        harmonics (bool): Flag to also remove harmonics; the default is False.

    Output:
        output_signal (array): output FIR-filtered signal.

    Configurable fields:{"name": "preprocessing.firfilt", "config": {"order" : 300, "sampling_rate": 1000.}, "inputs": ["signal", "low_cutoff", "high_cutoff"], "outputs": ["output_signal"]}

    See Also:
        scipy.signal.filtfilt
        scipy.signal.butter
        
    Notes:
    

    Example:
    

    References:
        .. [1]
        
    """
    
    # check inputs
    if signal is None:
        raise TypeError, "An input signal is needed."
    
    if center is None:
        raise TypeError, "A center frequency is needed."
    
    if width is None:
        raise TypeError, "A width is needed."
    
    # ensure floats
    center = float(center)
    width = float(width)
    nyq = float(sampling_rate) / 2.
    
    # freqs
    lc = center - width / 2.
    if lc <= 0:
        raise ValueError, "Left notch edge is 0 or lower."
    hc = center + width / 2.
    if hc >= nyq:
        raise ValueError, "Right notch edge is Fs/2 or higher."
    
    if harmonics:
        nb = int(nyq / center)
        
        freqs = []
        for i in xrange(nb):
            freqs.append([i * center + lc, i * center + hc])
        
        freqs = numpy.array(freqs) / nyq
        
        if numpy.isclose(nyq % center, 0):
            prune = True
            nb -= 1
        else:
            prune = False
    else:
        nb = 1
        freqs = numpy.array([[lc, hc]]) / nyq
        prune = False
    
    if prune:
        # removes last item if center frequency is multiple of Nyquist frequency (harmonics)
        freqs = freqs[:-1]
    
    # filter
    output_signal = signal
    for i in xrange(nb):
        b, a = ss.butter(order, freqs[i], btype='bandstop')
        output_signal = ss.filtfilt(b, a, output_signal)
    
    return output_signal


def medFIR(signal, sampleRate=1000.):
    # ECG signal filter with two median filters (for baseline removal) and a low-pass FIR filter (40 Hz)
    
    # filter parameters
    order = int(0.3 * sampleRate)
    a1 = int(0.2 * sampleRate)
    if a1 % 2 == 0:
        a1 += 1
    a2 = int(0.6 * sampleRate)
    if a2 % 2 == 0:
        a2 += 1
    emgsamples = int(0.028 * sampleRate)
    b = numpy.ones(emgsamples, dtype='float') / emgsamples
    
    # baseline wander
    med1 = ss.medfilt(signal, a1)
    med2 = ss.medfilt(med1, a2)
    inter = signal - med2
    
    # low-pass
    inter2 = filt.filterSignal(Signal=inter, SamplingRate=sampleRate,
                                 FilterType='FIR', window='flattop', Order=order,
                                 Frequency=[40], BandType='lowpass')['Signal']
    
    # EMG noise
    filtered = filt._filterSignal(b, [1], inter2, checkPhase=True)
    
    return filtered


def PLA(signal=None, step=1, epsilon=0.1):
    # Piecewise Linear Approximation
    # from Vullings, ECG Segmentation Using Time-Warping, 1997
    
    # check inputs
    if signal is None:
        raise TypeError, "An input signal is needed."
    
    length = len(signal)
    out = []
    
    err = 0
    start = 0
    stop = step
    while stop < length:
        # increase segment
        while (err < epsilon) and (stop < length):
            # linear approximation
            a = (signal[stop] - signal[start]) / (stop - start)
            b = signal[start] - a * start
            
            # error
            aux = [(numpy.abs(signal[i] - a*i - b) / numpy.sqrt(1 + a**2)) for i in range(start, stop)]
            k = numpy.argmax(aux)
            err = aux[k]
            
            # add to segment
            stop += step
        
        # decrease segment
        while err > epsilon:
            # stop is at previous max
            stop = start + k
            
            # linear approximation
            a = (signal[stop] - signal[start]) / (stop - start)
            b = signal[start] - a * start
            
            # error
            aux = [(numpy.abs(signal[i] - a*i - b) / numpy.sqrt(1 + a**2)) for i in range(start, stop)]
            k = numpy.argmax(aux)
            err = aux[k]
        
        out.append((start, stop, a, b))
        start = stop
        stop  = start + step
    
    return out
    
    


#===============================================================================
# # filtering example
# import scipy, pylab, misc
# import scipy.signal as ss
# from misc.misc import centeredFFT  
# fs = 1000.  # sampling rate
# x = scipy.arange(-2*scipy.pi, 2*scipy.pi, 1/fs)
# y = scipy.cos(2*scipy.pi*5*x) + scipy.cos(2*scipy.pi*10*x) + scipy.cos(2*scipy.pi*45*x) + scipy.cos(2*scipy.pi*100*x)
# [b, a] = ss.butter(4, 2.*50./fs, 'low')
# y_lowf = ss.filtfilt(b, a, y)
# [b, a] = ss.butter(4, [2.*45./fs, 2.*150./fs], 'band')
# y_bandf = ss.filtfilt(b, a, y)
# fig = pylab.figure()
# fig.suptitle('FFT')
# ax = fig.add_subplot(311)
# ax.set_title('Cosine with frequencies 5, 10, 45, 100Hz')
# f,X = centeredFFT(y, fs, False, ax)
# ax = fig.add_subplot(312)
# ax.set_title('Low-pass filtered: 50Hz')
# f,X = centeredFFT(y_lowf, fs, False, ax)
# ax = fig.add_subplot(313)
# ax.set_title('Band-pass filtered: 45-150Hz')
# f,X = centeredFFT(y_bandf, fs, False, fig.add_subplot(313))
# pylab.show()
#===============================================================================
