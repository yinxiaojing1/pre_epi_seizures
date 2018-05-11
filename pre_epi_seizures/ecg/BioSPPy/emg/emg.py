"""
.. module:: emg
   :platform: Unix, Windows
   :synopsis: This module provides various functions to handle EMG signals.

.. moduleauthor:: Filipe Canento


"""
import sys
sys.path.append("../")
import plux
import sync
import peakd
import unittest
import numpy as np
import scipy as sp
import pylab as pl
import filt as flt
import tools as tls

def filt(Signal=None, SamplingRate=1000., UpperCutoff=None, LowerCutoff=100., Order=4.):
    """
    Filters an input EMG signal. 

    If only input signal is provide, it returns the filtered EMG signal 
    assuming a 1000Hz sampling frequency and a default high-pass filter 
    with a cutoff frequency of 100Hz.

    Kwargs:
        Signal (array): input signal.
        
        SamplingRate (float): Sampling frequency (Hz).
        
        UpperCutoff (float): Low-pass filter cutoff frequency (Hz).
        
        LowerCutoff (float): High-pass filter cutoff frequency (Hz).
        
        Order (int): Filter order.

    Kwrvals:
        Signal (array): output filtered signal.
        
    Configurable fields:{"name": "emg.filt", "config": {"SamplingRate": "1000.", "LowerCutoff": "100.", "Order": "4."}, "inputs": ["Signal", "UpperCutoff"], "outputs": ["Signal"]}

    See Also:
        flt.zpdfr

    Notes:
        

    Example:
        Signal = load(...)
        SamplingRate = ...
        res = filt(Signal=Signal, SamplingRate=SamplingRate)
        plot(res['Signal'])

    References:
        .. [1] 
        
    """
    # Check
    if Signal is None:
        raise TypeError, "An input signal is needed."
    # Filter signal
    Signal=flt.zpdfr(Signal=Signal, 
                    SamplingRate=SamplingRate, 
                    UpperCutoff=UpperCutoff, 
                    LowerCutoff=LowerCutoff, 
                    Order=Order)['Signal']
    # Output
    kwrvals={}
    kwrvals['Signal']=Signal

    return kwrvals

def onoff(Signal=None, SamplingRate=1000., Thres=None, ws=50.0):
    """
    EMG signal onset detection.

    Kwargs:
        Signal (array): input signal.
        
        SamplingRate (float): Sampling frequency (Hz).
        
        Thres (float): detection threshold			
        
        ws (float): detection window size in milliseconds

    Kwrvals:
        onoff (array): indexes of EMG onsets
        
    Configurable fields:{"name": "emg.onoff", "config": {"SamplingRate": "1000.", "ws": "50.0"}, "inputs": ["Signal", "Thres"], "outputs": ["onoff"]}

    See Also:
        flt.smooth

    Notes:

    Example:


    References:
        .. [1] 
        
    """
    # Check
    if Signal is None:
        raise TypeError, "An input signal is needed."
    # Full-Wave-Rectify the raw data
    fwlo = abs(Signal)
    # Window size (in samples)
    sws = SamplingRate*(0.001*ws)
    # Smooth
    mvgav = flt.smooth(Signal=fwlo, Window={'Length':sws, 'Type':'boxzen', 'Parameters':None})['Signal']
    if not Thres:
        Thres = 1.2*np.mean(abs(mvgav))+2.0*np.std(abs(mvgav))			#*review

    st = pl.find(mvgav>Thres)
    ed = pl.find(mvgav<=Thres)

    onoff = np.union1d(np.intersect1d(st-1,ed),np.intersect1d(st+1,ed))

    if(np.any(onoff)):
        if(onoff[-1]>=len(Signal)):
            onoff[-1] = -1

    # Output
    kwrvals={}
    kwrvals['onoff']=onoff/SamplingRate if SamplingRate else onoff
    kwrvals['Signal']=mvgav

    return kwrvals
    
def emg(Signal=None, SamplingRate=1000.0, Filter=()):
    """
    EMG signal processing and feature extraction.

    Kwargs:
        Signal (array): input signal.
        
        SamplingRate (float): Sampling frequency (Hz).
        
        Filter (dict): filter parameters.

    Kwrvals:
        Signal (array): output filtered signal (see notes 1).
        
        onoff (array): indexes of EMG onsets
        
    Configurable fields:{"name": "emg.emg", "config": {"SamplingRate": "1000.0"}, "inputs": ["Signal", "Filter"], "outputs": ["Signal", "onoff"]}

    See Also:
        filt
        
        onoff

    Notes:

    Example:


    References:
        .. [1] 
        
    """
    # Check
    if Signal is None:
        raise TypeError, "An input signal is needed."
    if Filter:
        Filter.update({'Signal': Signal})
        if not Filter.has_key('SamplingRate'): Filter.update({'SamplingRate': SamplingRate})
        Signal=filt(**Filter)['Signal']

    #onset detection										#*review
    res=onoff(Signal=Signal)

    # kwrvals
    kwrvals={}
    # if Filter is not None: kwrvals['Signal']=Signal
    kwrvals['Signal']=res['Signal']
    kwrvals['onoff']=res['onoff']

    # kwrvals['OnsetTime']=
    # kwrvals['DownsetTime']=

    return kwrvals	
    
def features(Signal=None, SamplingRate=1000., Filter={}):
    """
    Retrieves relevant EMG signal features.

    Kwargs:
        Signal (array): input signal.
        
        SamplingRate (float): Sampling frequency (Hz).
        
        Filter (dict): filter parameters.

    Kwrvals:
        Signal (array): output filtered signal (see notes 1).
        
        onoff (array): indexes of EMG onsets
        
        mean (float): mean
        
        std (float): standard deviation
        
        var (float): variance
        
        skew (ndarry): skewness
        
        kurtosis (array): kurtosis
        
        ad (float): absolute deviation	
        
    Configurable fields:{"name": "emg.features", "config": {"SamplingRate": "1000."}, "inputs": ["Signal", "Filter"], "outputs": ["Signal", "onoff", "mean", "std", "var", "skew", "kurtosis", "ad"]}

    See Also:
        filt
        
        onoff
        
        tls.statsf

    Notes:


    Example:


    References:
        .. [1] 
        
    """
    # Check
    if Signal is None:
        raise TypeError, "An input signal is needed."
    if Filter:
        Filter.update({'Signal': Signal})
        if not Filter.has_key('SamplingRate'): Filter.update({'SamplingRate': SamplingRate})
        Signal=filt(**Filter)['Signal']
    # Init
    kwrvals = {}
    # Feature Extraction
    res = emg(Signal=Signal, SamplingRate=SamplingRate)
    for k in res: kwrvals[k] = res[k]
    res = tls.statsf(Signal=Signal)
    for k in res: kwrvals[k] = res[k]
    # Out
    return kwrvals
	
if __name__=='__main__':

	class testemg(unittest.TestCase):
		"""
		A test class for the emg module.
			"""
		def setUp(self):
			# Init
			self.Signal = plux.loadbpf("../signals/emg.txt")
			self.Signal = self.Signal[:,3]
			self.SamplingRate = float(Signal.header['SamplingFrequency'])
			# ...
		def testfilt(self):
			# Test if a dict is returned by the filt function
			self.res = filt(Signal=self.Signal, SamplingRate=self.SamplingRate)
			assert type(self.res) is dict, "Returned value by the filt function is not a dict."
			# Test if the right exception is raised when no input signal is given
			self.assertRaises(TypeError, filt, None)
			# ...
		def testonoff(self):
			# Test if a dict is returned by the onoff function
			self.res = onoff(Signal=self.Signal, SamplingRate=self.SamplingRate)
			assert type(self.res) is dict, "Returned value by the onoff function is not a dict."
			# Test if the right exception is raised when no input signal is given
			self.assertRaises(TypeError, onoff, None)
			# ...			
		def testemg(self):
			# Test if a dict is returned by the emg function
			self.res = emg(Signal=self.Signal, SamplingRate=self.SamplingRate)
			assert type(self.res) is dict, "Returned value by the emg function is not a dict."
			# Test if the right exception is raised when no input signal is given
			self.assertRaises(TypeError, emg, None)
			# ...
		def testfeatures(self):
			# Test if a dict is returned by the features function
			self.res = features(Signal=self.Signal, SamplingRate=self.SamplingRate)
			assert type(self.res) is dict, "Returned value by the features function is not a dict."
			# Test if the right exception is raised when no input signal is given
			self.assertRaises(TypeError, features, None)
			# ...			
		# ...		
	# Example:
	# Load Data
	RawSignal = plux.loadbpf("../signals/emg.txt")
	RawSignal = RawSignal[:,3]
	SamplingRate=float(RawSignal.header['SamplingFrequency'])
	# Unit conversion
	RawSignal = RawSignal.toV()
	# Filter
	Signal = filt(Signal=RawSignal)['Signal']						#*losing bparray information
	# Convert to bparray
	Signal = plux.bparray(Signal,RawSignal.header)
	# Time array
	Time = np.linspace(0,len(Signal)/SamplingRate,len(Signal))
	# Beat information
	res = emg(Signal=Signal,SamplingRate=SamplingRate)
	# Plot
	fig=pl.figure()
	ax=fig.add_subplot(111)
	ax.plot(Time,Signal,'k')
	if(np.any(res['onoff'])): ax.vlines(res['onoff'],min(Signal),max(Signal),'r',lw=3)
	ax.set_xlabel('Time (sec)')
	ax.set_ylabel('EMG ('+Signal.header['Units']+')')
	ax.set_title("EMG onset detection")
	ax.axis('tight')
	ax.legend(('EMG','On&Off'), 'best', shadow=True)
	ax.grid('on')
	fig.show()
	# Unitest
	unittest.main()
