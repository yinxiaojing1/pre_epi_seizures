"""
.. module:: eda
   :platform: Unix, Windows
   :synopsis: This module provides various functions to handle EDA signals.

.. moduleauthor:: Filipe Canento


"""
import sys
import unittest
import numpy as np
import pylab as pl
import models as edamodels
sys.path.append("../")
import plux
import sync
import peakd
import filt as flt
import tools as tls
reload(edamodels)

def filt(Signal=None, SamplingRate=1000., UpperCutoff=0.25, Order=2):
    """
    Filters an input EDA signal. 

    If only input signal is provide, it returns the filtered EDA signal 
    assuming a 1000Hz sampling frequency and a default low-pass filter 
    with a cutoff frequency of 0.25Hz.

    Kwargs:
        Signal (array): input signal.
        
        SamplingRate (float): Sampling frequency (Hz).
        
        UpperCutoff (float): Low-pass filter cutoff frequency (Hz).
        
        LowerCutoff (float): High-pass filter cutoff frequency (Hz).
        
        Order (int): Filter order.

    Kwrvals:
        Signal (array): output filtered signal.
            
    Configurable fields:{"name": "eda.filt", "config": {"UpperCutoff": "0.25", "SamplingRate": "1000.", "Order": "2"}, "inputs": ["Signal"], "outputs": ["Signal"]}

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
    # Low pass filter
    Signal=flt.zpdfr(Signal=Signal, SamplingRate=SamplingRate, UpperCutoff=UpperCutoff, LowerCutoff=None, Order=Order)['Signal']

    # Output
    kwrvals={}
    kwrvals['Signal']=Signal

    return kwrvals
    
def scr(Signal=None, SamplingRate=1000., Method='basic', Filter={}):
    """
    Detects and extracts Skin Conductivity Responses (SCRs) information such as:
    SCRs amplitudes, onsets, peak instant, rise, and half-recovery times.

    Kwargs:
        Signal (array): input EDA signal.
        
        SamplingRate (float): Sampling frequency (Hz).
        
        Method (string): SCR detection algorithm.
        
        Filter (dict): filter parameters.

    Kwrvals:
        Signal (array): output filtered signal (see notes 1)
            
        Amplitude (array): signal pulses amplitudes (in the units of the input signal)
            
        Onset (array): indexes (or instants in seconds, see notes 2.a) of the SCRs onsets
            
        Peak (array): indexes (or instants in seconds, see notes 2.a) of the SCRs peaks	
            
        Rise (array): SCRs rise times (in seconds)
            
        HalfRecovery (array): SCRs half-recovery times (in seconds)
            
    Configurable fields:{"name": "eda.scr", "config": {"SamplingRate": "1000.", "Method": ""basic""}, "inputs": ["Signal", "Filter"], "outputs": ["Signal", "Amplitude", "Onset", "Peak", "Rise", "HalfRecovery"]}

    See Also:
        filt
        
        models.basicSCR
        
        models.KBKSCR

    Notes:
        1 - If a filter is given as a parameter, then the returned keyworded values dict has a 'Signal' key.
        
        2 - If the sampling rate is defined, then:
            a) keys 'onset', and 'peak' are converted to instants of occurrence in seconds.
        
    Example:


    References:
        .. [1] 
        
    """
    # Init
    kwrvals = {}
    # Call appropriate method
    if Method is 'basic':
        kwrvals = edamodels.basicSCR(Signal=Signal, SamplingRate=SamplingRate, Filter=Filter)
    elif Method is 'KBK':
        kwrvals = edamodels.KBKSCR(Signal=Signal, SamplingRate=SamplingRate)
    else:
        print "Method not implemented."
    # Output
    return kwrvals
   
def scl(Signal=None, SamplingRate=1000., Filter={'UpperCutoff': 0.05}):
    """


    Kwargs:


    Kwrvals:


    Configurable fields:{"name": "eda.scl", "config": {"Filter": "{"UpperCutoff":0.05}", "SamplingRate": "1000."}, "inputs": ["Signal"], "outputs": []}

    See Also:


    Notes:


    Example:


    References:
        .. [1]	
        
    """
    if Signal is None: raise TypeError("an input signal must be provided")

    if Filter:
        Filter.update({'Signal': Signal})
        if not Filter.has_key('SamplingRate'): Filter.update({'SamplingRate': SamplingRate})
        Signal=filt(**Filter)['Signal']

    kwrvals={}

    if Filter: kwrvals['Signal']=Signal
    #kwrvals['ShiftTime']=
    #kwrvals['Slope']=

    return kwrvals

def features(Signal=None, SamplingRate=1000., Filter={}):
    """
    Retrieves relevant EDA signal features.

    Kwargs:
        Signal (array): input EDA signal
        
        SamplingRate (float): sampling frequency (Hz)
        
        Filter (dict): Filter parameters

    Kwrvals:
        Signal (array): output filtered signal (see notes 1)
        
        Amplitude (array): signal pulses amplitudes (in the units of the input signal)
        
        Onset (array): indexes (or instants in seconds, see notes 2.a) of the SCRs onsets
        
        Peak (array): indexes (or instants in seconds, see notes 2.a) of the SCRs peaks	
        
        Rise (array): SCRs rise times (in seconds)
        
        HalfRecovery (array): SCRs half-recovery times (in seconds)
        
        mean (float): mean
        
        std (float): standard deviation
        
        var (float): variance
        
        skew (ndarry): skewness
        
        kurtosis (array): kurtosis
        
        ad (float): absolute deviation	

    Configurable fields:{"name": "eda.features", "config": {"SamplingRate": "1000."}, "inputs": ["Signal", "Filter"], "outputs": ["Signal", "Amplitude", "Onset", "Peak", "Rise", "HalfRecovery", "mean", "std", "var", "skew", "kurtosis", "ad"]}

    See Also:
        filt
        models.basicSCR
        tls.statsf

    Notes:
        1 - If a filter is given as a parameter, then the returned keyworded values dict has a 'Signal' key.
        
        2 - If the sampling rate is defined, then:
            a) keys 'onset', and 'peak' are converted to instants of occurrence in seconds.

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
    res = scr(Signal=Signal, SamplingRate=SamplingRate)
    for k in res: kwrvals[k] = res[k]
    res = tls.statsf(Signal=Signal)
    for k in res: kwrvals[k] = res[k]
    # Out
    return kwrvals
	
if __name__=='__main__':

	class testeda(unittest.TestCase):
		"""
		A test class for the eda module.
			"""
		def setUp(self):
			# Init
			self.Signal = plux.loadbpf("../signals/eda.txt")
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
		def testscr(self):
			# Test if a dict is returned by the scr function
			self.res = scr(Signal=self.Signal, SamplingRate=self.SamplingRate)
			assert type(self.res) is dict, "Returned value by the scr function is not a dict."
			# Test if the right exception is raised when no input signal is given
			self.assertRaises(TypeError, scr, None)
			# ...
		def testscl(self):
			# Test if a dict is returned by the scl function
			self.res = scl(Signal=self.Signal, SamplingRate=self.SamplingRate)
			assert type(self.res) is dict, "Returned value by the scl function is not a dict."
			# Test if the right exception is raised when no input signal is given
			self.assertRaises(TypeError, scl, None)
			# ...	
		def testKBKSCR(self):
			# Test if a dict is returned by the KBKSCR function
			self.res = edamodels.KBKSCR(Signal=self.Signal, SamplingRate=self.SamplingRate)
			assert type(self.res) is dict, "Returned value by the KBKSCR function is not a dict."
			# Test if the right exception is raised when no input signal is given
			self.assertRaises(TypeError, edamodels.KBKSCR, None)
			# ...	
		def testfeatures(self):
			# Test if a dict is returned by the features function
			self.res = features(Signal=self.Signal, SamplingRate=self.SamplingRate)
			assert type(self.res) is dict, "Returned value by the features function is not a dict."
			# Test if the right exception is raised when no input signal is given
			self.assertRaises(TypeError, features, None)
			# ...			
		# ...	
	# Example 1:
	# Load Data
	RawSignal = plux.loadbpf("../signals/eda.txt")
	RawSignal = RawSignal[:,3]
	SamplingRate=float(RawSignal.header['SamplingFrequency'])
	# Unit conversion
	RawSignal = RawSignal.touS()
	# Filter
	Signal = filt(Signal=RawSignal,SamplingRate=SamplingRate, UpperCutoff=0.4)['Signal']		#*losing bparray information
	# Convert to bparray
	Signal = plux.bparray(Signal,RawSignal.header)
	# Time array
	Time = np.linspace(0,len(Signal)/SamplingRate,len(Signal))
	# SCRs information
	res = scr(Signal=Signal,SamplingRate=SamplingRate)
	# Plot
	fig=pl.figure()
	ax=fig.add_subplot(111)
	ax.plot(Time,Signal,'k')
	ax.vlines(res['Onset'], Signal.min(), Signal.max(),'g', lw=3)	
	ax.legend(('EDA','Onset'), 'best', shadow=True)
	ax.set_xlabel('Time (sec)')
	ax.set_ylabel('EDA ('+Signal.header['Units']+')')
	ax.set_title("Detection of SCRs.")
	ax.grid('on')
	fig.show()	
	#---------
	# Unitest
	unittest.main()	
