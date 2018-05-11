"""
.. module:: resp
   :platform: Unix, Windows
   :synopsis: This module provides various functions to handle RESP signals.

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

def filt(Signal=None, SamplingRate=1000., UpperCutoff=0.35, LowerCutoff=0.1, Order=2.):
    """
    Filters an input RESP signal. 
        
    If only input signal is provide, it returns the filtered RESP signal 
    assuming a 1000Hz sampling frequency and ...

    Kwargs:
        Signal (array): input signal.
        
        SamplingRate (float): Sampling frequency (Hz).
        
        UpperCutoff (float): Low-pass filter cutoff frequency (Hz).
        
        LowerCutoff (float): High-pass filter cutoff frequency (Hz).
        
        Order (int): Filter order.

    Kwrvals:
        Signal (array): output filtered signal.

    Configurable fields:{"name": "resp.filt", "config": {"UpperCutoff": "0.35", "SamplingRate": "1000.", "LowerCutoff": "0.1", "Order": "2."}, "inputs": ["Signal"], "outputs": ["Signal"]}

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
    
def resp(Signal=None, SamplingRate=1000.0, Filter=()):
    """
    Respiratory signal information.

    Kwargs:
        Signal (array): input signal.
        
        SamplingRate (float): Sampling frequency (Hz).
        
        Filter (dict): filter parameters.
        
    Kwrvals:
        Signal (array): output filtered signal.
        
        FR (array): instant respiratory frequency (Hz)
        
        ZC (array): zero crossings indexes        

    Configurable fields:{"name": "resp.resp", "config": {"SamplingRate": "1000.0"}, "inputs": ["Signal", "Filter"], "outputs": ["Signal", "FR", "ZC"]}

    See Also:
        filt

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
    #Compute zero crossings
    zc = tls.zerocross(Signal=Signal)['ZC']

    t=np.linspace(0,len(Signal)/SamplingRate, len(Signal))					#*review
    fr = 1.0/(np.diff(t[zc][::2]))
    fr = fr[np.where(fr<=0.35)[0]]

    # kwrvals
    kwrvals={}
    if Filter: kwrvals['Signal']=Signal
    if np.any(fr): 
        kwrvals['FR']=fr
    kwrvals['ZC']=zc
    return kwrvals			
    
def features(Signal=None, SamplingRate=1000., Filter={}):
    """
    Retrieves relevant RESP signal features.

    Kwargs:
        Signal (array): input signal.
        
        SamplingRate (float): Sampling frequency (Hz).
        
        Filter (dict): filter parameters.
        
    Kwrvals:
        Signal (array): output filtered signal.
        
        FR (array): instant respiratory frequency (Hz)
        
        ZC (array): zero crossings indexes
        
        mean (float): mean
        
        std (float): standard deviation
        
        var (float): variance
        
        skew (ndarry): skewness
        
        kurtosis (array): kurtosis
        
        ad (float): absolute deviation	        

    Configurable fields:{"name": "resp.features", "config": {"SamplingRate": "1000."}, "inputs": ["Signal", "Filter"], "outputs": ["Signal", "FR", "ZC", "mean", "std", "var", "skew", "kurtosis", "ad"]}

    See Also:
        filt
        
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
    res = resp(Signal=Signal, SamplingRate=SamplingRate)
    for k in res: kwrvals[k] = res[k]
    res = tls.statsf(Signal=Signal)
    for k in res: kwrvals[k] = res[k]
    # Out
    return kwrvals
	
if __name__=='__main__':

    class testresp(unittest.TestCase):
        """
        A test class for the resp module.
            """
        def setUp(self):
            # Init
            self.Signal = plux.loadbpf("../signals/resp.txt")
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
        def testresp(self):
            # Test if a dict is returned by the resp function
            self.res = resp(Signal=self.Signal, SamplingRate=self.SamplingRate)
            assert type(self.res) is dict, "Returned value by the resp function is not a dict."
            # Test if the right exception is raised when no input signal is given
            self.assertRaises(TypeError, resp, None)
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
    RawSignal = plux.loadbpf("../signals/resp.txt")
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
    res = resp(Signal=Signal,SamplingRate=SamplingRate)
    # Plot
    fig=pl.figure()
    ax=fig.add_subplot(111)
    ax.plot(Time,Signal,'k')
    ax.plot(Time[res['ZC']],Signal[res['ZC']],'ro')
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('RESP ('+Signal.header['Units']+')')
    ax.legend(('RESP','ZC'), 'best', shadow=True)
    cmd="Resp. Rate Mean: "+(60*round(np.mean(res['FR']),2)).__str__()+" resp/min"
    ax.set_title(cmd)
    ax.axis('tight')
    ax.grid('on')
    fig.show()
    # Unitest
    unittest.main()