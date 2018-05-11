"""
.. module:: ecg
   :platform: Unix, Windows
   :synopsis: This module provides various functions to handle ECG signals.

.. moduleauthor:: Filipe Canento


"""
import unittest
import scipy
import pylab
import models as ecgmodels
import plux
import filt as flt
import tools as tls

def filt(Signal=None, SamplingRate=1000., UpperCutoff=16., LowerCutoff=8., Order=4):
    """

    Filters an input ECG signal. 

    By default, the return is the filtered Signal assuming a 
    1000Hz sampling frequency and the following filter sequence:
        1. 4th order low-pass filter with cutoff frequency of 16Hz;
        2. 4th order high-pass filter with cutoff frequency of 8Hz;
        3. d[]/dt;
        4. 80ms Hamming Window Smooth.

    Kwargs:
        Signal (array): input signal.
            
        SamplingRate (float): Sampling frequency (Hz).
            
        UpperCutoff (float): Low-pass filter cutoff frequency (Hz).
            
        LowerCutoff (float): High-pass filter cutoff frequency (Hz).
            
        Order (int): Filter order.
            
    Kwrvals:
        Signal (array): output filtered signal.
            
    Configurable fields:{"name": "ecg.filt", "config": {"UpperCutoff": "16.", "SamplingRate": "1000.", "LowerCutoff": "8.", "Order": "4"}, "inputs": ["Signal"], "outputs": ["Signal"]}

    See Also:
        flt.zpdfr
        
        flt.smooth

    Notes:

    Example:
        Signal = load(...)
        
        SamplingRate = ...
        
        res = filt(Signal=Signal, SamplingRate=SamplingRate)
        
        plot(res['Signal'])

    References:
        .. [1] P.S. Hamilton, Open Source ECG Analysis Software Documentation, E.P.Limited
        http://www.eplimited.com/osea13.pdf
        
    """
    # Check
    if Signal is None:
        raise TypeError, "An input signal is needed."
    # Filter signal
    Signal = flt.zpdfr(Signal=Signal, 
                       SamplingRate=SamplingRate, 
                       UpperCutoff=UpperCutoff,
                       LowerCutoff=LowerCutoff,
                       Order=Order)['Signal']
    # d[]/dt
    Signal = abs(scipy.diff(Signal,1)*SamplingRate)
    # Smooth
    Signal = flt.smooth(Signal=Signal, Window={'Length':0.08*SamplingRate, 'Type':'hamming', 'Parameters':None})['Signal']
    # Signal=Signal[0.1*SamplingRate:]                            
    # Output
    kwrvals = {}
    kwrvals['Signal'] = Signal
    
    return kwrvals

def ecg(Signal=None, SamplingRate=1000., Filter={}):
    """

    Determine ECG signal information.

    Kwargs:
        Signal (array): input ECG signal.
        
        SamplingRate (float): Sampling frequency (Hz).
        
        Filter (dict): Filter parameters.        

    Kwrvals:
        R (array): heart beat indexes (or instants in seconds if sampling rate is defined).        

    Configurable fields:{"name": "ecg.ecg", "config": {"SamplingRate": "1000."}, "inputs": ["Signal", "Filter"], "outputs": ["R"]}

    See Also:
        filt
        models.hamilton

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
    #Detection Rules
    R = ecgmodels.hamilton(Signal=Signal, SamplingRate=SamplingRate)['R']

    kwrvals = {}
    kwrvals['R'] = R

    # kwrvals['PTime']=
    # kwrvals['QTime']=
    # kwrvals['RTime']=
    # kwrvals['STime']=
    # kwrvals['TTime']=

    # kwrvals['PAmplitude']=
    # kwrvals['QAmplitude']=
    # kwrvals['RAmplitude']=
    # kwrvals['SAmplitude']=
    # kwrvals['TAmplitude']=

    # kwrvals['IBI']=
    # kwrvals['HR']=

    return kwrvals

def features(Signal=None, SamplingRate=1000., Filter={}):
    """

    Retrieves relevant ECG signal features.

    Kwargs:
        Signal (array): input ECG signal.
        
        SamplingRate (float): Sampling frequency (Hz).
        
        Filter (dict): Filter parameters.    

    Kwrvals:
        R (array): ECG R-peak indexes (or instants in seconds if sampling rate is defined)
        
        mean (float): mean
        
        std (float): standard deviation
        
        var (float): variance
        
        skew (ndarry): skewness
        
        kurtosis (array): kurtosis
        
        ad (float): absolute deviation

    Configurable fields:{"name": "ecg.features", "config": {"SamplingRate": "1000."}, "inputs": ["Signal", "Filter"], "outputs": ["R", "mean", "std", "var", "skew", "kurtosis", "ad"]}

    See Also:
        filt
        models.hamilton
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
    res = ecg(Signal=Signal, SamplingRate=SamplingRate)
    for k in res:
        kwrvals[k] = res[k]
    res = tls.statsf(Signal=Signal)
    for k in res:
        kwrvals[k] = res[k]
    # Out
    return kwrvals

if __name__=='__main__':

    class testecg(unittest.TestCase):
        """
        A test class for the ecg module.
            """
        def setUp(self):
            # Init
            self.Signal = plux.loadbpf("../signals/ecg.txt")
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
        def testbeats(self):
            # Test if a dict is returned by the hamilton function
            self.res = ecgmodels.hamilton(Signal=self.Signal, SamplingRate=self.SamplingRate)
            assert type(self.res) is dict, "Returned value by the hamilton function is not a dict."
            # Test if the right exception is raised when no input signal is given
            self.assertRaises(TypeError, ecgmodels.hamilton, None)
            # ...
        def testecg(self):
            # Test if a dict is returned by the ecg function
            self.res = ecg(Signal=self.Signal, SamplingRate=self.SamplingRate)
            assert type(self.res) is dict, "Returned value by the ecg function is not a dict."
            # Test if the right exception is raised when no input signal is given
            self.assertRaises(TypeError, ecg, None)
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
    RawSignal = plux.loadbpf("../signals/ecg.txt")
    RawSignal = RawSignal[:,3]
    SamplingRate=float(RawSignal.header['SamplingFrequency'])
    # Unit conversion
    RawSignal = RawSignal.tomV()
    # Filter
    Signal = filt(Signal=RawSignal,SamplingRate=SamplingRate, UpperCutoff=30.0, LowerCutoff=0.5)['Signal']        #*losing bparray information
    # Convert to bparray
    Signal = plux.bparray(Signal,RawSignal.header)
    # Time array
    Time = scipy.linspace(0,len(Signal)/SamplingRate,len(Signal))
    # Beat information
    res = ecg(Signal=Signal,SamplingRate=SamplingRate)
    # Plot
    fig=pylab.figure()
    ax=fig.add_subplot(111)
    ax.plot(Time,Signal,'k')
    ax.vlines(Time[res['R']],min(Signal),max(Signal),'r', lw=3)
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('ECG ('+Signal.header['Units']+')')
    ax.set_title("R-peak detection")
    ax.axis('tight')
    ax.legend(('ECG','R-peak'), 'best', shadow=True)
    ax.grid('on')
    # fig.show()
    pylab.savefig('../temp/fig1.png')
    # Unitest
    unittest.main()
