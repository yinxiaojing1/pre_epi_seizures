"""
.. module:: bvp
   :platform: Unix, Windows
   :synopsis: This module provides various functions to handle BVP signals.

.. moduleauthor:: Filipe Canento


"""
import os
import sys
import glob
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


def onset(Signal=None, SamplingRate=1000.):
    """

    Determines the onsets of the BVP signal pulses. 
    Skips very corrupted signal parts.

    Kwargs:
        Signal (array): input signal.
        
        SamplingRate (float): sampling frequency (Hz).

    Kwrvals:
        Onset (array): 

    Configurable fields:{"name": "bvp.onset", "config": {"SamplingRate": "1000."}, "inputs": ["Signal"], "outputs": ["Onset"]}

    See Also:


    Notes:


    Example:


    References:
        .. [1] 
        
    """
    # Check
    if Signal is None:
        raise TypeError, "An input signal is needed."
    # Init
    idx, GO, start = [], True, 0
    window_size = 5 # Analyze window_size seconds of signal
    Nerror=[]
    while(GO):
        try:
            Signal_part = Signal[start:start+window_size*int(SamplingRate)]
        except IndexError:
            Signal_part = Signal[start:-1]
            GO = False
        # Break if remaining signal length is less than 1 second
        if(len(Signal_part)<1*SamplingRate): break
        # Compute SSF
        q = peakd.ssf(Signal=Signal_part)
        sq = q['Signal']-np.mean(q['Signal'])
        ss = q['SSF']*25
        sq = sq[1:]
        # pidx = sq>ss
        # pidx = pidx.astype('int')
        # dpidx = np.diff(pidx)
        # dpidx[dpidx<0]=0
        # dpidx = np.where(dpidx!=0)[0]
        # dpidx +=1
        # above code only does not work when there're small fluctuations of the signal
        sss = (np.diff(ss))*100
        sss[sss<0] = 0
        sss = sss-2.0*np.mean(sss) #eliminates small variations
        pk = peakd.sgndiff(sss)['Peak']
        pk = pk[pl.find(sss[pk]>0)]
        pk += 100
        dpidx = pk
        # Analyze signal between maximums of 2nd derivative of ss +100 samples (dpidx indexes)
        detected=False
        for i in range(1,len(dpidx)+1):
            try:
                st,end = dpidx[i-1],dpidx[i]
            except IndexError:
                st,end = dpidx[-1], -1
            # Error estimation
            # try:
                # Ne = MAR(filt(Signal_part[1:][st:end])['Signal'])['Ne']
            # except ValueError:
                # Ne = 0
            Ne = 0
            # Skip if error is too big, i.e, signal is too corrupted
            Nerror+=[abs(np.mean(Ne))]
            if(abs(np.mean(Ne))>1e-1):	# empirical value, REVIEW: has to depend on maximum amplitude
                continue
            s = sq[st:end]
            M = peakd.sgndiff(s)['Peak']
            m = peakd.sgndiff(-s)['Peak']
            try:
                M = M[np.argmax(s[M])] # get max index
                m = m[np.argmin(s[m])] # get min index
            except ValueError:
                continue
            if(s[M]-s[m]>0 and m-M>150):#: and m-M < 2000 and m-M>100): # maximum has to be larger than minimum
                                                        # interval between maximum and minimum bounds
                idx += [st+start]
                detected=True
        # Next round continues from previous detected beat + 100 samples to avoid double detections
        if(detected):
            start = idx[-1]+100
        # if no beat was detected, it moves window_size seconds forward
        else:
            start += window_size*int(SamplingRate) 
        # print start,
        # if(raw_input('>')=='q'): break
    idx = np.array(idx)
    # Heart Rate upper and lower bounds
    hr = SamplingRate*(60.0/(np.diff(idx)))
    idx = idx[np.intersect1d(pl.find(hr>30),pl.find(hr<200))]

    # pl.figure(707)
    # pl.figure(707).clf()
    # # pl.plot(Signal)
    # pl.plot(sq,'b')
    # # pl.plot(ss,'g')
    # # pl.plot(sss,'k')
    # # pl.plot(pk, sss[pk],'k.')
    # if(np.any(idx)): pl.vlines(idx, -0.05,0.05,'r')
    # pl.grid('on')
    # pl.show()

    # kwrvals
    kwrvals = {}
    kwrvals['Onset'] = idx/SamplingRate if SamplingRate else idx
    kwrvals['Ne'] = np.array(Nerror)
    kwrvals['Signal'] = filt(Signal[1:])['Signal']

    return kwrvals
    
def filt(Signal=None, SamplingRate=1000., UpperCutoff=8., LowerCutoff=1., Order=4.):
    """

    Filters an input BVP signal.

    If only input signal is provide, it returns the filtered
    signal assuming a 1000Hz sampling frequency and the default 
    filter parameters: low-pass filter with cutoff frequency of 8Hz 
    followed by a high-pass filter with cutoff frequency of 1Hz.

    Kwargs:
        Signal (array): input signal.
        
        SamplingRate (float): sampling frequency (Hz).
        
        UpperCutoff (float):  Low-pass filter cutoff frequency (Hz).
        
        LowerCutoff (float): High-pass filter cutoff frequency (Hz).
        
        Order (int): Filter order.

    Kwrvals:
        Signal (array): output filtered signal.
        
    Configurable fields:{"name": "bvp.filt", "config": {"UpperCutoff": "8.", "SamplingRate": "1000.", "LowerCutoff": "1.", "Order": "4."}, "inputs": ["Signal"], "outputs": ["Signal"]}

    See Also:
        flt.zpdfr

    Notes:


    Example:


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

def pulse(Signal=None, SamplingRate=1000., Filter={}):
    """

    Determines BVP signal pulse information.

    Kwargs:
        Signal (array): input signal.
        
        SamplingRate (float): sampling frequency (Hz).
        
        Filter (dict):  filter parameters.

    Kwrvals:
        Signal (array): output filtered signal (see notes 1). 
        
        Amplitude (array): signal pulses amplitudes (in the units of the input signal). (TODO)
            
        Onset (array): indexes (or instants in seconds, see notes 2.b) of the pulses onsets. 
            
        Peak (array): indexes (or instants in seconds, see notes 2.b) of the pulses peaks. (TODO)
        
        DicroticNotch (array): indexes (or instants in seconds, see notes 2.b) of the pulses dicrotic notchs. (TODO)
        
        IBI (array): Inter-Beat Intervals in msec (see notes 2.a).
        
        HR (array): Instantaneous Heart Rates in b.p.m. (see notes 2.a).        
        
    Configurable fields:{"name": "bvp.pulse", "config": {"SamplingRate": "1000."}, "inputs": ["Signal", "Filter"], "outputs": ["Signal", "Amplitude", "Onset", "Peak", "DicroticNotch", "IBI", "HR"]}

    See Also:
        filt
        
        onset

    Notes:
        1 - If a filter is given as a parameter, then the returned keyworded values dict has a 'Signal' key.
        
        2 - If the sampling rate is defined, then:
            a) the returned keyworded values dict has keys 'IBI' and 'HR'.
            b) TODO: keys 'onset', 'peak', and 'DicroticNotch' are converted to instants of occurrence in seconds.    

    Example:
        bvp = ...
        SamplingRate = ...
        res = pulse(Signal=bvp, SamplingRate=SamplingRate)    

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
    SamplingRate=float(SamplingRate)
    res = onset(Signal=Signal, SamplingRate=SamplingRate)
    # Determine amplitudes
    # Signal = (Signal-min(Signal))/(max(Signal)-min(Signal))
    # total_mxs=peakd.sgndiff(Signal)['Peak']
    # total_mns=peakd.sgndiff(-Signal)['Peak']
    # mxs=total_mxs[Signal[total_mxs]>0.8]
    # mns=total_mns[Signal[total_mns]<0.2]
    # mns, mxs=sync.pair(mns,mxs)
    # amp=Signal[mxs]-Signal[mns]
    # # Dicrotic Notch
    # dn=[]
    # aux = total_mns[Signal[total_mns]>0.2]
    # for i in range(0,len(mns)-1):
        # aux2=aux[aux>mns[i]]
        # aux2=aux2[aux2<mns[i+1]]
        # dn+=[aux2[np.argmin(Signal[aux2])]]
    # kwrvals
    kwrvals={}
    # if Filter is not None: kwrvals['Signal']=Signal
    # kwrvals['Amplitude']=amp
    kwrvals['Signal']=res['Signal']
    kwrvals['Onset']=res['Onset']# kwrvals['Onset']=mns
    # kwrvals['Peak']=mxs
    # kwrvals['DicroticNotch']=dn
    if SamplingRate:
        kwrvals['HR']=(60.0/(np.diff(res['Onset']))) # Heart Rate array in bpm
    # kwrvals['Recovery']=
    # kwrvals['RiseTime']=

    return kwrvals
    
def features(Signal=None, SamplingRate=1000., Filter={}):
    """

    Retrieves relevant BVP signal features.

    Kwargs:
        Signal (array): input signal.
        
        SamplingRate (float): sampling frequency (Hz).
        
        Filter (dict):  filter parameters.

    Kwrvals:
        Signal (array): output filtered signal (see notes 1). 
        
        Amplitude (array): signal pulses amplitudes (in the units of the input signal).(TODO) 
            
        Onset (array): indexes (or instants in seconds, see notes 2.b) of the pulses onsets. 
            
        Peak (array): indexes (or instants in seconds, see notes 2.b) of the pulses peaks. (TODO)
        
        DicroticNotch (array): indexes (or instants in seconds, see notes 2.b) of the pulses dicrotic notchs. (TODO)
        
        IBI (array): Inter-Beat Intervals in msec (see notes 2.a).
        
        HR (array): Instantaneous Heart Rates in b.p.m. (see notes 2.a).
        
        mean (float): mean
        
        std (float): standard deviation
        
        var (float): variance
        
        skew (ndarry): skewness
        
        kurtosis (array): kurtosis
        
        ad (float): absolute deviation
        
    Configurable fields:{"name": "bvp.features", "config": {"SamplingRate": "1000."}, "inputs": ["Signal", "Filter"], "outputs": ["Signal", "Amplitude", "Onset", "Peak", "DicroticNotch", "IBI", "HR", "mean", "std", "var", "skew", "kurtosis", "ad"]}

    See Also:
            filt
            
            pulse
            
            tls.statsf

    Notes:
        1 - If a filter is given as a parameter, then the returned keyworded values dict has a 'Signal' key.
        
        2 - If the sampling rate is defined, then:
            a) the returned keyworded values dict has keys 'IBI' and 'HR'.
            b) keys 'onset', 'peak', and 'DicroticNotch' are converted to instants of occurrence in seconds.   

    Example:
        bvp = ...
        SamplingRate = ...
        res = pulse(Signal=bvp, SamplingRate=SamplingRate)    

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
    res = pulse(Signal=Signal, SamplingRate=SamplingRate)
    for k in res: kwrvals[k] = res[k]
    res = tls.statsf(Signal=Signal)
    for k in res: kwrvals[k] = res[k]
    # Out
    return kwrvals
    
if __name__=='__main__':

	class testbvp(unittest.TestCase):
		"""
		A test class for the bvp module.
			"""
		def setUp(self):
			# Init
			self.Signal = plux.loadbpf("../signals/bvp.txt")
			self.Signal = self.Signal[:,3]
			self.SamplingRate = float(Signal.header['SamplingFrequency'])
			self.DataFolder = r'../signals/bvp'
			self.Benchmark = 0
			# ...
		def testfilt(self):
			# Test if a dict is returned by the filt function
			self.res = filt(Signal=self.Signal, SamplingRate=self.SamplingRate)
			assert type(self.res) is dict, "Returned value by the filt function is not a dict."
			# Test if the right exception is raised when no input signal is given
			self.assertRaises(TypeError, filt, None)
			# ...
		def testpulse(self):
			# Test if a dict is returned by the pulse function
			self.res = pulse(Signal=self.Signal, SamplingRate=self.SamplingRate)
			assert type(self.res) is dict, "Returned value by the pulse function is not a dict."
			# Test if the right exception is raised when no input signal is given
			self.assertRaises(TypeError, pulse, None)
			# ...
		def testfeatures(self):
			# Test if a dict is returned by the features function
			self.res = features(Signal=self.Signal, SamplingRate=self.SamplingRate)
			assert type(self.res) is dict, "Returned value by the features function is not a dict."
			# Test if the right exception is raised when no input signal is given
			self.assertRaises(TypeError, features, None)
			# ...			
		# def testpulseBENCHMARK(self):
		# # Test if the number of detected beats is correct by comparison with known value.
			# for file in glob.glob(os.path.join(self.DataFolder, '*.txt')):
				# self.Signal = plux.loadbpf(file)
				# self.Signal = self.Signal[:,3]
				# self.SamplingRate = float(Signal.header['SamplingFrequency'])
				# self.res = filt(Signal=self.Signal, SamplingRate=self.SamplingRate)
				# self.res = pulse(Signal=self.res['Signal'], SamplingRate=self.SamplingRate)
				# self.NrBeats = 
				# if(len(self.res['Onset']) == self.NrBeats):
					# self.Benchmark+=1
				# # ...
			# ...
		# ...	
	# Example:
	# Load Data
	RawSignal = plux.loadbpf("../signals/bvp.txt")
	RawSignal = RawSignal[:,3]
	SamplingRate=float(RawSignal.header['SamplingFrequency'])
	# Unit conversion
	RawSignal = RawSignal.tomV()
	# Filter
	Signal = filt(Signal=RawSignal,SamplingRate=SamplingRate)['Signal']		#*losing bparray information
	# Convert to bparray
	Signal = plux.bparray(Signal,RawSignal.header)
	# Normalization: 0-1
	Signal = (Signal-min(Signal))/(max(Signal)-min(Signal))
	# Time array
	Time = np.linspace(0,len(Signal)/SamplingRate,len(Signal))
	# Pulse information
	res = pulse(Signal=Signal,SamplingRate=SamplingRate)
	# Plot
	fig=pl.figure()
	ax=fig.add_subplot(111)
	ax.plot(Time,Signal,'b')
	# ax.plot(Time[res['Peak']], Signal[res['Peak']],'g^')
	ax.vlines(res['Onset'], Signal.min(), Signal.max(),'r', lw=3)	
	ax.legend(('BVP','Onset'), 'best', shadow=True)
	ax.set_xlabel('Time (sec)')
	ax.set_ylabel('BVP (normalized)')
	ax.axis([Time[0],Time[-1],-0.1,1.1])
	ax.set_title("Example of a BVP signal and features. HR = %s bpm" % (round(np.mean(res['HR']),0)))
	ax.grid('on')
	fig.show()
	# Unitest
	# unittest.main()		
