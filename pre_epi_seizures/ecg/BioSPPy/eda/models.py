import sys
import unittest
import numpy as np
import pylab as pl
sys.path.append("../")
import plux
import sync
import peakd
import filt as flt
import tools as tls
import eda

def gamboa(to, tp, yo, yp):
	"""

	Kwargs:


	Kwrvals:

			
	See Also:


	Notes:

		
	Example:


	References:
		.. [1] 
		
	"""
	t=np.linspace(0,(tp-to)*5)

	b=4.0/float(tp-to)
	a=(yo-yp)*(b**3)/(16*np.exp(-2)+432*np.exp(-6))

	kwrvals={}

	kwrvals['Event']=a*np.exp(-b*t)*t**4

	return kwrvals
	
def basicSCR(Signal=None, SamplingRate=1000., Filter={}):
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
            
    See Also:
        filt

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
        raise TypeError("an input signal must be provided")
    if Filter:
        Filter.update({'Signal': Signal})
        if not Filter.has_key('SamplingRate'): Filter.update({'SamplingRate': SamplingRate})
        Signal=eda.filt(**Filter)['Signal']
    try:
        SamplingRate = float(SamplingRate)
        # Compute 1st order derivative
        ds=Signal#np.diff(Signal)
        # Determine maximums and minimuns
        pi=peakd.sgndiff(ds)['Peak'] #max of ds
        ni=peakd.sgndiff(-ds)['Peak'] #min of ds
        # Pair vectors
        if(len(pi)!=0 and len(ni)!=0): (pi,ni)=sync.pair(pi,ni)
        li=min(len(pi), len(ni))
        i1=pi[:li]
        i3=ni[:li]
        # Indexes
        i0=i1-(i3-i1)/2.
        if (i0[0]<0): i0[0]=0
        i2=(i1+i3)/2.

        # Amplitude
        a=np.array(map(lambda i: max(Signal[i1[i]:i3[i]]), np.arange(0, li)))
		
		# Times
        rt=(i2-i0)
        hdt=(i3-i0)
        # Gamboa model
        #	for i in range(0, li):
        #		scr=gamboa(i1[i]*dt, i3[i]*dt, ds[i1[i]], ds[i3[i]])

        #scr amplitude (uS),rise time (s), 1/2 decay time (s)
        #        a[i]=np.max(scr)
        # Determine t0-t3 if sampling frequency is provided
        # if SamplingRate is not None: dt=1/SamplingRate;i0*=dt;i1*=dt;i2*=dt;i3*=dt;rt*=dt;hdt*=dt # deviates indexes from real position
        # kwrvals
        kwrvals={}
        if Filter is not None: kwrvals['Signal']=Signal
        kwrvals['Amplitude']=a
        kwrvals['Onset']=i3
        kwrvals['Peak']=i1
    except Exception as e:
        kwrvals = {'Amplitude': [], 'Onset': [], 'Peak': []}

    return kwrvals

def KBKSCR(Signal=None, SamplingRate=1000.):
	"""
	Detects and extracts Skin Conductivity Responses (SCRs) information such as:
	SCRs amplitudes, onsets, peak instant, rise, and half-recovery times.

	Kwargs:
		Signal (array): input EDA signal.
		
		SamplingRate (float): Sampling frequency (Hz).

	Kwrvals:
		Signal (array): output filtered signal (see notes 1)
			
		Amplitude (array): signal pulses amplitudes (in the units of the input signal)
			
		Onset (array): indexes (or instants in seconds, see notes 2.a) of the SCRs onsets
			
		Peak (array): indexes (or instants in seconds, see notes 2.a) of the SCRs peaks	
			
		TODO: Rise (array): SCRs rise times (in seconds)
			
		TODO: HalfRecovery (array): SCRs half-recovery times (in seconds)
			
	See Also:
		flt.zpdfr 

	Notes:
		1 - If the sampling rate is defined, then:
			a) keys 'onset', and 'peak' are converted to instants of occurrence in seconds.
			
		2- Less sensitive than Gamboa algorithm, but does not solve the overlapping SCRs problem.
		
	Example:


	References:
		.. [1]	K.H. Kim, S.W. Bang, and S.R. Kim
				"Emotion recognition system using short-term monitoring of physiological signals"
				Med. Biol. Eng. Comput., 2004, 42, 419-427
		
	"""
	# Check
	if Signal is None:
		raise TypeError, "An input signal is needed."
	SamplingRate = float(SamplingRate)
	# Low-pass filter and Downsampling
	Order = 4
	UpperCutoff = 20.
	Signal = flt.zpdfr(Signal=Signal,
						SamplingRate=SamplingRate,
						UpperCutoff=UpperCutoff,
						LowerCutoff=None,
						Order=Order)['Signal']
	k = int(SamplingRate/UpperCutoff)
	Signal = Signal[::k]
	# Differentiation
	ResSignal = np.diff(Signal,1)
	# Smoothing Convolution with Bartlett (20)
	ResSignal = flt.smooth(Signal=ResSignal, Window={'Length':20, 'Type':'bartlett', 'Parameters':None})['Signal']
	# Double Thresholding
	zc=tls.zerocross(Signal=ResSignal)['ZC']
	if(np.all(ResSignal[:zc[0]]>0)): zc=zc[1:]
	if(np.all(ResSignal[zc[-1]:]>0)): zc=zc[:-1]
	# Exclude SCRs with an amplitude smaller than 10% of the maximum
	thres = 0.1*np.max(ResSignal)
	scrs,amps,ZC,pks=[],[], [],[]
	for i in range(0,len(zc)-1,2):
		scrs += [ResSignal[zc[i]:zc[i+1]]]
		aux=scrs[-1].max()
		if(aux>thres):
			amps += [aux]
			ZC += [zc[i]]
			ZC += [zc[i+1]]
			pks += [zc[i]+np.argmax(ResSignal[zc[i]:zc[i+1]])]
	scrs,amps,ZC,pks=np.array(scrs),np.array(amps),np.array(ZC),np.array(pks)
	if SamplingRate: dt=1/UpperCutoff;ZC*=dt;pks*=dt
	# kwrvals
	kwrvals={}
	kwrvals['Signal']=Signal
	kwrvals['Amplitude']=amps
	kwrvals['Onset']=ZC[::2]
	# kwrvals['HalfRise']=
	kwrvals['Peak']=pks
	# kwrvals['HalfRecovery']=
	# kwrvals['RiseTime']=
	# kwrvals['HalfDecayTime']=

	return kwrvals	

