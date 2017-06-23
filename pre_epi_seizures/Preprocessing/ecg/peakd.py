"""
	This module provides various functions to ...

	Functions:

	sgndiff()
	ssf()

	"""
import numpy as np
import scipy as sp
import filt as flt
import pylab as pl

#----------------------------------------------------------------------------------------------------------------------------
# Determines Signal peaks.
def sgndiff(Signal=None):
	"""
	Determines Signal peaks.

	Kwargs:
		Signal (array): input signal

	Kwrvals:
		Peak (array): peak indexes


	See Also:


	Notes:
		

	Example:


	References:
		.. [1] 
		
	"""
	# Check
	if Signal is None:
		raise TypeError("An input signal is needed.")
	# kwrvals
	kwrvals={}
	#kwrvals['Peak']=np.array(sp.where(sp.diff(sp.sign(sp.diff(Signal)))==-2))[0,:]+1
	kwrvals['Peak']=np.array(pl.find(sp.diff(sp.sign(sp.diff(Signal)))==-2))+2

	return kwrvals
#----------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------
# Slope Sum Function
def ssf(Signal=None, SamplingRate=1000., Filter={}):
	"""
	Determines Signal peaks.

	Kwargs:
		Signal (array): input signal
		
		SamplingRate (float): Sampling frequency (Hz)
		
		Filter (dict): Filter coefficients

	Kwrvals:
		Signal (array):
		
		Onset (array): 
		
		SSF (array):
		

	See Also:


	Notes:
		

	Example:


	References:
		.. [1] W.Zong, T.Heldt, G.B. Moody, and R.G. Mark,
				"An Open-source Algorithm to Detect Onset of Arterial Blood Pressure Pulses", 
				Computers in Cardiology 2003; 30:259-262
		
	"""
	# Check
	if Signal is None:
		raise TypeError("An input signal is needed.")
	# Low-pass filter
	if Filter:
		Filter.update({'Signal': Signal})
		if not Filter.has_key('SamplingRate'): Filter.update({'SamplingRate': SamplingRate})
		y=flt.zpdfr(**Filter)['Signal']
	else:
		y=flt.zpdfr(Signal=Signal, SamplingRate=SamplingRate, UpperCutoff=4., LowerCutoff=None, Order=2.)['Signal']
	# Slope Sum Function
	dy = np.diff(y)
	du = dy
	du[du<0] = 0
	win = 0.250*SamplingRate 						# original: 128 ms window
	ssf = flt.smooth(du,Window={'Length':win, 'Type':'boxcar','Parameters':None})['Signal']
	# Decision rule
	thres = 0.6*3.*np.mean(ssf[:10*SamplingRate])	# a threshold base value is established
													# and is initialized at three times the mean SSF signal 
													# (averaged over the first ten seconds of the recording). 
													# actual threshold is taken to be 60% of the threshold base value
	win = np.int(0.150*SamplingRate) 				# 150 ms threshold window
	refp = 0.300*SamplingRate						# eyeclosing (refractory) period, 300ms -> max 200 bpm
	value, slide, onset, dmm = 0, 0, [], []
	# while(slide<len(ssf)):
		# try:
			# i = np.where(ssf[slide:]>thres)[0][0]	# SSF signal crosses the threshold
		# except IndexError:	# no i
			# break
		# prec = i-win
		# prec = prec if prec>0 else 0
		# min = np.min(ssf[prec:i+1])
		# try:
			# max = np.max(ssf[i:i+win+1])
		# except IndexError:	# i+win+1 > len(ssf)
			# break
		# if(max-min > value): 						# accept pulse detection criterion
			# dmm+=[max-min]
			# thres = 0.6*max
			# onset += [slide+np.where(ssf[slide:]>0.01*max)[0][0]]
			# slide = onset[-1]
		# slide += refp	# slide window

	# kwrvals
	kwrvals={}
	kwrvals['SSF']=ssf
	kwrvals['Signal']=y
	kwrvals['Onset']=np.array(onset)

	# fig=pl.figure()
	# ax=fig.add_subplot(111)
	# ax.plot(kwrvals['Signal']-np.mean(kwrvals['Signal']))
	# ax.plot(kwrvals['SSF']*50,'g')
	# ax.vlines(kwrvals['Onset'],-0.20,0.20,'r')
	# for a,b in zip(kwrvals['Onset'],dmm):
		# ax.text(a, 0.2, b)
		# ax.text(a, -0.2, a)
	# ax.axis('tight')
	# ax.grid('on')
	# fig.show()	
	return kwrvals 
#----------------------------------------------------------------------------------------------------------------------------
