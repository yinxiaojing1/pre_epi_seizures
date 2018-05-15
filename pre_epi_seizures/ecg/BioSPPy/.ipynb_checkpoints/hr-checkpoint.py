import numpy as np

def hr(Signal=None, SamplingRate=1000.):
	"""
		ToDo.

		Parameters
		__________
		ToDo : ToDo
			ToDo.

		Returns
		_______
		kwrvals : dict
			A keyworded return values dict is returned with the following keys: 
			IBI : ndarray
				ToDo.
			HR : ndarray
				ToDo.

		See Also
		________
			bvp.pulse
			ecg.ecg 
		
		Example
		_______
		ToDo

		References
		__________
		.. ToDo
	"""
	if Signal is None:
		raise TypeError, "An input signal is needed."
	
	kwrvals={}
	kwrvals['IBI']=np.diff(Signal)
	
	# if SamplingRate is not None: kwrvals['IBI']=kwrvals['IBI']/float(SamplingRate) 
		
	kwrvals['HR']=(60.0/kwrvals['IBI'])

	return kwrvals

