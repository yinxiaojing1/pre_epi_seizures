"""
	This module provides various functions to ...

	Functions:

	step()
	threshold()
	match()
	pair()

	"""	
import pylab as pl

#----------------------------------------------------------------------------------------------------------------------------
# Find the indexes where a dirac, unit step or unit pulse function in the input signal rises or falls
def step(Signal=None, Shift='>'):
	"""
		Find the indexes where a dirac, unit step or unit pulse function in the input 
		signal rises or falls.

		This implementation is based on the discrete difference of the input signal.

		Parameters
		__________
		Signal : ndarray
			Input signal data with a dirac, unit step or unit pulse function.
		Shift : str
			Direction of detection, `>` for rise or `<` for fall. 
			Default: `>`

		Returns
		_______
		kwrvals : dict
			A keyworded return values dict is returned with the following keys: 
			Event : ndarray
				The indexes within the input signal `Signal` where the `Shift` was detected.

		See Also
		________
			sync.threshold
			sync.match 
		
		Example
		_______
		x = zeros(6)
		x[2:4] = 1
		plot(x)
		vlines(step(x)['Events'],min(x),max(x),'r','dashed')
		vlines(step(x,'<')['Events'],min(x),max(x),'g','dashed')
		legend(('unit pulse','rise', 'fall'))

		References
		__________
		.. [1] Wikipedia, "Dirac Delta Function". 
		http://en.wikipedia.org/wiki/Dirac_delta_function
		.. [2] Wikipedia, "Heaviside Step Function".
		http://en.wikipedia.org/wiki/Heaviside_step_function
		.. [3] Wikipedia, "Unit Pulse Function".
		http://en.wikipedia.org/wiki/Rectangular_function
	"""
	# Check
	if Signal is None:
		raise TypeError, "An input signal is needed."
	# 
	th=(1 if Shift=='>' else -1)
	kwrvals={}
	kwrvals['Event']=pl.find(pl.diff(Signal)==th)

	return kwrvals
#----------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------
# Find the indexes where a the input signal rises above or falls bellow a given threshold.
def threshold(Signal=None, Threshold=.1, Shift='>'):
	"""
		Find the indexes where a the input signal rises above or falls bellow a given
		threshold.

		Parameters
		__________
		Signal : ndarray
			Input signal data.
		Threshold : int, float
			Detection threshold. If this is an `int` greater or equal than one it will be 
			directly used as the threshold. If this is a `float`, the threshold is 
			computed as the `Threshold` percentage of the signal span.
			Default: .1
		Shift : str
			Direction of detection, `>` for rise or `<` for fall. 
			Default: `>`

		Returns
		_______
		kwrvals : dict
			A keyworded return values dict is returned with the following keys: 
			Event : ndarray
				The indexes within the input signal `Signal` where a `Shift` with respect to 
				`Threshold` was detected.

		See Also
		________
			sync.step
			sync.match

		Notes
		_____
		This method is primarily designed to detect shifts in the input signal with
		respect to a threshold computed as a percentage of the signal span.

		Example
		_______
		x = zeros(6)
		x[2:4] = 1
		plot(x)
		th=.5
		plot(ones(len(x))*th,'k--')
		vlines(threshold(x,th)['Event'],min(x),max(x),'r','dashed')
		vlines(threshold(x,th,'<')['Event'],min(x),max(x),'g','dashed')
		legend(('unit pulse','rise', 'fall'))
		
	"""
	# Check
	if Signal is None:
		raise TypeError, "An input signal is needed."	
	if Threshold<1.: Threshold=Signal.min()+Threshold*(Signal.max()-Signal.min())
	
	return step((Signal>Threshold).astype('int'), Shift)

#----------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------
# 
def match(Signal=None, Window=None):
	"""
		.

		Parameters
		__________
		Signal : ndarray
			Input signal data.
		Window : 

		Returns
		_______
		kwrvals : dict
			A keyworded return values dict is returned with the following keys: 
			Event : 
		

		See Also
		________
			pl.convolve

		Notes
		_____


		Example
		_______

		
	"""
	# Check
	if Signal is None:
		raise TypeError, "An input signal is needed."	
	# 
	lw=len(Window)
	lx=len(Signal)
	
	if lw>lx: raise AttributeError("Window length must be less or equal to the length of the input signal.")  
	
	kwrvals={}
	#kwrvals['dt']=pl.array(map(lambda i: pl.sqrt(sum((Signal[i+lw]-Window)**2)), pl.arange(0, lx-lw))).argmin()
	kwrvals['dt']=pl.convolve(Window,Signal).argmax()

	return kwrvals
	# return pl.array(map(lambda i: pl.sqrt(sum((x[i+lw]-w)**2)), pl.arange(0, lx-lw))).argmin()
#----------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------
#
def pair(x,y):
	"""
		.

		Parameters
		__________
		x : array
			
		y : array

		Returns
		_______
		x : array
			
		y : array
	
	
		See Also
		________
		

		Notes
		_____


		Example
		_______

		
	"""
	if y[0]<x[0]: y=y[1:]
	if x[-1]>y[-1]: x=x[:-1]
	#else: x=x[x>=y]
	if len(x)>len(y): x=x[:-1]

	return (x,y)
#----------------------------------------------------------------------------------------------------------------------------

