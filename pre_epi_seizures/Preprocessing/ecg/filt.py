import numpy as np
import pylab as pl
import scipy.signal as sig
import scipy.signal.windows as wnds
import unittest
import tools

def wn(Frequency=None, SamplingRate=1000.):
    """
        .
		
        Parameters
        __________
        
        
        Returns
        _______
        
        
        See Also
        ________
        
        
        Notes
        _____
        
        
        Example
        _______
        
        
        References
        __________
        .. [1] 
        
        http://
        """	

    return 2.*Frequency/SamplingRate


def zpdfr(Signal=None, SamplingRate=None, UpperCutoff=None, LowerCutoff=None, Order=4.):
    """
        .
		
        Parameters
        __________
        
        
        Returns
        _______
        
        
        See Also
        ________
        
        
        Notes
        _____
        
        
        Example
        _______
        
        
        References
        __________
        .. [1] 
        
        http://
        """	
    
    if SamplingRate: 
        UpperCutoff=wn(Frequency=UpperCutoff, SamplingRate=float(SamplingRate)) if UpperCutoff else None
        LowerCutoff=wn(Frequency=LowerCutoff, SamplingRate=float(SamplingRate)) if LowerCutoff else None
    
    kwrvals={}
    
    kwrvals['Signal']=Signal

    if UpperCutoff:
        #Determine filter coefficients
        [b,a]=sig.butter(Order,UpperCutoff,'low')
    
        #Filter signal
        kwrvals['Signal']=sig.filtfilt(b,a,kwrvals['Signal'])

    if LowerCutoff:
        #Determine filter coefficients
        [b,a]=sig.butter(Order,LowerCutoff,'high')
        
        #Filter signal
        kwrvals['Signal']=sig.filtfilt(b,a,kwrvals['Signal'])
    
    return kwrvals


def smooth(Signal=None, Window={}):
    """
        Smooth data using a N-point moving average filter.

        This implementation uses the convolution of a filter kernel with the input
        signal `x` to compute the smoothed signal.
        
        Parameters
        __________
        x : ndarray
        Input signal data.
        n : int, float, ndarray, optional
        Number of points of the filter kernel. If this is an `int`, the filter 
        kernel will have `n` points. If this is a `float`, the number of points
        for the filter kernel will be set as `n*size(x)`. If this is a `ndarray`,
        it will be directly taken as the filter kernel.
        Default: 10
        wtype : str, function, optional
        Method that should be used to determine the filter kernel. If this is a
        `function`, it will be invoked with parameters `n`and `args` to determine
        the filter kernel. If this is a `str`, the function with the matching or
        most similar name belonging to the module ``scipy.signal.windows`` is used.
        Default: ``boxzen``
        *args : optional
        Additional parameters that may be required by the filter kernel function
        
        Returns
        _______
        y : ndarray
        A smoothed version of the input signal computed using a filter kernel of 
        size `n` generated according to `wtype`
        
        See Also
        ________
        scipy.signal.windows
        
        Notes
        _____
        A combination smoothing method ``boxzen`` was introduced and is currently used
        as default to produce the output signal.
        
        This method first smooths the signal using the ``scipy.windows.boxcar`` window 
        and then smooths ir again using the ``scipy.windows.parzen`` window.
        
        The resulting signals can be quite interesting, as ``boxcar``retains a great
        proximity to the original data waveforms, and ``parzen`` removes the rough edges.
        
        Example
        _______
        t = arange(0,2*pi,.1)
        x = sin(t)+0.5*(rand(len(t))-0.5)
        y = smooth(x)
        plot(x,'.')
        plot(y)
        legend(('original data', 'smoothed with boxzen'))
        
        References
        __________
        .. [1] Wikipedia, "Moving Average". http://en.wikipedia.org/wiki/Moving_average
        .. [2] S. W. Smith, "Moving Average Filters - Implementation by Convolution". 
        http://www.dspguide.com/ch15/1.htm
        """ 
    x=Signal
    n=Window['Length'] if ('Length' in Window) else 10
    wtype=Window['Type'] if ('Type' in Window) else 'boxzen'
    args=Window['Parameters'] if ('Parameters' in Window) else None
    
    nsiz=np.size(n)
    
    if (nsiz<1):
        raise ValueError("N must be a number or an array.")

    # Validate if a window vector was directly provided as parameter
    if (nsiz>1):
        w=n
    else:
        # Validate if the combination method was selected
        if (wtype=='boxzen'):
            return smooth(smooth(x, Window={'Length':n, 'Type':'boxcar','Parameters':None})['Signal'], Window={'Length':n, 'Type':'parzen', 'Parameters':None})

    # Retrieve the function that produces the window according to the selected type
    #wfcn=windowfcn(wtype)
    wfcn=tools.seekfcn(wtype, wnds)
    
    xsiz=len(x)
    
    # If the parameter `n` defines as a percentage of the data vector size compute its value
    if (n<1):
        n=pl.ceil(n*xsiz)

    # If the parameter `n` exceeds the data vector size adjust its value
    if (n>=xsiz):
        n=xsiz-1

    # If a Gaussian window type was selected without the standard deviation parameter set to a default value
    if (wfcn==wnds.gaussian and np.size(args)==0):
        args=(np.ceil(n*.1),)

    # Compute the window
    w=wfcn(n, args)
    
    # Compute the smoothed signal
    kwrvals={}
    kwrvals['Signal']=np.convolve(w/w.sum(), x, mode='same')
    
    return kwrvals


## LEGACY: Updated to 'tools.seekfcn'
#def windowfcn(w):
#    """
#        Retrieve the appropriate window function that corresponds to the descriptor `w`.
#        
#        Parameters
#        __________
#        w : str, function
#        Name or descriptor of the window function. If this is a `function`, it will be 
#        directly returned. If this is a `str`, a window function with a matching name 
#        will be searched for in module ``scipy.signal.windows``.
#        
#        Returns
#        _______
#        f : function
#        The window function corresponding to the descriptor `w`
#        
#        See Also
#        ________
#        scipy.signal.windows
#        
#        Notes
#        _____
#        If no window function with a name matching the descriptor `w` is found in the 
#        module ``scipy.signal.windows``, the window function with the most similar name 
#        is be returned and a warning is issued.
#        
#        Example
#        _______
#        f = windowfcn('gaussian')
#        f = windowfcn('gauss')
#        """
#    
#    wtype=type(w).__name__
#    
#    # Check if the descriptor `w` is already a function and return it directly
#    if (wtype=='function'):
#        return w
#    
#    # Check if the descriptor `w` is a string
#    if (wtype=='str'):
#        # Validate if a window function matching the descriptor `w` belongs to ``scipy.signal.windows``
#        try:
#            return eval('wnds.'+w)
#        # Search for the most similar name if no direct match was found 
#        except:
#            # List the window functions that are available in ``scipy.signal.windows``
#            wlist=dir(wnds)
#            
#            # Search for the window function names that contain the descriptor `w`
#            wfcn=filter(lambda wname: w in wname and '__' not in wname, wlist)
#        
#            # Raise an exception if no similar window function was found
#            if (np.size(wfcn)==0):
#                raise AttributeError("'"+wnds.__name__+"' object has no attribute '"+w+"'.")
#            
#            # Return the most similar window function and raise a warning signaling it
#            w=wfcn[0]
#            print "Warning: using '"+w+"' window."
#            w=eval('wnds.'+w)
#            
#            return w
#                
#    raise TypeError("'"+wtype+"' object cannot be resolved to a window function descriptor")        


if __name__=='__main__':
    
    class testSmooth(unittest.TestCase):
        """
            A test class for the smooth module.
            """
    
        def setUp(self):
            # Define a centered pulse dirac input signal with 100 points
            self.npts=100.
            self.signal=np.zeros(self.npts)
            self.signal[self.npts/2.]=1.
    
    
        def testWindowFcn(self):
            # Test if a window function is returned when passed as descriptor
            self.assertEqual(windowfcn(wnds.boxcar), wnds.boxcar)
            # Test if a window function is returned when a matching function name is passed as descriptor
            self.assertEqual(windowfcn('parzen'), wnds.parzen)
            # Test if a window function is returned when part of the function name is passed as descriptor
            self.assertEqual(windowfcn('gauss'), wnds.gaussian)
        
            # Test if the right exception is raised when the window function descriptor does not correspond in part or in full to a window function
            self.assertRaises(AttributeError, windowfcn, 'foo')
            # Test if the right exception is raised when the window function descriptor does not have a valid type
            self.assertRaises(TypeError, windowfcn, np.array([]))
    
    
        def testSmooth(self):
            # Test the default window size (10% of input signal length)
            self.assertEqual(sum(smooth(self.signal,wtype='boxcar')!=0), self.npts*0.1)
            # Test the default window type ('boxzen')
            self.assertEqual(sum(smooth(self.signal)), sum(smooth(self.signal,wtype='boxzen')))
            # Test the default standard deviation for 'gaussian' windows (10% of window size)
            self.assertEqual(sum(smooth(self.signal, 10, 'gaussian')), sum(smooth(self.signal, 10, 'gaussian', 1.)))
            # Test a window passed as parameter 
            self.assertEqual(sum(smooth(self.signal, 10, 'boxcar')), sum(smooth(self.signal, wnds.boxcar(10))))
            # Test the window normalization step
            n=10.;self.assertEqual(sum(smooth(self.signal, 10, 'boxcar')==1/n), n)
                
            # Test if the right exception is raised when the window size is <=0
            self.assertRaises(ValueError, smooth, self.signal, 0)

    
    # Example
    # -------
    t=np.arange(0,2*pl.pi,.1)

    x=pl.sin(t)    
    w=0.5*(pl.rand(len(t))-0.5)
    xw=x+w
    
    y=smooth(xw, wtype='boxzen')
    
    pl.figure()
    
    pl.plot(x)
    pl.plot(xw)
    pl.plot(y)
    
    pl.legend(['original', 'noisy', 'boxzen'])
    pl.show()
    

    unittest.main()

