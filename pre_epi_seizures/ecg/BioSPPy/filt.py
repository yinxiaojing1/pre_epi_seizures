"""
"""

# Imports
# built-in
import unittest

# 3rd party
import numpy as np
import pylab as pl
import scipy.signal as ss
import scipy.signal.windows as wnds



def _wn(Frequency=None, SamplingRate=1000.):
    """
    --INTERNAL FUNCTION--
    
    Normalize frequency to Nyquist frequency (Fs/2).
    
    Kwargs:
        Frequency (int, float, list, array): The frequency (or array of frequencies) to convert.
        
        SamplingRate (int, float): The sampling frequency (Hz).
    
    Kwrvals:
        freq (float, array): The normalized frequency (or array of frequencies).
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1] 
    
    """
    
    # check inputs
    if Frequency is None:
        raise TypeError, "Please specify a frequency to normalize."
    
    # convert inputs to correct representation
    try:
        Frequency = float(Frequency)
    except TypeError:
        # maybe Frequency is list or array
        Frequency = np.array(Frequency, dtype='float')
    
    SamplingRate = float(SamplingRate)
    
    return 2. * Frequency / SamplingRate


def _getFilter(FilterType='FIR', Order=None, Frequency=None, SamplingRate=1000., BandType='lowpass', **kwargs):
    """
    --INTERNAL FUNCTION--
    
    Compute the digital (FIR or IIR) filter coefficients with the given parameters.
    
    Supported filter functions: FIR, Butterworth, Chebyshev Type I, Chebyshev Type II, Elliptic, and Bessel.
    
    Kwargs:
        FilterType (str): The filter function: 'FIR', 'butter', 'cheby1', 'cheby2', 'ellip', or 'bessel' (default='FIR').
        
        Order (int): Order of the filter.
        
        Frequency (int, float, list, array): The cutoff frequency (or array of frequencies).
        
        SamplingRate (int, float): The sampling frequency (Hz).
        
        BandType (str): The type of the filter: 'lowpass', 'highpass', 'bandpass', or 'bandstop' (default='lowpass').
        
        **kwargs (dict): Additional keyword arguments are passed to the underlying scipy.signal function.
    
    Kwrvals:
        b (array): Numerator of the filter.
        
        a (array): Denominator of the filter.
    
    See Also:
        scipy.signal.firwin
        scipy.signal.butter
        scipy.signal.cheby1
        scipy.signal.cheby2
        scipy.signal.ellip
        scipy.signal.bessel
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1] 
    
    """
    
    # check inputs
    if Order is None:
        raise TypeError, "Please specify the order of the filter."
    if Frequency is None:
        raise TypeError, "Please specify the cutoff frequency."
    if BandType not in ['lowpass', 'highpass', 'bandpass', 'bandstop']:
        raise ValueError, "Unknown filter type '%s'; choose 'lowpass', 'highpass', 'bandpass', or 'bandstop'." % str(BandType)
    
    # convert frequencies
    Frequency = _wn(Frequency, SamplingRate)
    
    # get coeffs
    b, a = [], []
    if FilterType == 'FIR':
        # FIR filter
        a = np.array([1])
        if BandType in ['lowpass', 'bandstop']:
            b = ss.firwin(numtaps=Order+1, cutoff=Frequency, pass_zero=True, **kwargs)
        elif BandType in ['highpass', 'bandpass']:
            b = ss.firwin(numtaps=Order+1, cutoff=Frequency, pass_zero=False, **kwargs)
    elif FilterType == 'butter':
        # Butterworth filter
        b, a = ss.butter(N=Order, Wn=Frequency, btype=BandType, analog=False, output='ba', **kwargs)
    elif FilterType == 'cheby1':
        # Chebyshev type I filter
        b, a = ss.cheby1(N=Order, Wn=Frequency, btype=BandType, analog=False, output='ba', **kwargs)
    elif FilterType == 'cheby2':
        # chevyshev type II filter
        b, a = ss.cheby2(N=Order, Wn=Frequency, btype=BandType, analog=False, output='ba', **kwargs)
    elif FilterType == 'ellip':
        # Elliptic filter
        b, a = ss.ellip(N=Order, Wn=Frequency, btype=BandType, analog=False, output='ba', **kwargs)
    elif FilterType == 'bessel':
        # Bessel filter
        b, a = ss.bessel(N=Order, Wn=Frequency, btype=BandType, analog=False, output='ba', **kwargs)
    
    return b, a


def _getInititalState(b, a, alpha=1):
    """
    --INTERNAL FUNCTION--
    
    Get an initial filter state that corresponds to the steady state of the step response.
    
    Kwargs:
        b (array): Numerator coefficients.
        
        a (array): Denominator coefficients.
        
        alpha (int, float): Scaling factor.
    
    Kwrvals:
        zi (array): The initial filter state
    
    See Also:
        _getFilter
        _plotFilter
        scipy.signal.lfilter
        scipy.signal.lfilter_zi
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1] 
    
    """
    
    return alpha * ss.lfilter_zi(b, a)


def _filterSignal(b, a, signal, zi=None, checkPhase=True, **kwargs):
    """
    --INTERNAL FUNCTION--
    
    Filter a signal given the filter coefficients.
    
    Kwargs:
        b (array): Numerator coefficients.
        
        a (array): Denominator coefficients.
        
        signal (array): The signal to filter.
        
        zi (array): Initial filter state (default=None)
        
        checkPhase (bool): If True, use the forward-backward filter technique.
        
        **kwargs (dict): Additional keyword arguments are passed to the underlying filtering function.
    
    Kwrvals:
        filtered (array): The filtered signal.
        
        zf (array, optional): The final filter state.
    
    See Also:
        _getFilter
        _plotFilter
        scipy.signal.lfilter
        scipy.signal.filtfilt
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1] 
    
    """
    
    # check inputs
    if checkPhase and zi is not None:
        raise ValueError, "Incompatible arguments: the initial filter state can not be set when checkPhase is True."
    
    # filter
    if checkPhase:
        out = ss.filtfilt(b, a, signal, **kwargs)
    else:
        out = ss.lfilter(b, a, signal, zi=zi, **kwargs)
    
    return out


def filterSignal(Signal=None, SamplingRate=1000., FilterType='FIR', Order=None,
                 Frequency=None, BandType='lowpass', **kwargs):
    """
    Filter a signal according to the given parameters.
    Uses a forward-backward filter implementation. Therefore, the combined filter has linear phase.
    
    Supported filter functions: FIR, Butterworth, Chebyshev Type I, Chebyshev Type II, Elliptic, and Bessel.
    
    Kwargs:
        Signal (array): The signal to filter.
        
        SamplingRate (int, float): The sampling frequency (Hz).
        
        FilterType (str): The filter function: 'FIR', 'butter', 'cheby1', 'cheby2', 'ellip', or 'bessel' (default='FIR').
        
        Order (int): Order of the filter.
        
        Frequency (int, float, list, array): The cutoff frequency (or list/array of low and high cutoff frequencies).
        
        BandType (str): The type of the filter: 'lowpass', 'highpass', 'bandpass', or 'bandstop' (default='lowpass').
        
        **kwargs (dict): Additional keyword arguments are passed to the underlying scipy.signal function.
    
    Kwrvals:
        Signal (array): The filtered signal.
        
        SamplingRate (float): The sampling frequency (Hz).
        
        Filter (dict): The filter parameters.
    
    See Also:
        _getFilter
        plotFilter
        scipy.signal.filtfilt
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1] 
    
    """
    
    # check inputs
    if Signal is None:
        raise TypeError, "Please specify a signal to filter."
    
    # get filter
    b, a = _getFilter(FilterType=FilterType, Order=Order, Frequency=Frequency,
                      SamplingRate=SamplingRate, BandType=BandType, **kwargs)
    
    # filter
    filtered = _filterSignal(b, a, Signal, checkPhase=True)
    
    # output
    opts = dict(**kwargs)
    opts['FilterType'] = FilterType
    opts['Order'] = Order
    opts['Frequency'] = Frequency
    opts['BandType'] = BandType
    
    output = {'Signal': filtered,
              'SamplingRate': SamplingRate,
              'Filter': opts,
              }
    
    return output


def _plotFilter(b, a, SamplingRate=1000., nfreqs=512, ax=None):
    """
    --INTERNAL FUNCTION--
    
    Compute and plot the frequency response of a digital filter.
    
    Kwargs:
        b (array): Numerator of the filter.
        
        a (array): Denominator of the filter.
        
        nfreqs (int): Number of frequency points to compute.
        
        SamplingRate (int, float): The sampling frequency (Hz).
    
    Kwrvals:
        fig (matplotlib.figure.Figure): Figure object.
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1] 
    
    """
    
    # compute frequency response
    w, h = ss.freqz(b, a, nfreqs)
    
    # convert frequencies
    w = (w * SamplingRate) / (2 * np.pi)
    
    # plot
    if ax is None:
        fig = pl.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure
    
    # amplitude
    ax.semilogy(w, np.abs(h), 'b')
    ax.set_ylabel('Amplitude (dB)', color='b')
    ax.set_xlabel('Frequency (Hz)')
    
    # phase
    angles = np.unwrap(np.angle(h))
    ax2 = ax.twinx()
    ax2.plot(w, angles, 'g')
    ax2.set_ylabel('Angle (radians)', color='g')
    
    ax.grid()
    ax.axis('tight')
    
    return fig


def plotFilter(FilterType='FIR', Order=None, Frequency=None, SamplingRate=1000.,
               BandType='lowpass', path=None, show=True, **kwargs):
    """
    Plot the frequency response of the filter specified with the given parameters.
    
    Supported filter functions: FIR, Butterworth, Chebyshev Type I, Chebyshev Type II, Elliptic, and Bessel.
    
    Kwargs:
        FilterType (str): The filter function: 'FIR', 'butter', 'cheby1', 'cheby2', 'ellip', or 'bessel' (default='FIR').
        
        Order (int): Order of the filter.
        
        Frequency (int, float, list, array): The cutoff frequency (or array of frequencies).
        
        SamplingRate (int, float): The sampling frequency (Hz).
        
        BandType (str): The type of the filter: 'lowpass', 'highpass', 'bandpass', or 'bandstop' (default='lowpass').
        
        path (str): If given, the plot will be saved to the file specified (default=None).
        
        show (bool): If True, show the plot immediately.
        
        **kwargs (dict): Additional keyword arguments are passed to the underlying scipy.signal function.
    
    Kwrvals:
        
    
    See Also:
        filterSignal
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1] 
    
    """
    
    # get filter
    b, a = _getFilter(FilterType=FilterType, Order=Order, Frequency=Frequency,
                      SamplingRate=SamplingRate, BandType=BandType, **kwargs)
    
    # plot
    fig = _plotFilter(b, a, SamplingRate)
    
    # save to file
    if path is not None:
        try:
            if '.png' not in path:
                path += '.png'
            fig.savefig(path, dpi=200, bbox_inches='tight')
        except Exception, e:
            raise StandardError, "Filter plot could not be saved: %s" % str(e)
    
    # show
    if show:
        pl.show()
    
    # close
    pl.close(fig)


def firfilt(data, n, l, h, SamplingRate):
    """
    --DEPRECATED--
    """
    bfir1 = ss.firwin(n+1, [2*float(l)/SamplingRate, 2*float(h)/SamplingRate], pass_zero=False)
    return ss.lfilter(bfir1, [1], data)[n:]


def zpdfr(Signal=None, SamplingRate=None, UpperCutoff=None, LowerCutoff=None, Order=4.):
    """
    --DEPRECATED FUNCTION--

    Kwargs:
        

    Kwrvals:
        

    See Also:


    Notes:
        

    Example:


    References:
        .. [1] 
        
    """
    
    if SamplingRate: 
        UpperCutoff=_wn(Frequency=UpperCutoff, SamplingRate=float(SamplingRate)) if UpperCutoff else None
        LowerCutoff=_wn(Frequency=LowerCutoff, SamplingRate=float(SamplingRate)) if LowerCutoff else None
    
    kwrvals={}
    
    kwrvals['Signal']=Signal

    if UpperCutoff:
        #Determine filter coefficients
        [b,a]=ss.butter(Order,UpperCutoff,'low')
    
        #Filter signal
        kwrvals['Signal']=ss.filtfilt(b,a,kwrvals['Signal'])

    if LowerCutoff:
        #Determine filter coefficients
        [b,a]=ss.butter(Order,LowerCutoff,'high')
        
        #Filter signal
        kwrvals['Signal']=ss.filtfilt(b,a,kwrvals['Signal'])
    
    return kwrvals


def smooth(Signal=None, Window={}):
    """
    Smooth data using a N-point moving average filter.

    This implementation uses the convolution of a filter kernel with the input
    signal `x` to compute the smoothed signal.

    Parameters
    
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
    
    y : ndarray
    A smoothed version of the input signal computed using a filter kernel of 
    size `n` generated according to `wtype`

    See Also
    
    scipy.signal.windows

    Notes
    
    A combination smoothing method ``boxzen`` was introduced and is currently used
    as default to produce the output signal.

    This method first smooths the signal using the ``scipy.windows.boxcar`` window 
    and then smooths ir again using the ``scipy.windows.parzen`` window.

    The resulting signals can be quite interesting, as ``boxcar``retains a great
    proximity to the original data waveforms, and ``parzen`` removes the rough edges.

    Example
    
    t = arange(0,2*pi,.1)
    x = sin(t)+0.5*(rand(len(t))-0.5)
    y = smooth(x)
    plot(x,'.')
    plot(y)
    legend(('original data', 'smoothed with boxzen'))

    References
    
    .. [1] Wikipedia, "Moving Average". http://en.wikipedia.org/wiki/Moving_average
    .. [2] S. W. Smith, "Moving Average Filters - Implementation by Convolution". 
    http://www.dspguide.com/ch15/1.htm

    Kwargs:
        

    Kwrvals:
        

    See Also:


    Notes:
        

    Example:


    References:
        .. [1] 
        
    """
    x=Signal
    n=Window['Length'] if Window.has_key('Length') else 10
    wtype=Window['Type'] if Window.has_key('Type') else 'boxzen'
    args=Window['Parameters'] if Window.has_key('Parameters') else None

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
    wfcn=windowfcn(wtype)

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


def windowfcn(w):
    """
    Retrieve the appropriate window function that corresponds to the descriptor `w`.

    Parameters

    w : str, function
    Name or descriptor of the window function. If this is a `function`, it will be 
    directly returned. If this is a `str`, a window function with a matching name 
    will be searched for in module ``scipy.signal.windows``.

    Returns

    f : function
    The window function corresponding to the descriptor `w`

    See Also

    scipy.signal.windows

    Notes

    If no window function with a name matching the descriptor `w` is found in the 
    module ``scipy.signal.windows``, the window function with the most similar name 
    is be returned and a warning is issued.

    Example

    f = windowfcn('gaussian')
    f = windowfcn('gauss')    

    Kwargs:
        

    Kwrvals:
        

    See Also:


    Notes:
        

    Example:


    References:
        .. [1] 
        
    """    
    wtype=type(w).__name__

    # Check if the descriptor `w` is already a function and return it directly
    if (wtype=='function'):
        return w

    # Check if the descriptor `w` is a string
    if (wtype=='str'):
        # Validate if a window function matching the descriptor `w` belongs to ``scipy.signal.windows``
        try:
            return eval('wnds.'+w)
        # Search for the most similar name if no direct match was found 
        except:
            # List the window functions that are available in ``scipy.signal.windows``
            wlist=dir(wnds)
            
            # Search for the window function names that contain the descriptor `w`
            wfcn=filter(lambda wname: w in wname and '__' not in wname, wlist)
        
            # Raise an exception if no similar window function was found
            if (np.size(wfcn)==0):
                raise AttributeError("'"+wnds.__name__+"' object has no attribute '"+w+"'.")
            
            # Return the most similar window function and raise a warning signaling it
            w=wfcn[0]
            print "Warning: using '"+w+"' window."
            w=eval('wnds.'+w)
            
            return w
                
    raise TypeError("'"+wtype+"' object cannot be resolved to a window function descriptor")        


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
