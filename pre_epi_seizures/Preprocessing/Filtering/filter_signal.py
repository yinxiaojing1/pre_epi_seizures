from __future__ import division

import numpy as np
import scipy
import scipy.signal as ss

def get_filter(ftype='FIR', band='lowpass', order=None,
               frequency=None, sampling_rate=500., **kwargs):
    """Compute digital (FIR or IIR) filter coefficients with the given
    parameters.
    Parameters
    ----------
    ftype : str
        Filter type:
            Finite Impulse Response filter ('FIR');
            Butterworth filter ('butter');
            Chebyshev filters ('cheby1', 'cheby2');
            Elliptic filter ('ellip');
            Bessel filter ('bessel').
    band : str
        Band type:
            Low-pass filter ('lowpass');
            High-pass filter ('highpass');
            Band-pass filter ('bandpass');
            Band-stop filter ('bandstop').
    order : int
        Order of the filter.
    frequency : int, float, list, array
        Cutoff frequencies; format depends on type of band:
            'lowpass' or 'bandpass': single frequency;
            'bandpass' or 'bandstop': pair of frequencies.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    kwargs : dict, optional
        Additional keyword arguments are passed to the underlying
        scipy.signal function.
    Returns
    -------
    b : array
        Numerator coefficients.
    a : array
        Denominator coefficients.
    See Also:
        scipy.signal
    """

    # check inputs
    if order is None:
        raise TypeError("Please specify the filter order.")
    if frequency is None:
        raise TypeError("Please specify the cutoff frequency.")
    if band not in ['lowpass', 'highpass', 'bandpass', 'bandstop']:
        raise ValueError(
            "Unknown filter type '%r'; choose 'lowpass', 'highpass', \
            'bandpass', or 'bandstop'."
            % band)

    # convert frequencies
    frequency = _norm_freq(frequency, sampling_rate)

    # get coeffs
    b, a = [], []
    if ftype == 'FIR':
        # FIR filter
        if order % 2 == 0:
            order += 1
        a = np.array([1])
        if band in ['lowpass', 'bandstop']:
            b = ss.firwin(numtaps=order,
                          cutoff=frequency,
                          pass_zero=True, **kwargs)
        elif band in ['highpass', 'bandpass']:
            b = ss.firwin(numtaps=order,
                          cutoff=frequency,
                          pass_zero=False, **kwargs)
        elif ftype == 'butter':
        # Butterworth filter
        b, a = ss.butter(N=order,
                         Wn=frequency,
                         btype=band,
                         analog=False,
                         output='ba', **kwargs)
    elif ftype == 'cheby1':
        # Chebyshev type I filter
        b, a = ss.cheby1(N=order,
                         Wn=frequency,
                         btype=band,
                         analog=False,
                         output='ba', **kwargs)
    elif ftype == 'cheby2':
        # chevyshev type II filter
        b, a = ss.cheby2(N=order,
                         Wn=frequency,
                         btype=band,
                         analog=False,
                         output='ba', **kwargs)
    elif ftype == 'ellip':
        # Elliptic filter
        b, a = ss.ellip(N=order,
                        Wn=frequency,
                        btype=band,
                        analog=False,
                        output='ba', **kwargs)
    elif ftype == 'bessel':
        # Bessel filter
        b, a = ss.bessel(N=order,
                         Wn=frequency,
                         btype=band,
                         analog=False,
                         output='ba', **kwargs)

    return b, a


def _norm_freq(frequency=None, sampling_rate=500.):
    """Normalize frequency to Nyquist Frequency (Fs/2).
    Parameters
    ----------
    frequency : int, float, list, array
        Frequencies to normalize.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    Returns
    -------
    wn : float, array
        Normalized frequencies.
    """
    # check inputs
    if frequency is None:
        raise TypeError("Please specify a frequency to normalize.")

    # convert inputs to correct representation
    try:
        frequency = float(frequency)
    except TypeError:
        # maybe frequency is a list or array
        frequency = np.array(frequency, dtype='float')
    Fs = float(sampling_rate)
    wn = 2. * frequency / Fs
    return wn

def _filter_signal(b, a, signal, zi=None, check_phase=True, **kwargs):
    """Filter a signal with given coefficients.
    Parameters
    ----------
    b : array
        Numerator coefficients.
    a : array
        Denominator coefficients.
    signal : array
        Signal to filter.
    zi : array, optional
        Initial filter state.
    check_phase : bool, optional
        If True, use the forward-backward technique.
    **kwargs : dict, optional
        Additional keyword arguments are passed to the underlying filtering
        function.
    Returns
    -------
    filtered : array
        Filtered signal.
    zf : array
        Final filter state.
    Notes
    -----
    If check_phase is True, zi cannot be set.
    """
    # check inputs
    if check_phase and zi is not None:
        raise ValueError(
            "Incompatible arguments: initial filter state cannot be set when \
            check_phase is True.")
    if check_phase:
        filtered = ss.filtfilt(b, a, signal, **kwargs)
        zf = None
    else:
        filtered, zf = ss.lfilter(b, a, signal, zi=zi, **kwargs)
    return filtered, zf

def filter_signal(signal=None, ftype='FIR', band='lowpass',
                  order=None, frequency=None,
                  sampling_rate=500., **kwargs):
    """Filter a signal according to the given parameters.
    Parameters
    ----------
    signal : array
        Signal to filter.
    ftype : str
        Filter type:
            * Finite Impulse Response filter ('FIR');
            * Butterworth filter ('butter');
            * Chebyshev filters ('cheby1', 'cheby2');
            * Elliptic filter ('ellip');
            * Bessel filter ('bessel').
    band : str
        Band type:
            * Low-pass filter ('lowpass');
            * High-pass filter ('highpass');
            * Band-pass filter ('bandpass');
            * Band-stop filter ('bandstop').
    order : int
        Order of the filter.
    frequency : int, float, list, array
        Cutoff frequencies; format depends on type of band:
            * 'lowpass' or 'bandpass': single frequency;
            * 'bandpass' or 'bandstop': pair of frequencies.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    **kwargs : dict, optional
        Additional keyword arguments are passed to the underlying
        scipy.signal function.
    Returns
    -------
    signal : array
        Filtered signal.
    Notes
    -----
    * Uses a forward-backward filter implementation. Therefore, the combined
      filter has linear phase.
    """
    if signal is None:
        raise TypeError("Please specify a signal to filter.")
    b, a = get_filter(ftype=ftype,
                      order=order,
                      frequency=frequency,
                      sampling_rate=sampling_rate,
                      band=band, **kwargs)
    signal = _filter_signal(b, a, signal, check_phase=True)[0]
    return signal
