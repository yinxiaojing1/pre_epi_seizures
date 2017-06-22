"""
.. module:: models
   :platform: Unix, Windows
   :synopsis: This module provides various methods to segment ECG signals.

.. moduleauthor:: Filipe Canento, Carlos Carreiras, Francisco David
"""

# Imports
# built-in
import collections
import traceback

# 3rd party
# from numba import int32, jit, float64
import numpy as np
import pylab as pl
import scipy
import scipy.signal as ss

# local
import peakd
import filt as flt
import tools



def definepeak(signal, srate):
    
    bin_nr = 50
    meankill = 1/3.3
    upmargin = 0.91
    w_peaks = peakd.sgndiff(Signal=signal)['Peak']
    w_negpeaks = peakd.sgndiff(Signal=signal, a=1)['Peak']
    zerdiffs = np.where(scipy.diff(signal) == 0)[0]
    w_peaks = np.concatenate((w_peaks, zerdiffs))
    w_negpeaks = np.concatenate((w_negpeaks, zerdiffs))

    poscounts, posedges = np.histogram(signal[w_peaks], bin_nr)
    negcounts, negedges = np.histogram(signal[w_negpeaks], bin_nr)

    poscond = poscounts > 0
    # posinds = pl.find(poscounts > 0)
    negcond = negcounts > 0
    # neginds = pl.find(negcounts > 0)
    derppos = poscounts[poscond]
    derpneg = negcounts[negcond]
    poscond = np.concatenate(([False], poscond))
    derp_edgepos = posedges[poscond]
    derp_edgeneg = negedges[negcond]

    meanpos = int(meankill*np.mean(derppos))
    meanneg = int(meankill*np.mean(derpneg))
    
    if meanpos < 2:
        meanpos = 2
    if meanneg < 2:
        meanneg = 2
    
    killpos = derppos >= meanpos
    killneg = derpneg >= meanneg
    # derppos = derppos[killpos]
    # derpneg = derpneg[killneg]
    derp_edgepos = derp_edgepos[killpos]
    derp_edgeneg = derp_edgeneg[killneg]
    
    # print derppos, '\n', derp_edgepos
    # print derpneg, '\n', derp_edgeneg
    
    negmax = np.min(derp_edgeneg)
    posmax = np.max(derp_edgepos)
    
    if posmax >= upmargin*abs(negmax):
        # print 'UP', posmax, upmargin*negmax, meanpos, meanneg
        up = True
    else:
        # print 'DOWN', posmax, upmargin*negmax, meanpos, meanneg
        up = False
    
    return up


def monhe(Signal=None, SamplingRate=1000.):
    """
        GREAT but:
            discard crossings close to one another by less than 100 ms

    """
    
    rawend = np.copy(Signal)
    
    # 1 - filter block (chebyshev 4th order 6-18 Hz)
    nyqfreq = SamplingRate/2.
    filtband = [6/nyqfreq, 18/nyqfreq]
    
    num, den = ss.cheby2(4, 40, filtband, btype='bandpass')
    # filtafter2 = ss.filtfilt(num, den, raw)
    filtafter = ss.filtfilt(num, den, rawend)
    
    # 2 - differentiate the signal and normalize according to max derivative in the signal
    diffsig = np.diff(filtafter)
    diffmax = np.max(np.abs(diffsig))
    dsignal = diffsig / diffmax
    
    # 3 - Get Shannon energy envelope
    diffsquare = dsignal**2
    logdiff = np.log(diffsquare)
    shannon = -1*diffsquare*logdiff
    
    # 4 - Two sided zero-phase filtering
    windowlen = 0.15*SamplingRate    # safe length of a QRS pulse
    rectangular = ss.boxcar(windowlen)
    smoothfirst = ss.convolve(shannon, rectangular, mode='same')
    revfirst = smoothfirst[::-1]
    smoothsec = ss.convolve(revfirst, rectangular, mode='same')
    smoothfinal = smoothsec[::-1]
    
    # 5 - Hilbert transform applied to the smoothed shannon energy envelope
    hilbbuff = ss.hilbert(smoothfinal)
    hilbsign = np.imag(hilbbuff)
    
    # 6 - Get moving average of the hilbert transform so as to subtract after
    n = 2.5*SamplingRate
    movwind = np.ones(n) / n
    movav = ss.convolve(hilbsign, movwind, mode='same')
    analyser = hilbsign - movav
    
    # 7 - Get zero crossings (from negative to positive) of the 'analyser' signal
    zero_crossings = np.where(np.diff(np.sign(analyser)))[0]
    zero_crossings = zero_crossings[zero_crossings > 0.05*SamplingRate]    # discard boundary effect that might appear at start
    crossers = analyser[zero_crossings]
    beats = zero_crossings[crossers < 0]
    crossdiffs = np.diff(beats)
    dangerous = crossdiffs < 0.15*SamplingRate          # to avoid stupid repetitions
    dangerous = np.nonzero(dangerous)[0]
    if len(dangerous):
        # print 'DANGER', beats[dangerous]
        beats = np.delete(beats, dangerous)
    
    # 7.1 -------- EXTRA ANTI-FALSE-POSITIVES --------
    store_size = 5
    index_store = 0
    anti_fp = 0.26
    anti_massive = 4
    anti_badset = 3
    reset_count = 0
    cross_storer = np.zeros(store_size)
    crossderivs = np.diff(analyser)
    beats = sorted(list(beats))
    iterator = beats[:]
    evilbeats = []
    for b in iterator:
        cross_med = np.median(cross_storer)
        # print 'info', b, crossderivs[b], anti_fp*cross_med, cross_med, anti_massive*cross_med
        # massive slopes can be eliminated here too (decided not too because it helps in defining Agrafioti windows)
        if crossderivs[b] > anti_fp*cross_med:
            reset_count = 0
            if crossderivs[b] < anti_massive*cross_med or cross_med < 1e-10:
                # print 'store'
                cross_storer[index_store] = crossderivs[b]
                index_store += 1
                if index_store == store_size:
                    index_store = 0
        else:
            reset_count += 1
            # print '\tEVIL SLOPE', b, crossderivs[b], anti_fp*cross_med, reset_count
            evilbeats.append(b)
            beats.remove(b)
        if reset_count >= anti_badset:
            # print '\tRESET'
            reset_count = 0
            cross_storer = np.zeros(store_size)
    beats = np.array(beats)
    evilbeats = np.array(evilbeats)
    
    # 8 ----------------------------------------- Find the R-peak exactly -----------------------------------------
    search = 0.15*SamplingRate
    adjacency = 0.03*SamplingRate
    diff_nr = 0.01*SamplingRate
    rawbeats = []
    for b in xrange(len(beats)):
        if beats[b]-search < 0:
            rawwindow = rawend[0:beats[b]+search]
            add = 0
        elif beats[b]+search >= len(rawend):
            rawwindow = rawend[beats[b]-search:len(rawend)]
            add = beats[b]-search
        else:
            rawwindow = rawend[beats[b]-search:beats[b]+search]
            add = beats[b]-search
        # ----- get peaks -----
        w_peaks = peakd.sgndiff(Signal=rawwindow)['Peak']
        zerdiffs = np.where(scipy.diff(rawwindow) == 0)[0]
        w_peaks = np.concatenate((w_peaks, zerdiffs))

        pospeaks = sorted(zip(rawwindow[w_peaks], w_peaks), reverse=True)
        # print '\n peaksssss', pospeaks

        try:
            twopeaks = [pospeaks[0]]
        except IndexError:
            twopeaks = []
        
        # ----------- getting positive peaks -----------
        for i in xrange(len(pospeaks)-1):
            if abs(pospeaks[0][1] - pospeaks[i+1][1]) > adjacency:
                twopeaks.append(pospeaks[i+1])
                break
        
        poslen = len(twopeaks)
        
        if poslen == 2:
            # --- get maximum slope for max peak ---
            if twopeaks[0][1] < diff_nr:
                diff_f = np.diff(rawwindow[0:twopeaks[0][1]+diff_nr])
            elif twopeaks[0][1] + diff_nr >= len(rawwindow):
                diff_f = np.diff(rawwindow[twopeaks[0][1]-diff_nr:len(rawwindow)])
            else:
                diff_f = np.diff(rawwindow[twopeaks[0][1]-diff_nr:twopeaks[0][1]+diff_nr])
            max_f = np.max(np.abs(diff_f))
            # --- get maximum slope for second peak ---
            if twopeaks[1][1] < diff_nr:
                diff_s = np.diff(rawwindow[0:twopeaks[1][1]+diff_nr-1])
            elif twopeaks[1][1] + diff_nr >= len(rawwindow):
                diff_s = np.diff(rawwindow[twopeaks[1][1]-diff_nr+1:len(rawwindow)])
            else:
                diff_s = np.diff(rawwindow[twopeaks[1][1]-diff_nr+1:twopeaks[1][1]+diff_nr-1])
            max_s = np.max(np.abs(diff_s))
            if max_f > max_s:
                # print '\tbigup'
                assignup = [twopeaks[0][0], twopeaks[0][1]]
            else:
                # print '\tsmallup'
                assignup = [twopeaks[1][0], twopeaks[1][1]]
            rawbeats.append(assignup[1] + add)
        elif poslen == 1:
            rawbeats.append(twopeaks[0][1] + add)
        else:
            rawbeats.append(beats[b])
    
    # 8 ----------------------------------------- END OF POINT 8 -----------------------------------------
    
    # kwrvals
    kwrvals = {'Signal': rawend, 'R': sorted(list(frozenset(rawbeats)))}

    return kwrvals


def monhe2(Signal=None, Filtered=None, SamplingRate=1000., checkSign=True):
    """
        GREAT but:
            discard crossings close to one another by less than 100 ms

    """
    # 0 - Remove EMG, powerline and baseline shift
    raw = Signal
    filtered = Filtered
    
    # 0.5 - Choose sign of peaks (batch)
    if checkSign:
        up = definepeak(filtered, SamplingRate)
    else:
        up = True

    # 1 - filter block (chebyshev 4th order 6-18 Hz)
    nyqfreq = SamplingRate/2.
    filtband = [6/nyqfreq, 18/nyqfreq]

    num, den = ss.cheby2(4, 40, filtband, btype='bandpass')
    filtafter = ss.filtfilt(num, den, raw)
    # filtafter = ss.filtfilt(num, den, rawend)

    # 2 - differentiate the signal and normalize according to max derivative in the signal
    diffsig = np.diff(filtafter)
    diffmax = np.max(np.abs(diffsig))
    dsignal = diffsig / diffmax

    # 3 - Get Shannon energy envelope
    diffsquare = dsignal**2
    logdiff = np.log(diffsquare)
    shannon = -1*diffsquare*logdiff

    # 4 - Two sided zero-phase filtering
    windowlen = int(0.15*SamplingRate)    # safe length of a QRS pulse
    rectangular = ss.boxcar(windowlen)
    smoothfirst = ss.convolve(shannon, rectangular, mode='same')
    revfirst = smoothfirst[::-1]
    smoothsec = ss.convolve(revfirst, rectangular, mode='same')
    smoothfinal = smoothsec[::-1]

    # 5 - Hilbert transform applied to the smoothed shannon energy envelope
    hilbbuff = ss.hilbert(smoothfinal)
    hilbsign = np.imag(hilbbuff)

    # 6 - Get moving average of the hilbert transform so as to subtract after
    n = int(2.5*SamplingRate)
    movwind = np.ones(n) / n
    movav = ss.convolve(hilbsign, movwind, mode='same')
    analyser = hilbsign - movav

    # 7 - Get zero crossings (from negative to positive) of the 'analyser' signal
    zero_crossings = np.where(np.diff(np.sign(analyser)))[0]
    zero_crossings = zero_crossings[zero_crossings > 0.05*SamplingRate]    # discard boundary effect that might appear at start
    crossers = analyser[zero_crossings]
    beats = zero_crossings[crossers < 0]
    crossdiffs = np.diff(beats)
    dangerous = crossdiffs < 0.15*SamplingRate          # to avoid stupid repetitions
    dangerous = np.nonzero(dangerous)[0]
    if len(dangerous):
        # print 'DANGER', beats[dangerous]
        beats = np.delete(beats, dangerous)

    # 7.1 -------- EXTRA ANTI-FALSE-POSITIVES --------
    store_size = 5
    index_store = 0
    anti_fp = 0.243
    anti_massive = 4
    anti_badset = 3
    reset_count = 0
    cross_storer = np.zeros(store_size)
    crossderivs = np.diff(analyser)
    beats = sorted(list(beats))
    iterator = beats[:]
    evilbeats = []
    for b in iterator:
        cross_med = np.median(cross_storer)
        # print 'info', b, crossderivs[b], anti_fp*cross_med, cross_med, anti_massive*cross_med
        # massive slopes can be eliminated here too (decided not too because it helps in defining Agrafioti windows)
        if crossderivs[b] > anti_fp*cross_med:
            reset_count = 0
            if crossderivs[b] < anti_massive*cross_med or cross_med < 1e-10:
                # print 'store'
                cross_storer[index_store] = crossderivs[b]
                index_store += 1
                if index_store == store_size:
                    index_store = 0
        else:
            reset_count += 1
            # print '\tEVIL SLOPE', b, crossderivs[b], anti_fp*cross_med, reset_count
            evilbeats.append(b)
            beats.remove(b)
        if reset_count >= anti_badset:
            # print '\tRESET'
            reset_count = 0
            cross_storer = np.zeros(store_size)
    beats = np.array(beats, dtype=int)
    evilbeats = np.array(evilbeats)

    # 8 ----------------------------------------- Find the R-peak exactly -----------------------------------------
    search = int(0.15*SamplingRate)
    adjacency = int(0.03*SamplingRate)
    diff_nr = int(0.01*SamplingRate)
    rawbeats = []
    for b in xrange(len(beats)):
        if beats[b]-search < 0:
            rawwindow = filtered[0:beats[b]+search]
            add = 0
        elif beats[b]+search >= len(filtered):
            rawwindow = filtered[beats[b]-search:len(filtered)]
            add = beats[b]-search
        else:
            rawwindow = filtered[beats[b]-search:beats[b]+search]
            add = beats[b]-search
        # ----- get peaks -----
        w_peaks = peakd.sgndiff(Signal=rawwindow)['Peak']
        w_negpeaks = peakd.sgndiff(Signal=rawwindow, a=1)['Peak']
        zerdiffs = np.where(scipy.diff(rawwindow) == 0)[0]
        w_peaks = np.concatenate((w_peaks, zerdiffs))
        w_negpeaks = np.concatenate((w_negpeaks, zerdiffs))

        if up:
            pospeaks = sorted(zip(rawwindow[w_peaks], w_peaks), reverse=True)
        else:
            pospeaks = sorted(zip(rawwindow[w_negpeaks], w_negpeaks))
        # print '\n peaksssss', pospeaks

        try:
            twopeaks = [pospeaks[0]]
        except IndexError:
            twopeaks = []

        # ----------- getting peaks -----------
        for i in xrange(len(pospeaks)-1):
            if abs(pospeaks[0][1] - pospeaks[i+1][1]) > adjacency:
                twopeaks.append(pospeaks[i+1])
                break

        poslen = len(twopeaks)

        if poslen == 2:
            # --- get maximum slope for max peak ---
            if twopeaks[0][1] < diff_nr:
                diff_f = np.diff(rawwindow[0:twopeaks[0][1]+diff_nr])
            elif twopeaks[0][1] + diff_nr >= len(rawwindow):
                diff_f = np.diff(rawwindow[twopeaks[0][1]-diff_nr:len(rawwindow)])
            else:
                diff_f = np.diff(rawwindow[twopeaks[0][1]-diff_nr:twopeaks[0][1]+diff_nr])
            max_f = np.max(np.abs(diff_f))
            # --- get maximum slope for second peak ---
            if twopeaks[1][1] < diff_nr:
                diff_s = np.diff(rawwindow[0:twopeaks[1][1]+diff_nr-1])
            elif twopeaks[1][1] + diff_nr >= len(rawwindow):
                diff_s = np.diff(rawwindow[twopeaks[1][1]-diff_nr+1:len(rawwindow)])
            else:
                diff_s = np.diff(rawwindow[twopeaks[1][1]-diff_nr+1:twopeaks[1][1]+diff_nr-1])
            max_s = np.max(np.abs(diff_s))
            if max_f > max_s:
                # print '\tbigup'
                assignup = [twopeaks[0][0], twopeaks[0][1]]
            else:
                # print '\tsmallup'
                assignup = [twopeaks[1][0], twopeaks[1][1]]
            rawbeats.append(assignup[1] + add)
        elif poslen == 1:
            rawbeats.append(twopeaks[0][1] + add)
        else:
            rawbeats.append(beats[b])
    # 8 ----------------------------------------- END OF POINT 8 -----------------------------------------
    
    # kwrvals
    kwrvals = {'Signal': filtered, 'R': sorted(list(frozenset(rawbeats)))}

    return kwrvals


def hamilton_tst(Signal=None, SamplingRate=1000., init=None, hand=True):
    """

    Algorithm to detect ECG beat indexes.

    Kwargs:
        Signal (array): input filtered ECG signal.

        SamplingRate (float): Sampling frequency (Hz).

        Filter (dict): Filter parameters.

    Kwrvals:
        Signal (array): output filtered signal if Filter is defined.

        R (array): R peak indexes (or instants in seconds if sampling rate is defined).

        init (dict): dict with initial values of some variables

            npeaks (int): number of detected heart beats.

            indexqrs (int): most recent QRS complex index.

            indexnoise (int): most recent noise peak index.

            indexrr (int): most recent R-to-R interval index.

            qrspeakbuffer (array): 8 most recent QRS complexes.

            noisepeakbuffer (array): 8 most recent noise peaks.

            rrinterval (array): 8 most recent R-to-R intervals.

            DT (float): QRS complex detection threshold.

            offset (int): signal start in samples.

    Configurable fields:{"name": "models.hamilton", "config": {"SamplingRate": "1000."}, "inputs": ["Signal", "Filter", "init"], "outputs": ["Signal", "R", "init", "npeaks", "indexqrs", "indexnoise", "indexrr", "qrspeakbuffer", "noisepeakbuffer", "rrinterval", "DT", "offset"]}

    See Also:
        filt

    Notes:

    Example:

    References:
        .. [1] P.S. Hamilton, Open Source ECG Analysis Software Documentation, E.P.Limited
        http://www.eplimited.com/osea13.pdf

    """
    # Check
    if Signal is None:
        raise TypeError("An input signal is needed.")

    # 0.1 - Choose sign of peaks (batch)
    # up = definepeak(Signal, SamplingRate)
    up = 1
    
    RawSignal = np.copy(Signal)

    # 0.2 - Get transformed signal
    UpperCutoff = 16.
    LowerCutoff = 8.
    Order = 4
    Signal = flt.zpdfr(Signal=Signal, SamplingRate=SamplingRate, UpperCutoff=UpperCutoff,
                       LowerCutoff=LowerCutoff, Order=Order)['Signal']
    Signal = abs(np.diff(Signal, 1)*SamplingRate)
    # Signal = flt.smooth(Signal=Signal, Window={'Length': 0.08*SamplingRate, 'Type': 'hamming',
    #                                        'Parameters': None})['Signal']
    
    # moving average
    n = int(0.15 * SamplingRate)
    weights = np.ones(n+1, dtype='float') / float(n+1)
    Signal = flt._filterSignal(weights, [1.], Signal, checkPhase=False)[n:]
    
    # 0.3 - Initialize Buffers
    if init is None:
        init_ecg = 8
        if len(Signal)/1.*SamplingRate < init_ecg:
            init_ecg = int(len(Signal)/1.*SamplingRate)
        qrspeakbuffer = np.zeros(init_ecg)
        noisepeakbuffer = np.zeros(init_ecg)
        rrinterval = SamplingRate*np.ones(init_ecg)
        a, b = 0, int(SamplingRate)
        all_peaks = np.array(peakd.sgndiff(Signal)['Peak'])
        nulldiffs = np.where(np.diff(Signal) == 0)[0]
        all_peaks = np.concatenate((all_peaks, nulldiffs))
        all_peaks = np.array(sorted(frozenset(all_peaks)))
        for i in range(0, init_ecg):
            peaks = peakd.sgndiff(Signal=Signal[a:b])['Peak']
            nulldiffs = np.where(np.diff(Signal[a:b]) == 0)[0]
            peaks = np.concatenate((peaks, nulldiffs))
            peaks = np.array(sorted(frozenset(peaks)))
            try:
                qrspeakbuffer[i] = max(Signal[a:b][peaks])
            except Exception as e:
                print e
            a += int(SamplingRate)
            b += int(SamplingRate)
        # Set Thresholds
        # Detection_Threshold = Average_Noise_Peak + TH*(Average_QRS_Peak-Average_Noise_Peak)
        ANP = np.median(noisepeakbuffer)
        AQRSP = np.median(qrspeakbuffer)
        TH = 0.475  # 0.45 for CVP, 0.475 for  ECGIDDB                                  # 0.3125 - 0.475
        DT = ANP + TH*(AQRSP - ANP)
        init = {}
        init['qrspeakbuffer'] = qrspeakbuffer
        init['noisepeakbuffer'] = noisepeakbuffer
        init['rrinterval'] = rrinterval
        init['indexqrs'] = 0
        init['indexnoise'] = 0
        init['indexrr'] = 0
        init['DT'] = DT
        init['npeaks'] = 0
    beats = []
    twaves = np.array([])

    # ---> Heuristic Thresholds
    lim = int(np.ceil(0.2*SamplingRate))
    elapselim = int(np.ceil(0.36*SamplingRate))
    slopelim = 0.7
    artlim = 2.75
    diff_nr = int(np.ceil(0.01*SamplingRate))
    if diff_nr <= 1:
        diff_nr = 2

    # ---> Peak Detection
    for f in all_peaks:
        # 1 - Checking if f-peak is larger than any peak following or preceding it by less than 200 ms
        peak_cond = np.array((all_peaks > f - lim) * (all_peaks < f + lim) * (all_peaks != f))
        peaks_within = all_peaks[peak_cond]
        if peaks_within.any() and max(Signal[peaks_within]) > Signal[f]:
            # # ---> Update noise buffer
            # init['noisepeakbuffer'][init['indexnoise']] = Signal[f]
            # init['indexnoise'] += 1
            # # print 'NOISE'
            # if init['indexnoise'] == init_ecg:
            #     init['indexnoise'] = 0
            # # print 'TINY'
            continue
        # print 'DT', init['DT']
        if Signal[f] > init['DT']:
            #---------------------FRANCIS---------------------
            # 2 - look for both positive and negative slopes in raw signal
            # if f < diff_nr:
            #     diff_now = np.diff(RawSignal[0:f+diff_nr])
            # elif f + diff_nr >= len(RawSignal):
            #     diff_now = np.diff(RawSignal[f-diff_nr:len(Signal)])
            # else:
            #     diff_now = np.diff(RawSignal[f-diff_nr:f+diff_nr])
            # diff_signer = diff_now[ diff_now > 0]
            # # print 'diff signs:', diff_signer, '\n', diff_now
            # if len(diff_signer) == 0 or len(diff_signer) == len(diff_now):
            #     print 'BASELINE SHIFT'
            #     continue
            #RR INTERVALS
            if init['npeaks'] > 0:
                # 3 - in here we check point 3 of the Hamilton paper (checking whether T-wave or not)
                prev_rpeak = beats[init['npeaks']-1]
                elapsed = f - prev_rpeak
                # print 'elapsed', elapsed
                # if the previous peak was within 360 ms interval
                if elapsed < elapselim:
                    # check current and previous slopes
                    # print '---', f, prev_rpeak, diff_nr, '---'
                    if f < diff_nr:
                        diff_now = np.diff(Signal[0:f+diff_nr])
                    elif f + diff_nr >= len(Signal):
                        diff_now = np.diff(Signal[f-diff_nr:len(Signal)])
                    else:
                        diff_now = np.diff(Signal[f-diff_nr:f+diff_nr])
                    if prev_rpeak < diff_nr:
                        diff_prev = np.diff(Signal[0:prev_rpeak+diff_nr])
                    elif prev_rpeak+diff_nr >= len(Signal):
                        diff_prev = np.diff(Signal[prev_rpeak-diff_nr:len(Signal)])
                    else:
                        diff_prev = np.diff(Signal[prev_rpeak-diff_nr:prev_rpeak+diff_nr])
                    slope_now = np.max(np.abs(diff_now))
                    slope_prev = np.max(np.abs(diff_prev))
                    # print 'diff_now', diff_now
                    # print 'diff_prev', diff_prev
                    # print '\tf -->', f, 'slopes: now -', slope_now, 'prev -', slope_prev, 'lim -', slopelim*slope_prev
                    if slope_now < slopelim*slope_prev:
                        # print 'T-WAVE'
                        twaves = np.concatenate((twaves, [f]))
                        continue
                if not hand or Signal[f] < artlim*np.median(qrspeakbuffer):
                    # print 'GOT IT GOOD', f
                    beats += [int(f)]
                else:
                    continue
                # ---> Update R-R interval
                init['rrinterval'][init['indexrr']] = beats[init['npeaks']]-beats[init['npeaks']-1]
                init['indexrr'] += 1
                if init['indexrr'] == init_ecg:
                    init['indexrr'] = 0
            elif not hand or Signal[f] < artlim*np.median(qrspeakbuffer):
                # print 'GOT IT GOOD', f
                beats += [int(f)]
            else:
                continue
            # ---> Update QRS buffer
            init['npeaks'] += 1
            qrspeakbuffer[init['indexqrs']] = Signal[f]
            init['indexqrs'] += 1
            if init['indexqrs'] == init_ecg:
                init['indexqrs'] = 0
        if Signal[f] <= init['DT']:
            RRM = np.median(init['rrinterval'])
            if len(beats) >= 2:
                elapsed = f-beats[init['npeaks']-1]
                if elapsed >= 1.5*RRM and elapsed > elapselim:
                    prev_rpeak = beats[init['npeaks']-1]
                    rrpeak_cond = np.array( (all_peaks > prev_rpeak + lim) * (all_peaks < f + 1) * (all_peaks != twaves) )
                    peaks_rr = all_peaks[rrpeak_cond]
                    contender = peaks_rr[np.argmax(Signal[peaks_rr])]
                    if Signal[contender] > 0.5*init['DT']:
                        # print 'GOT IT RR', contender, f
                        beats += [int(contender)]
                        # ---> Update R-R interval
                        if init['npeaks'] > 0:
                            init['rrinterval'][init['indexrr']] = beats[init['npeaks']]-beats[init['npeaks']-1]
                            init['indexrr'] += 1
                            if init['indexrr'] == init_ecg:
                                init['indexrr'] = 0
                        # ---> Update QRS buffer
                        init['npeaks'] += 1
                        qrspeakbuffer[init['indexqrs']] = Signal[contender]
                        init['indexqrs'] += 1
                        if init['indexqrs'] == init_ecg:
                            init['indexqrs'] = 0
                    else:
                        # ---> Update noise buffer
                        init['noisepeakbuffer'][init['indexnoise']] = Signal[f]
                        init['indexnoise'] += 1
                        # print 'NOISE'
                        if init['indexnoise'] == init_ecg:
                            init['indexnoise'] = 0
                else:
                    # ---> Update noise buffer
                    init['noisepeakbuffer'][init['indexnoise']] = Signal[f]
                    init['indexnoise'] += 1
                    # print 'NOISE'
                    if init['indexnoise'] == init_ecg:
                        init['indexnoise'] = 0
            else:
                # ---> Update noise buffer
                init['noisepeakbuffer'][init['indexnoise']] = Signal[f]
                init['indexnoise'] += 1
                # print 'NOISE'
                if init['indexnoise'] == init_ecg:
                    init['indexnoise'] = 0

        # --> Update Detection Threshold
        ANP = np.median(init['noisepeakbuffer'])
        AQRSP = np.median(qrspeakbuffer)
        init['DT'] = ANP + TH*(AQRSP - ANP)

    fig = pl.figure()
    mngr = pl.get_current_fig_manager()
    mngr.window.setGeometry(950, 50, 1000, 800)
    ax = fig.add_subplot(111)
    ax.plot(Signal, 'b', label='Signal')
    ax.grid('on')
    ax.axis('tight')
    if np.any(np.array(beats)):
        ax.plot(np.array(beats), Signal[np.array(beats)], 'g^', ms=10, label='rpeak')

    # 8 - Find the R-peak exactly
    search = int(np.ceil(0.15*SamplingRate))
    adjacency = int(np.ceil(0.03*SamplingRate))
    diff_nr = int(np.ceil(0.01*SamplingRate))
    if diff_nr <= 1:
        diff_nr = 2
    rawbeats = []
    for b in xrange(len(beats)):
        if beats[b]-search < 0:
            rawwindow = RawSignal[0:beats[b]+search]
            add = 0
        elif beats[b]+search >= len(RawSignal):
            rawwindow = RawSignal[beats[b]-search:len(RawSignal)]
            add = beats[b]-search
        else:
            rawwindow = RawSignal[beats[b]-search:beats[b]+search]
            add = beats[b]-search
        # ----- get peaks -----
        if up:
            w_peaks = peakd.sgndiff(Signal=rawwindow)['Peak']
        else:
            w_peaks = peakd.sgndiff(Signal=rawwindow, a=1)['Peak']
        zerdiffs = np.where(np.diff(rawwindow) == 0)[0]
        w_peaks = np.concatenate((w_peaks, zerdiffs))
        if up:
            pospeaks = sorted(zip(rawwindow[w_peaks], w_peaks), reverse=True)
        else:
            pospeaks = sorted(zip(rawwindow[w_peaks], w_peaks))
        try:
            twopeaks = [pospeaks[0]]
        except IndexError:
            twopeaks = []

        # ----------- getting peaks -----------
        for i in xrange(len(pospeaks)-1):
            if abs(pospeaks[0][1] - pospeaks[i+1][1]) > adjacency:
                twopeaks.append(pospeaks[i+1])
                break

        poslen = len(twopeaks)
        # print twopeaks, poslen, diff_nr, twopeaks[1][1]-diff_nr+1, twopeaks[1][1]+diff_nr-1

        if poslen == 2:
            # --- get maximum slope for max peak ---
            if twopeaks[0][1] < diff_nr:
                diff_f = np.diff(rawwindow[0:twopeaks[0][1]+diff_nr])
            elif twopeaks[0][1] + diff_nr >= len(rawwindow):
                diff_f = np.diff(rawwindow[twopeaks[0][1]-diff_nr:len(rawwindow)])
            else:
                diff_f = np.diff(rawwindow[twopeaks[0][1]-diff_nr:twopeaks[0][1]+diff_nr])
            max_f = np.max(np.abs(diff_f))
            # --- get maximum slope for second peak ---
            if twopeaks[1][1] < diff_nr:
                diff_s = np.diff(rawwindow[0:twopeaks[1][1]+diff_nr-1])
            elif twopeaks[1][1] + diff_nr >= len(rawwindow):
                diff_s = np.diff(rawwindow[twopeaks[1][1]-diff_nr+1:len(rawwindow)])
            else:
                diff_s = np.diff(rawwindow[twopeaks[1][1]-diff_nr+1:twopeaks[1][1]+diff_nr-1])
            # print diff_s, np.abs(diff_s)
            max_s = np.max(np.abs(diff_s))
            if max_f > max_s:
                # print '\tbigup'
                assignup = [twopeaks[0][0], twopeaks[0][1]]
            else:
                # print '\tsmallup'
                assignup = [twopeaks[1][0], twopeaks[1][1]]
            rawbeats.append(assignup[1] + add)
        elif poslen == 1:
            rawbeats.append(twopeaks[0][1] + add)
        else:
            rawbeats.append(beats[b])

    # kwrvals
    kwrvals = {}
    kwrvals['Signal'] = RawSignal
    kwrvals['init'] = init
    kwrvals['R'] = sorted(list(frozenset(rawbeats)))#/SamplingRate if SamplingRate else beats

    return kwrvals


def hamilton(Signal=None, SamplingRate=1000., Params=None, hand=True):
    """

    Algorithm to detect ECG beat indexes.

    Kwargs:
        Signal (array): input filtered ECG signal.
        
        SamplingRate (float): Sampling frequency (Hz).
        
        Params (dict): Initial conditions:
        
        hand (bool): Signal is obtained from the hands.        

    Kwrvals:
        Signal (array): output filtered signal if Filter is defined.
        
        R (array): R peak indexes (or instants in seconds if sampling rate is defined).
        
        init (dict): dict with initial values of some variables
        
            npeaks (int): number of detected heart beats.
            
            indexqrs (int): most recent QRS complex index.
            
            indexnoise (int): most recent noise peak index.
            
            indexrr (int): most recent R-to-R interval index.
            
            qrspeakbuffer (array): 8 most recent QRS complexes.
            
            noisepeakbuffer (array): 8 most recent noise peaks.
            
            rrinterval (array): 8 most recent R-to-R intervals.
            
            DT (float): QRS complex detection threshold.
            
            offset (int): signal start in samples.

    Configurable fields:{"name": "models.hamilton", "config": {"SamplingRate": "1000."}, "inputs": ["Signal", "Filter", "init"], "outputs": ["Signal", "R", "init", "npeaks", "indexqrs", "indexnoise", "indexrr", "qrspeakbuffer", "noisepeakbuffer", "rrinterval", "DT", "offset"]}

    See Also:
        filt

    Notes:

    Example:

    References:
        .. [1] P.S. Hamilton, Open Source ECG Analysis Software Documentation, E.P.Limited
        http://www.eplimited.com/osea13.pdf
        
    """
    # Check
    if Signal is None:
        raise TypeError, "An input signal is needed."
    
    RawSignal = np.array(Signal, copy=True)

    SamplingRate = float(SamplingRate)
    v1s = int(SamplingRate)
    
    # filter
    UpperCutoff = 25.
    LowerCutoff = 3.
    Order = 4
    Signal = flt.zpdfr(Signal=Signal, SamplingRate=SamplingRate, UpperCutoff=UpperCutoff, LowerCutoff=LowerCutoff, Order=Order)['Signal']
    Signal = np.abs(np.diff(Signal, 1) * SamplingRate)
    Signal = flt.smooth(Signal=Signal, Window={'Length': 0.08*SamplingRate, 'Type': 'hamming', 'Parameters': None})['Signal']
    
    if Params is None:
        Params = {}
        
        init_ecg = 8  # seconds for initialization
        if len(Signal) / SamplingRate < init_ecg:
            init_ecg = int(len(Signal) / SamplingRate)
        qrspeakbuffer = np.zeros(init_ecg)
        noisepeakbuffer = np.zeros(init_ecg)
        peak_idx_test = np.zeros(init_ecg)
        noise_idx = np.zeros(init_ecg)
        rrinterval = SamplingRate * np.ones(init_ecg)
        
        # In order to make an initial estimate, we detect the maximum peaks in eight consecutive 1-second intervals.
        # These eight peaks are used as are initial eight values in the QRS peak buffer, 
        # we set the initial eight noise peaks to 0, and we set the initial threshold accordingly. 
        # We initially set the eight most recent R-to-R intervals to 1 second.
        
        # Init QRS buffer
        a, b = 0, v1s
        all_peaks = peakd.sgndiff(Signal)['Peak']
        # print all_peaks, '\n'
        for i in range(init_ecg):
            peaks = peakd.sgndiff(Signal=Signal[a:b])['Peak']
            
            try:
                ind = np.argmax(Signal[a:b][peaks])
            except ValueError:
                pass
            else:
                # peak amplitude
                qrspeakbuffer[i] = Signal[a:b][peaks][ind]
                # peak location
                peak_idx_test[i] = peaks[ind] + a
            
            a += v1s
            b += v1s
        
        # Set Thresholds                                
        # Detection_Threshold = Average_Noise_Peak + TH*(Average_QRS_Peak-Average_Noise_Peak)
        ANP = np.median(noisepeakbuffer)
        AQRSP = np.median(qrspeakbuffer)
        TH = 0.475  # 0.3125 - 0.475
        DT = ANP + TH * (AQRSP - ANP)
        DT_vec = []
        Params['qrspeakbuffer'] = qrspeakbuffer
        Params['noisepeakbuffer'] = noisepeakbuffer
        Params['rrinterval'] = rrinterval
        Params['indexqrs'] = 0
        Params['indexnoise'] = 0
        Params['indexrr'] = 0
        Params['DT'] = DT
        Params['npeaks'] = 0
        Params['offset'] = 0
    
    beats = []
    
    #Detection Rules
    #1 - ignore all peaks that precede or follow larger peaks by less than 200ms (=0.2*SamplingRate samples)
    lim = int(np.ceil(0.2 * SamplingRate))
    diff_nr = int(np.ceil(0.045 * SamplingRate))
    bpsi, bpe = int(Params['offset']), 0
    
    for f in all_peaks:
        DT_vec += [Params['DT']]
        #1 - Checking if f-peak is larger than any peak following or preceding it by less than 200 ms
        peak_cond = np.array( (all_peaks > f - lim) * (all_peaks < f + lim) * (all_peaks != f) )
        peaks_within = all_peaks[peak_cond]
        if (peaks_within.any() and (max(Signal[peaks_within]) > Signal[f]) ):
            continue
        #4 - If the peak is larger than the detection threshold call it a QRS complex, otherwise call it noise
        if Signal[f] > Params['DT']:
            
            #2 - look for both positive and negative slopes in raw signal
            if f < diff_nr:
                diff_now = np.diff(RawSignal[0:f+diff_nr])
            elif f + diff_nr >= len(RawSignal):
                diff_now = np.diff(RawSignal[f-diff_nr:len(Signal)])
            else:
                diff_now = np.diff(RawSignal[f-diff_nr:f+diff_nr])
            diff_signer = diff_now[ diff_now > 0]
            # print 'diff signs:', diff_signer, '\n', diff_now
            if len(diff_signer) == 0 or len(diff_signer) == len(diff_now):
                # print 'BASELINE SHIFT'
                continue
            #RR INTERVALS
            if Params['npeaks'] > 0:
                #3 - in here we check point 3 of the Hamilton paper
                # that is, we check whether our current peak is a valid R-peak.
                prev_rpeak = beats[Params['npeaks']-1]
                
                elapsed = f - prev_rpeak
                # print 'elapsed', elapsed
                # if the previous peak was within 360 ms interval
                if elapsed < np.ceil(0.36 * SamplingRate):
                    # check current and previous slopes
                    if prev_rpeak < diff_nr:
                        diff_prev = np.diff(RawSignal[0:prev_rpeak+diff_nr])
                    elif prev_rpeak+diff_nr >= len(RawSignal):
                        diff_prev = np.diff(RawSignal[prev_rpeak-diff_nr:len(Signal)])
                    else:
                        diff_prev = np.diff(RawSignal[prev_rpeak-diff_nr:prev_rpeak+diff_nr])
                    # print 'diff_now', diff_now
                    # print 'diff_prev', diff_prev
                    # print prev_rpeak
                    slope_now = max(diff_now)
                    slope_prev = max(diff_prev)
                    
                    # raw_input('_')
                    # Show = True
                    if(slope_now < 0.5 * slope_prev):
                        # if current slope is smaller than half the previous one, then it is a T-wave
                        # print 'T-WAVE'
                        continue
                if not hand or Signal[f] < 3. * np.median(qrspeakbuffer):  # avoid retarded noise peaks
                    # print 'GOT IT GOOD'
                    beats += [int(f)+bpsi]
                else:
                    continue
                
                if bpe == 0:
                    Params['rrinterval'][Params['indexrr']] = beats[Params['npeaks']] - beats[Params['npeaks']-1]
                    Params['indexrr'] += 1
                    if Params['indexrr'] == init_ecg:
                        Params['indexrr'] = 0
                else:
                    if beats[Params['npeaks']] > beats[bpe-1] + 100:
                        Params['rrinterval'][Params['indexrr']] = beats[Params['npeaks']]-beats[Params['npeaks']-1]
                        Params['indexrr'] += 1
                        if Params['indexrr'] == init_ecg:
                            Params['indexrr'] = 0
            
            elif not hand or Signal[f] < 3. * scipy.median(qrspeakbuffer):
                # print 'GOT IT GOOD'
                beats += [int(f)+bpsi]
            else:
                continue
            
            Params['npeaks'] += 1
            qrspeakbuffer[Params['indexqrs']] = Signal[f]
            peak_idx_test[Params['indexqrs']] = f
            Params['indexqrs'] += 1
            if Params['indexqrs'] == init_ecg:
                Params['indexqrs'] = 0
        if Signal[f] <= Params['DT']:  # 4 - not valid
            # 5 - If no QRS has been detected within 1.5 R-to-R intervals, 
            # there was a peak that was larger than half the detection threshold, 
            # and the peak followed the preceding detection by at least 360 ms, 
            # classify that peak as a QRS complex
            tf = f + bpsi
            # RR interval median
            RRM = np.median(Params['rrinterval'])  # initial values are good?
            
            if len(beats) >= 2:
                elapsed = tf - beats[Params['npeaks']-1]
                
                if elapsed >= 1.5 * RRM and elapsed > np.ceil(0.36 * SamplingRate):
                    if Signal[f] > 0.5 * Params['DT']:
                        # print 'GOT IT RR'
                        beats += [int(f) + int(Params['offset'])]
                        #RR INTERVALS
                        if Params['npeaks'] > 0:
                            Params['rrinterval'][Params['indexrr']] = beats[Params['npeaks']] - beats[Params['npeaks'] - 1]
                            Params['indexrr'] += 1
                            if Params['indexrr'] == init_ecg:
                                Params['indexrr'] = 0
                        Params['npeaks'] += 1
                        qrspeakbuffer[Params['indexqrs']] = Signal[f]
                        peak_idx_test[Params['indexqrs']] = f
                        Params['indexqrs'] += 1
                        if Params['indexqrs'] == init_ecg:
                            Params['indexqrs'] = 0
                else:
                    Params['noisepeakbuffer'][Params['indexnoise']] = Signal[f]
                    noise_idx[Params['indexnoise']] = f
                    Params['indexnoise'] += 1
                    # print 'NOISE'
                    if Params['indexnoise'] == init_ecg:
                        Params['indexnoise'] = 0
            else:
                Params['noisepeakbuffer'][Params['indexnoise']] = Signal[f]
                noise_idx[Params['indexnoise']] = f
                Params['indexnoise'] += 1
                # print 'NOISE'
                if Params['indexnoise'] == init_ecg:
                    Params['indexnoise'] = 0
        
        #Update Detection Threshold
        ANP = np.median(Params['noisepeakbuffer'])
        AQRSP = np.median(qrspeakbuffer)
        Params['DT'] = ANP + 0.475 * (AQRSP - ANP)
    
    beats = np.array(beats)
    # kwrvals
    kwrvals = {}
    kwrvals['Params'] = Params
    
    lim = lim
    r_beats = []
    thres_ch = 0.85
    adjacency = 0.05 * SamplingRate
    for i in beats:
        error = [False, False]
        if i-lim < 0:
            window = RawSignal[0:i+lim]
            add = 0
        elif i+lim >= len(RawSignal):
            window = RawSignal[i-lim:len(RawSignal)]
            add = i - lim
        else:
            window = RawSignal[i-lim:i+lim]
            add = i - lim
        # meanval = np.mean(window)
        w_peaks = peakd.sgndiff(Signal=window)['Peak']
        w_negpeaks = peakd.sgndiff(Signal=window, a=1)['Peak']
        zerdiffs = np.where(scipy.diff(window) == 0)[0]
        w_peaks = np.concatenate((w_peaks, zerdiffs))
        w_negpeaks = np.concatenate((w_negpeaks, zerdiffs))
        
        pospeaks = sorted(zip(window[w_peaks], w_peaks), reverse=True)
        negpeaks = sorted(zip(window[w_negpeaks], w_negpeaks))
        
        try:
            twopeaks = [pospeaks[0]]
        except IndexError:
            pass
        try:
            twonegpeaks = [negpeaks[0]]
        except IndexError:
            pass
        
        # ----------- getting positive peaks -----------
        for i in xrange(len(pospeaks) - 1):
            if abs(pospeaks[0][1] - pospeaks[i+1][1]) > adjacency:
                twopeaks.append(pospeaks[i+1])
                break
        try:
            # posdiv = (twopeaks[0][0]-meanval)/(1.*twopeaks[1][0]-meanval)
            posdiv = abs( twopeaks[0][0]-twopeaks[1][0] )
        except IndexError:
            error[0] = True
            
        # ----------- getting negative peaks -----------
        for i in xrange(len(negpeaks)-1):
            if abs(negpeaks[0][1] - negpeaks[i+1][1]) > adjacency:
                twonegpeaks.append(negpeaks[i+1])
                break
        try:
            # negdiv = (twonegpeaks[0][0]-meanval)/(1.*twonegpeaks[1][0]-meanval)
            negdiv = abs( twonegpeaks[0][0]-twonegpeaks[1][0] )
        except IndexError:
            error[1] = True
            
        # ----------- choosing type of R-peak -----------
        if not sum(error):
            if posdiv > thres_ch * negdiv:
                # pos noerr
                r_beats.append(twopeaks[0][1] + add)
            else:
                # neg noerr
                r_beats.append(twonegpeaks[0][1] + add)
        elif sum(error) == 2:
            if abs(twopeaks[0][1]) > abs(twonegpeaks[0][1]):
                # pos allerr
                r_beats.append(twopeaks[0][1] + add)
            else:
                # neg allerr
                r_beats.append(twonegpeaks[0][1] + add)
        elif error[0]: 
            # pos poserr
            r_beats.append(twopeaks[0][1] + add)
        else: 
            # neg negerr
            r_beats.append(twonegpeaks[0][1] + add)
    
    rpeak = sorted(list(frozenset(r_beats)))
    
    # extract heartbeat segments
    segs, rpeak = tools.extractHeartbeats(RawSignal, rpeak, SamplingRate, 0.2, 0.4)
    
    kwrvals['R'] = rpeak
    kwrvals['Segments'] = segs
        
    return kwrvals


def hamilton_old(Signal=None, SamplingRate=1000., Filter=(), init=()):
    """

    Algorithm to detect ECG beat indexes.

    Kwargs:
        Signal (array): input filtered ECG signal.
        
        SamplingRate (float): Sampling frequency (Hz).
        
        Filter (dict): Filter parameters.        

    Kwrvals:
        Signal (array): output filtered signal if Filter is defined.
        
        R (array): R peak indexes (or instants in seconds if sampling rate is defined).
        
        init (dict): dict with initial values of some variables
        
            npeaks (int): number of detected heart beats.
            
            indexqrs (int): most recent QRS complex index.
            
            indexnoise (int): most recent noise peak index.
            
            indexrr (int): most recent R-to-R interval index.
            
            qrspeakbuffer (array): 8 most recent QRS complexes.
            
            noisepeakbuffer (array): 8 most recent noise peaks.
            
            rrinterval (array): 8 most recent R-to-R intervals.
            
            DT (float): QRS complex detection threshold.
            
            offset (int): signal start in samples.

    Configurable fields:{"name": "models.hamilton", "config": {"SamplingRate": "1000."}, "inputs": ["Signal", "Filter", "init"], "outputs": ["Signal", "R", "init", "npeaks", "indexqrs", "indexnoise", "indexrr", "qrspeakbuffer", "noisepeakbuffer", "rrinterval", "DT", "offset"]}

    See Also:
        filt

    Notes:

    Example:

    References:
        .. [1] P.S. Hamilton, Open Source ECG Analysis Software Documentation, E.P.Limited
        http://www.eplimited.com/osea13.pdf
        
    """
    # Check
    if Signal is None:
        raise TypeError, "An input signal is needed."
    if Filter:
        Filter.update({'Signal': Signal})
        if not Filter.has_key('SamplingRate'): Filter.update({'SamplingRate': SamplingRate})
        Signal = flt(**Filter)['Signal']
    # Init
    SamplingRate = float(SamplingRate)
    if not init:
        init_ecg = 8                                                #seconds for initialization
        qrspeakbuffer = scipy.zeros(init_ecg)
        noisepeakbuffer = scipy.zeros(init_ecg)
        rrinterval = SamplingRate*scipy.ones(init_ecg)
        # In order to make an initial estimate, we detect the maximum peaks in eight consecutive 1-second intervals.
        # These eight peaks are used as are initial eight values in the QRS peak buffer, 
        # we set the initial eight noise peaks to 0, and we set the initial threshold accordingly. 
        # We initially set the eight most recent R-to-R intervals to 1 second.
        # Init QRS buffer
        a,b = 0,SamplingRate
        for i in range(0,init_ecg):
            peaks = peakd.sgndiff(Signal=Signal[a:b])['Peak']
            try:
                qrspeakbuffer[i] = max(Signal[a:b][peaks])         #peak amplitude
            except Exception as e:
                print e
            a += SamplingRate
            b += SamplingRate
        # Set Thresholds                                
        # Detection_Threshold = Average_Noise_Peak + TH*(Average_QRS_Peak-Average_Noise_Peak)
        ANP = scipy.mean(noisepeakbuffer)
        AQRSP = scipy.mean(qrspeakbuffer)
        TH = 0.475                                                 #0.3125 - 0.475
        DT = ANP + TH*(AQRSP - ANP)
        init={}
        init['qrspeakbuffer'] = qrspeakbuffer
        init['noisepeakbuffer'] = noisepeakbuffer
        init['rrinterval'] = rrinterval
        init['indexqrs'] = 0
        init['indexnoise'] = 0
        init['indexrr'] = 0
        init['DT'] = DT
        init['npeaks'] = 0
        init['offset'] = 0
    beats = []
    #Detection Rules
    #1 - ignore all peaks that precede or follow larger peaks by less than 200ms (=0.2*SamplingRate samples)
    non_stop = True
    lim,a = 0.2*SamplingRate,0
    b = lim
    bpsi,bpe = int(init['offset']),0
    lim = lim-2
    while(non_stop):
        if(b<len(Signal)):
            aux = peakd.sgndiff(Signal=Signal[a:b])['Peak']
            if(scipy.size(aux)==0):
                a += lim
                b += lim
                continue
            bpeak = scipy.argmax(Signal[a:b][aux])
            bpeak = aux[bpeak]+a
        else:
            non_stop,b = False,len(Signal)
            if(b-a>0):
                aux = peakd.sgndiff(Signal=Signal[a:b])['Peak']
                if(scipy.size(aux)==0):
                    non_stop = False
                    continue
                bpeak = scipy.argmax(Signal[a:b][aux])
                bpeak = aux[bpeak]+a
            else:
                continue
        #2 - If a peak occurs, check to see whether the raw signal contained both positive and negative slopes. 
        #     If not, the peak represents a baseline shift.
        if(True):                                                #2 - valid - UPDATE: does not need (2), does not affect performance
            #4 - If the peak is larger than the detection threshold call it a QRS complex, otherwise call it noise
            if(Signal[bpeak] > init['DT']):                     #4 - valid                
                a = bpeak+lim
                beats+=[int(bpeak)+bpsi]
                #RR INTERVALS
                if(init['npeaks']>0):
                    if(bpe==0):
                        init['rrinterval'][init['indexrr']] = beats[init['npeaks']]-beats[init['npeaks']-1]
                        init['indexrr'] = init['indexrr']+1
                        if(init['indexrr'] == init_ecg): init['indexrr']=0
                    else:
                        if(beats[init['npeaks']] > beats[bpe-1]+100):
                            init['rrinterval'][init['indexrr']] = beats[init['npeaks']]-beats[init['npeaks']-1]
                            init['indexrr'] = init['indexrr']+1
                            if(init['indexrr']==init_ecg): init['indexrr'] = 0
                init['npeaks'] += 1
                qrspeakbuffer[init['indexqrs']] = Signal[bpeak]
                init['indexqrs'] += 1
                if(init['indexqrs']==init_ecg): init['indexqrs'] = 0
            if(Signal[bpeak] <= init['DT']):                    #4 - not valid
                a += lim
                init['noisepeakbuffer'][init['indexnoise']] = Signal[bpeak]
                init['indexnoise'] += 1
                if(init['indexnoise'] == init_ecg): init['indexnoise'] = 0
        else:
            a += lim
        # 5 - If no QRS has been detected within 1.5 R-to-R intervals, 
        # there was a peak that was larger than half the detection threshold, 
        # and the peak followed the preceding detection by at least 360 ms, 
        # classify that peak as a QRS complex
        tf = b+bpsi
        # RR interval mean
        RRM = scipy.mean(init['rrinterval'])
        if(len(beats)>=2):
            if(tf-beats[init['npeaks']-1] > 1.5*RRM):
                initial = beats[init['npeaks']-1]+0.36*SamplingRate-init['offset']
                final = tf+1-init['offset']
                aux = peakd.sgndiff(Signal=Signal[initial:final])['Peak']
                if(len(aux) > 0):
                    bpeak = scipy.argmax(Signal[initial:final][aux])
                    bpeak = aux[bpeak]+initial
                    if(Signal[bpeak] > 0.5*init['DT']):
                        beats+=[int(bpeak)+int(init['offset'])]
                        #RR INTERVALS
                        if(init['npeaks']>0):
                            init['rrinterval'][init['indexrr']] = beats[init['npeaks']]-beats[init['npeaks']-1]
                            init['indexrr'] += 1
                            if(init['indexrr'] == init_ecg): init['indexrr'] = 0
                        init['npeaks'] += 1
                        qrspeakbuffer[init['indexqrs']] = Signal[bpeak]
                        init['indexqrs'] += 1
                        if(init['indexqrs'] == init_ecg): init['indexqrs'] = 0
        b = a+lim
        #Update Detection Threshold
        ANP = scipy.median(init['noisepeakbuffer'])
        AQRSP = scipy.median(qrspeakbuffer)
        init['DT'] = ANP + 0.475*(AQRSP - ANP)
    beats = scipy.array(beats)
    # kwrvals
    kwrvals = {}
    kwrvals['Signal'] = Signal
    kwrvals['R'] = beats#/SamplingRate if SamplingRate else beats
    kwrvals['init'] = init

    return kwrvals


def engzee(Signal=None, SamplingRate=1000., Params=None):
    """
    
    Determine ECG signal information. Adaptation of the Engelse and Zeelenberg by [1].

    Kwargs:
        Signal (array): input ECG signal.
        
        SamplingRate (float): Sampling frequency (Hz).

        Params (dict): Initial conditions:
            CSignal (array): Continuity signal
            
            MM (array): MM threshold
            
            MMidx (int): MM threshold current index
            
            NN (array):  NN threshold
            
            NNidx (int): NN threshold current index
            
            offset (array): last intersection points
            
            prevR (collections.deque): Last R positions
            
            Rminus (float): Segmentation window size to the left of R (default: 200 ms)
            
            Rplus (float): Segmentation window size to the right of R (default: 400 ms)
            
            update (bool): Update flag
    
    Kwrvals:
        R (array): R peak indexes
        
        Segments (array): Extracted segments
        
        HH (float): Heart rate
        
        Params (dict):
            CSignal (array): Continuity signal
            
            MM (array): MM threshold
            
            MMidx (int): MM threshold current index
            
            NN (array):  NN threshold
            
            NNidx (int): NN threshold current index
            
            offset (array): last intersection points
            
            prevR (collections.deque): Last R positions
            
            Rminus (float): Segmentation window size to the left of R (default: 200 ms)
            
            Rplus (float): Segmentation window size to the right of R (default: 400 ms)
            
            update (bool): Update flag
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1] Andre Lourenco, Hugo Silva, Paulo Leite, Renato Lourenco and Ana Fred, 
            REAL TIME ELECTROCARDIOGRAM SEGMENTATION FOR FINGER BASED ECG BIOMETRICS
    
    """
    
    # check inputs
    if Signal is None:
        raise TypeError, "An input signal is needed."
    
    if Params is None:
        Params = {}
    
    # continuity signal
    px = Params.get('CSignal', np.array([]))
    x = np.concatenate((px, Signal))
    
    # Differentiator (1)
    y1 = map(lambda i: x[i] - x[i-4], xrange(4, len(x)))
    
    # Low pass filter (2)
    c = [1, 4, 6, 4, 1, -1, -4, -6, -4, -1]
    y2 = np.array(map(lambda n: np.dot(c, y1[n-9:n+1]), xrange(9, len(y1))))
    
    # previous and default parameters
    changeM = 0.75 * SamplingRate
    Miterate = 1.75 * SamplingRate
    inc = 1
    
    mmth = 0.48 # 0.48 for Santa Marta // maybe change to 0.53 or 0.55 to avoid some FPs!
    MM = Params.get('MM', mmth * max(y2[:Miterate]) * np.ones(3))
    MMidx = Params.get('MMidx', 0)
    Th = np.mean(MM) # 0.6 * max(y2[:Miterate])
    
    mmp = 0.2
    NN = Params.get('NN', mmp * min(y2[:Miterate]) * np.ones(2))
    NNidx = Params.get('NNidx', 0)
    ThNew = np.mean(NN) # 0.7 * min(y2[:Miterate])
    
    update = Params.get('update', False)
    
    offset = Params.get('offset', 0)
    
    Rminus = Params.get('Rminus', 0.2)
    Rplus = Params.get('Rplus', 0.4)
    
    prevR = Params.get('prevR', collections.deque(maxlen=5))
    
    # time windows
    v250ms = int(0.25 * SamplingRate)
    v1200ms = int(1.2 * SamplingRate)
    v180ms = int(0.180 * SamplingRate)
    # due to iteration in windows, sometimes, a peak is not detected when lying in window borders. this sample-advance tries to deal with it
    err_kill = int(SamplingRate / 100.)
    
    # vars in local time reference
    nthfpluss = []
    rpeak = []
    
    # Find nthf+ point
    while True:
        # If a previous intersection was found, continue the analysis from there
        if update:
            if inc * changeM + Miterate < len(y2):
                a = (inc-1) * changeM
                b = inc * changeM + Miterate
                Mnew = mmth * max(y2[a:b])
                Nnew = mmp * min(y2[a:b])
            elif len(y2) - (inc-1) * changeM > 1.5 * SamplingRate :
                a = (inc-1) * changeM
                Mnew = mmth * max(y2[a:])
                Nnew = mmp*min(y2[a:])
            if len(y2)-inc * changeM > Miterate:
                MM[MMidx] = Mnew if Mnew <= 1.5 * MM[MMidx-1] else 1.1 * MM[MMidx-1]
                NN[NNidx] = Nnew if abs(Nnew) <= 1.5 * abs(NN[NNidx-1]) else 1.1 * NN[NNidx-1]
            MMidx = scipy.mod(MMidx+1, len(MM))
            NNidx = scipy.mod(NNidx+1, len(NN))
            Th = np.mean(MM)
            ThNew = np.mean(NN)
            inc += 1
            update = False
        if nthfpluss:
            lastp = nthfpluss[-1] + 1
            if lastp < (inc-1) * changeM:
                lastp = (inc-1) * changeM
            y22 = y2[lastp:inc*changeM+err_kill]
            # find intersection with Th
            try:
                nthfplus = scipy.intersect1d(pl.find(y22>Th), pl.find(y22<Th)-1)[0]
            except IndexError:
                if inc * changeM > len(y2):
                    break
                else:
                    update = True
                    continue
            # adjust index
            # nthfplus += nthfpluss[-1]+1
            nthfplus += int(lastp)
            # if a previous R peak was found: 
            if rpeak:
                # check if intersection is within the 200-1200 ms interval. Modification: 300 ms -> 200 bpm
                if nthfplus - rpeak[-1] > v250ms and nthfplus - rpeak[-1] < v1200ms:
                    pass
                # if new intersection is within the <200ms interval, skip it. Modification: 300 ms -> 200 bpm
                elif nthfplus - rpeak[-1] < v250ms:
                    nthfpluss += [nthfplus]
                    continue
        # no previous intersection, find the first one
        else:
            try:
                nthfplus = int((inc-1)*changeM) + scipy.intersect1d(pl.find(y2[(inc-1)*changeM:inc*changeM+err_kill] > Th), pl.find(y2[(inc-1)*changeM:inc*changeM + err_kill] < Th)-1)[0]
            except IndexError:
                if inc * changeM > len(y2):
                    break
                else:
                    update = True
                    continue                
        nthfpluss += [nthfplus]
        # Define 160ms search region
        windowW = np.arange(nthfplus, nthfplus + v180ms)
        # Check if the condition y2[n] < Th holds for a specified 
        # number of consecutive points (experimentally we found this number to be at least 10 points)"
        i, f = windowW[0], windowW[-1] if windowW[-1] < len(y2) else -1
        hold_points = np.diff(pl.find(y2[i:f] < ThNew))
        cont = 0
        for hp in hold_points:
            if hp == 1:
                cont += 1
                if cont == int(np.ceil(SamplingRate/100.))-1:  # NOTE: NEEDS TO ADAPT TO SAMPLING RATE (-1 is because diff in line 936 eats a sample
                    max_shift = int(np.ceil(SamplingRate/50.))  # looks for X's max a bit to the right
                    if nthfpluss[-1] > max_shift:
                        rpeak += [np.argmax(x[i-max_shift:f]) + i - max_shift]
                    else:
                        rpeak += [np.argmax(x[i:f]) + i]
                    break
            else:
                cont=0
    
    # extract heartbeat segments
    segs, rpeak = tools.extractHeartbeats(x, rpeak, SamplingRate, Rminus, Rplus)
    
    # translate to global time reference
    R = [r + offset for r in rpeak]
    
    # update continuity signal and offset
    try:
        lr = rpeak[-1] + 1
    except IndexError:
        px = x
    else:
        px = x[lr:]
        offset += lr
    
    # compute heart rate
    prevR.extend(R)
    dR = np.diff(prevR)
    if len(dR) > 0:
        heart_rate = int(np.mean(SamplingRate * (60. / dR)))  # bpm
    else:
        heart_rate = 0
    
    # Output
    kwrvals = {}
    kwrvals['R'] = np.array(R)
    kwrvals['Segments'] = segs
    kwrvals['HR'] = heart_rate
    kwrvals['Params'] = {
                        'CSignal': px,
                        'MM': MM,
                        'MMidx': MMidx,
                        'NN': NN,
                        'NNidx': NNidx,
                        'offset': offset,
                        'prevR': prevR,
                        'Rminus': Rminus,
                        'Rplus': Rplus,
                        'update': update,
                        }
    return kwrvals


def batch_engzee(Signal=None, SamplingRate=1000.0, debug=False, IF=True):
    """
    
    
    Kwargs:
    
    
    Kwrvals:
    
            
    Configurable fields:{"name": "models.batch_engzee", "config": {"SamplingRate": "1000.0"}, "inputs": ["Signal"], "outputs": []}
    
    See Also:
    
    
    Notes:
    
        
    Example:
    
    
    References:
        .. [1] 
        
    """
    # Fir filter
    if IF:
        fn = 301
        lpc, hpc = 5., 20.
        bfir1 = ss.firwin(fn,[2*lpc/SamplingRate, 2*hpc/SamplingRate], pass_zero=False)
        yfir = ss.lfilter(bfir1, [1], Signal)
        yfir = yfir[fn:]
        FIRParams = {'fn': fn, 'lp': lpc, 'hp': hpc}
    else:
        yfir = Signal
        FIRParams = None
    # QRS Detection
    mrkr = 0
    win = 1.5*SamplingRate
    mrkrend = win
    res = engzee_old(Signal=yfir[:win]-1.*scipy.mean(yfir[:win]), SamplingRate=SamplingRate, initialfilter=False)
    dlen = len(yfir)
    while ( mrkrend < dlen ):    
        mrkrend += win
        mrkr = res['Params']['offset'][-1]
        res = engzee_old(Signal=yfir[mrkr:mrkrend]-1.*scipy.mean(yfir[mrkr:mrkrend]), SamplingRate=SamplingRate, initialfilter=False, Params=res['Params'])
        if debug:
            if raw_input('_')=='q':
                break
    # ECG Segmentation
    length = len(yfir)
    segs = []
    for r in res['R']:
        a = r - int(0.2 * SamplingRate) # 200 ms before R
        if a < 0:
            continue
        b = r + int(0.4 * SamplingRate) # 400 ms after R
        if b > length:
            continue
        segs.append(yfir[a:b])

    
    kwrvals = {}
    kwrvals['RawSignal'] = Signal
    kwrvals['SamplingRate'] = SamplingRate
    kwrvals['R'] = res['R']
    kwrvals['Segments'] = segs
    kwrvals['FilteredSignal'] = yfir
    kwrvals['FIRParams'] = FIRParams
    
    return kwrvals


def engzee_old(Signal=None, SamplingRate=1000., initialfilter=True, Filter={}, Params={}):
    """
    ### HAS PROBLEM WITH R PEAKS LOCATION IN ONLINE USE-CASE
    
    Determine ECG signal information. Adaptation of the Engelse and Zeelenberg by [1].

    Kwargs:
        Signal (array): input ECG signal.
        
        SamplingRate (float): Sampling frequency (Hz).
        
        Filter (dict): Filter parameters.

        Params (dict): Initial conditions
        
            Segments (list): list of ECG segments
            
            RawSegments (list): list of Raw ECG segments
        
            MM (array): MM threshold
            
            MMidx (int): MM threshold current index
            
            offset (array): last intersection points
            
            offsetidx (int): offset current index
            
            rpeak (array): R indexes
            
            nthfpluss (array): nthfplus intersection points
            
            Rminus (int): Segmentation window size to the left of R (default: 200 ms)
            
            Rplus (int): Segmentation window size to the right of R (default: 400 ms)
            

    Kwrvals:
        R (array): R peak indexes
        
        Params (dict):
            
            Segments (list): list of ECG segments
            
            RawSegments (list): list of Raw ECG segments
        
            MM (array): MM threshold
            
            MMidx (int): MM threshold current index
            
            offset (array): last intersection points
            
            offsetidx (int): offset current index
            
            rpeak (array): R indexes
            
            nthfpluss (array): nthfplus intersection points
            
            Rminus (int): Segmentation window size to the left of R (default: 200 ms)
            
            Rplus (int): Segmentation window size to the right of R (default: 400 ms)
            
    Configurable fields:{"name": "models.engzee", "config": {"SamplingRate": "1000.", "initialfilter": "True"}, "inputs": ["Signal", "Filter", "Params"], "outputs": ["R", "Params", "Segments", "RawSegments", "MM", "MMidx", "offset", "offsetidx", "rpeak", "nthfpluss", "Rminus", "Rplus"]}

    See Also:


    Notes:
        1. Tested for chest ECGs: paper thresholds, no initial filtering
        
        2. Finger/palm ECGs: there's an initial filtering and some parameters were modified
        
    Example:
        # Data filename
        
        fname = ...
        
        # Load data
        
        dataecg = plux.loadbpf(fname, usecols=(3,))
        
        # Init
        
        mrkr = 0
        
        SamplingRate = dataecg.header['SamplingFrequency']
        
        win = 3*SamplingRate
        
        mrkrend = win
        
        # ECG algorithm, segmentation: R-200ms to R+400ms
        
        res = ecgmodule.models.engzee(Signal=dataecg[:win], SamplingRate=SamplingRate, Params={'Rminus': int(0.2*SamplingRate), 'Rplus': int(0.4*SamplingRate)})
        
        dlen = len(dataecg)
        
        while ( mrkrend < dlen ):
        
            mrkr = res['Params']['offset'][-1]
            
            res = ecgmodule.models.engzee(Signal=dataecg[mrkr:mrkrend], SamplingRate=SamplingRate, Params=res['Params'])
            
        ECGSegments = res['Params']['Segments']

    References:
        .. [1] Andre Lourenco, Hugo Silva, Paulo Leite, Renato Lourenco and Ana Fred, 
        REAL TIME ELECTROCARDIOGRAM SEGMENTATION FOR FINGER BASED ECG BIOMETRICS
    """
    # fECGpattern = scipy.loadtxt('c:/work/pattern.ecg')
    # Check
    if Signal is None:
        raise TypeError, "An input signal is needed."
    if Filter:
        Filter.update({'Signal': Signal})
        if not Filter.has_key('SamplingRate'): Filter.update({'SamplingRate': SamplingRate})
        Signal = flt(**Filter)['Signal']        
    Method = 'online'
    # Initial filtering (for hands ECG)
    RawSignal = Signal
    x = Signal
    if initialfilter:
        x = flt.zpdfr(Signal=Signal, SamplingRate=1000., UpperCutoff=20., LowerCutoff=1., Order=6)['Signal']
        x = flt.zpdfr(Signal=x, SamplingRate=1000., LowerCutoff=5., Order=7)['Signal']
        x = flt.smooth(Signal=x, Window={'Length': 28})['Signal']
    # Differentiator (1)
    y1 = map(lambda i: x[i]-x[i-4], range(4,len(x)))
    # Low pass filter (2)
    c = [1,4,6,4,1]
    y2 = scipy.array(map(lambda n: scipy.dot(c,y1[n-4:n+1]), range(4,len(y1))))
    # Define threshold
    mmth=0.6#0.6
    MM = Params['MM'] if Params.has_key('MM') else mmth*max(y2)*scipy.ones(2)
    MMidx = Params['MMidx'] if Params.has_key('MMidx') else 0
    slope = scipy.linspace(1.0,0.8,1000)                                         # paper: 1.0-0.6, modification: 1.0-0.8
    mmp = 0.7
    Th = mmp*scipy.mean(MM) if Method is 'online' else 0.6*max(y2)            # paper: mmp=0.2, modification: mmp=0.7
    ThNew = -Th if Method is 'online' else 0.7*min(y2)
    # Init variables
    v200ms = 0.2*SamplingRate
    v300ms = 0.3*SamplingRate
    v1200ms = 1.2*SamplingRate
    v160ms = int(0.160*SamplingRate)
    offset = Params['offset'] if Params.has_key('offset') else [0]
    offsetidx = Params['offsetidx'] if Params.has_key('offsetidx') else 1
    rpeak = Params['rpeak'] if Params.has_key('rpeak') else []
    nthfpluss = Params['nthfpluss'] if Params.has_key('nthfpluss') else []
    Segments = Params['Segments'] if Params.has_key('Segments') else []
    RawSegments = Params['RawSegments'] if Params.has_key('RawSegments') else []
    Rminus = Params['Rminus'] if Params.has_key('Rminus') else int(0.2*SamplingRate)
    Rplus = Params['Rplus'] if Params.has_key('Rplus') else int(0.4*SamplingRate)
    nits = Params['nits'] if Params.has_key('nits') else 0
    nbeats = Params['nbeats'] if Params.has_key('nbeats') else 0
    rridxs = Params['rridxs'] if Params.has_key('rridxs') else []
    seg_size = Rplus - Rminus
    if seg_size <= 0:
        print "Segment size error: Rminus >= Rplus"
        return -1
    # Find nthf+ point
    while True:
        # If a previous intersection was found, continue the analysis from there
        if nthfpluss:
            lastp = nthfpluss[-1]+1-offset[-1]
            y22 = y2[lastp:]
            # find intersection with Th
            try:
                nthfplus = scipy.intersect1d(pl.find(y22>Th),pl.find(y22<Th)-1)[0]
            except IndexError:
                break
            # adjust index
            nthfplus += nthfpluss[-1]+1
            # if a previous R peak was found: 
            if rpeak and Method is 'online':
                # check if intersection is within the 200-1200 ms interval. Modification: 300 ms -> 200 bpm
                if nthfplus - rpeak[-1] > v300ms and nthfplus - rpeak[-1] < v1200ms:
                    pass
                    # decrease MM buffer at a low slope
                    # MM[MMidx] = MM[MMidx]*slope[nthfplus - rpeak[-1] - int(v300ms)]
                    # MMidx = scipy.mod(MMidx+1,len(MM))
                    # Th = mmp*scipy.mean(MM)
                    # ThNew = -Th
                # if new intersection is within the <200ms interval, skip it. Modification: 300 ms -> 200 bpm
                elif nthfplus - rpeak[-1] < v300ms:
                    nthfpluss += [nthfplus]
                    continue
        # no previous intersection, find the first one
        else:
            try:
                nthfplus = scipy.intersect1d(pl.find(y2>Th),pl.find(y2<Th)-1)[0]
            except IndexError:
                break
        nthfpluss += [nthfplus] # plot
    # Define 160ms search region
        windowW = scipy.arange(nthfplus, nthfplus+v160ms)
        windowW -= offset[-1]
        # "Check if the condition y2[n] < Th holds for a specified 
        # number of consecutive points (experimentally we found this number to be at least 10 points)"
        i,f = windowW[0], windowW[-1] if windowW[-1]<len(y2) else -1
        hold_points = scipy.diff(pl.find(y2[i:f]<ThNew))
        cont=0
        for hp in hold_points:
            if hp == 1:
                cont+=1
                if cont == int(scipy.ceil(SamplingRate/100.)):                                # NOTE: NEEDS TO ADAPT TO SAMPLING RATE
                    # "Upon finding a candidate R peak, the original signal,
                    # x[n] is scanned inside the obtained window, 
                    # and the peak is determined as the time instant corresponding
                    # to the highest amplitude signal"
                    if all(MM>max(x[i:f])):
                        rpeak += [scipy.argmax(x[i:f])+i+offset[-1]]
                        a,b = Rminus, Rplus
                        cur_r_peak = rpeak[-1]
                    # print max(x[i:f]), scipy.mean(x), max(x[i:f])-min(x[i:f]), scipy.std(x), abs(scipy.mean(scipy.diff(x)))
                    # if cur_r_peak-a-offset[offsetidx-1] >= 0 and cur_r_peak+b-offset[offsetidx-1] < len(x):
                    # Segments.append(x[cur_r_peak-a-offset[offsetidx-1]:cur_r_peak+b-offset[offsetidx-1]])
                    # RawSegments.append(RawSignal[cur_r_peak-a-offset[offsetidx-1]:cur_r_peak+b-offset[offsetidx-1]])
                    # Neighborhood to the left
                    # Neighborhood to the right
                    break
            else:
                cont=0
            # # # print hp, cont
        # # # print ""
        # # #-----
        # pl.figure(787)
        # pl.figure(787).clf()
        # pl.plot(x, 'b', label='x')
        # pl.plot(y2, 'g', label='y2')
        # pl.grid('on')        
        # pl.axis('tight')
        # pl.plot(range(len(y2)),Th*scipy.ones(len(y2)),'r--', label='Th')
        # pl.plot(range(len(y2)),ThNew*scipy.ones(len(y2)),'k--', label='ThNew')
        # pl.text(1000,900, str(Th))
        # pl.plot(scipy.array(nthfpluss[nits:])-offset[-1], y2[scipy.array(nthfpluss[nits:])-offset[-1]],'ko', label='nthfplus')
        # pl.plot(scipy.arange(i,i+len(y2[i:f])), y2[i:f], 'm')
        # if(scipy.any(scipy.array(rpeak[nbeats:])-offset[-1])): pl.plot(scipy.array(rpeak[nbeats:])-offset[-1], x[scipy.array(rpeak[nbeats:])-offset[-1]], 'r^', ms=10, label='rpeak')
        # ax = pl.figure(787).add_subplot(111)
        # # ax.axes.yaxis.set_visible(False)
        # # ax.axes.xaxis.set_visible(False)
        # pl.show()
        # if raw_input('_')=='q': break
        
    # pl.figure(987).clf()
    a,b = Rminus, Rplus
    i = nbeats-1
    for r in rpeak[nbeats:]:
        i += 1
        try:
            p = r-offset[-1]
            if p-a >= 0 and p+b < len(x):
                seg = x[p-a:p+b]
                Segments.append(seg)
                RawSegments.append(RawSignal[p-a:p+b])
                rridxs.append(i)
        except Exception as e:
            print e
    # pl.legend()
        #-----
    # Update buffer MM
    Mnew = mmth*max(y2)
    MM[MMidx] =  Mnew if Mnew <= 1.5*MM[MMidx-1] else 1.1*MM[MMidx-1]
    # print Mnew, MM[MMidx]
    MMidx = scipy.mod(MMidx+1,len(MM))
    offset.append(nthfpluss[-1])
    offsetidx += 1
    # Output
    kwrvals = {}
    kwrvals['R'] = scipy.array(rpeak)
    
    try:
        last_pks = kwrvals['R'][-10:]
        heart_rate = int(scipy.mean(SamplingRate*(60./scipy.diff(last_pks)))) # bpm
    except Exception as e:
        # print e
        heart_rate = 0
        
    kwrvals['Params'] = {
                        'Segments': Segments,
                        'RawSegments': RawSegments,
                        'MM': MM,
                        'MMidx': MMidx,
                        'offset': offset,
                        'offsetidx': offsetidx,
                        'rpeak': rpeak,
                        'nthfpluss': nthfpluss,
                        'Rminus': Rminus,
                        'Rplus': Rplus,
                        'HR': heart_rate,
                        'nits': len(nthfpluss),
                        'nbeats': len(rpeak),
                        'rridxs': rridxs
                        }
    return kwrvals


def engzee_incomplete(Signal=None, SamplingRate=1000., Filter=None, Params=None, initialFilter=False):
    """
    
    --- REVIEW DOCS ---
    
    Determine ECG signal information. Adaptation of the Engelse and Zeelenberg by [1].

    Kwargs:
        Signal (array): input ECG signal.
        
        SamplingRate (float): Sampling frequency (Hz).
        
        Filter (dict): Filter parameters.

        Params (dict): Initial conditions
        
            Segments (list): list of ECG segments
            
            RawSegments (list): list of Raw ECG segments
        
            MM (array): MM threshold
            
            MMidx (int): MM threshold current index
            
            offset (array): last intersection points
            
            offsetidx (int): offset current index
            
            rpeak (array): R indexes
            
            nthfpluss (array): nthfplus intersection points
            
            Rminus (int): Segmentation window size to the left of R (default: 200 ms)
            
            Rplus (int): Segmentation window size to the right of R (default: 400 ms)
        
        initialFilter (bool): Apply hands ECG filter (default: False).
        

    Kwrvals:
        R (array): R peak indexes
        
        Params (dict):
            
            Segments (list): list of ECG segments
            
            RawSegments (list): list of Raw ECG segments
        
            MM (array): MM threshold
            
            MMidx (int): MM threshold current index
            
            offset (array): last intersection points
            
            offsetidx (int): offset current index
            
            rpeak (array): R indexes
            
            nthfpluss (array): nthfplus intersection points
            
            Rminus (int): Segmentation window size to the left of R (default: 200 ms)
            
            Rplus (int): Segmentation window size to the right of R (default: 400 ms)
            
    Configurable fields:{"name": "models.engzee", "config": {"SamplingRate": "1000.", "initialfilter": "True"}, "inputs": ["Signal", "Filter", "Params"], "outputs": ["R", "Params", "Segments", "RawSegments", "MM", "MMidx", "offset", "offsetidx", "rpeak", "nthfpluss", "Rminus", "Rplus"]}

    See Also:


    Notes:
        1. Tested for chest ECGs: paper thresholds, no initial filtering
        
        2. Finger/palm ECGs: there's an initial filtering and some parameters were modified
        
    Example:
        # Data filename
        
        fname = ...
        
        # Load data
        
        dataecg = plux.loadbpf(fname, usecols=(3,))
        
        # Init
        
        mrkr = 0
        
        SamplingRate = dataecg.header['SamplingFrequency']
        
        win = 3*SamplingRate
        
        mrkrend = win
        
        # ECG algorithm, segmentation: R-200ms to R+400ms
        
        res = ecgmodule.models.engzee(Signal=dataecg[:win], SamplingRate=SamplingRate, Params={'Rminus': int(0.2*SamplingRate), 'Rplus': int(0.4*SamplingRate)})
        
        dlen = len(dataecg)
        
        while ( mrkrend < dlen ):
        
            mrkr = res['Params']['offset'][-1]
            
            res = ecgmodule.models.engzee(Signal=dataecg[mrkr:mrkrend], SamplingRate=SamplingRate, Params=res['Params'])
            
        ECGSegments = res['Params']['Segments']

    References:
        .. [1] Andre Lourenco, Hugo Silva, Paulo Leite, Renato Lourenco and Ana Fred, 
        REAL TIME ELECTROCARDIOGRAM SEGMENTATION FOR FINGER BASED ECG BIOMETRICS
    """
    
    # Check inputs
    if Signal is None:
        raise TypeError, "An input signal is needed."
    
    if Params is None:
        Params = {}
    
    # filter
    if Filter:
        Filter.update({'Signal': Signal})
        if not Filter.has_key('SamplingRate'): Filter.update({'SamplingRate': SamplingRate})
        Signal = flt(**Filter)['Signal']        
    
    # Initial filtering (for hands ECG)
    x = Signal
    if initialFilter:
        x = flt.zpdfr(Signal=Signal, SamplingRate=1000., UpperCutoff=20., LowerCutoff=1., Order=6)['Signal']
        x = flt.zpdfr(Signal=x, SamplingRate=1000., LowerCutoff=5., Order=7)['Signal']
        x = flt.smooth(Signal=x, Window={'Length': 28})['Signal']
    
    # x = Signal
    # continuity signal
    px = Params['CSignal'] if Params.has_key('CSignal') else scipy.array([])
    x = scipy.concatenate((px, x))
    
    # Differentiator (1)
    y1 = map(lambda i: x[i] - x[i-4], xrange(4, len(x)))
    
    # Low pass filter (2)
    c = [1, 4, 6, 4, 1]
    y2 = scipy.array(map(lambda n: scipy.dot(c, y1[n-4:n+1]), xrange(4, len(y1))))
    
    # previous and default parameters
    mmth = 0.6
    MM = Params['MM'] if Params.has_key('MM') else mmth * max(y2) * scipy.ones(2)
    MMidx = Params['MMidx'] if Params.has_key('MMidx') else 0
    
    mmp = 0.7
    Th = mmp * scipy.mean(MM)
    ThNew = -Th
    
    offset = Params['offset'] if Params.has_key('offset') else 0
    
    Rminus = Params['Rminus'] if Params.has_key('Rminus') and Params['Rminus'] > 0 else 0.2
    Rplus = Params['Rplus'] if Params.has_key('Rplus') and Params['Rplus'] > 0 else 0.4
    
    prevR = Params['prevR'] if Params.has_key('prevR') else collections.deque(maxlen=5)
    
    # time windows
    # v200ms = int(0.2 * SamplingRate)
    v300ms = int(0.3 * SamplingRate)
    v1200ms = int(1.2 * SamplingRate)
    v160ms = int(0.16 * SamplingRate)
    
    # find nthf+ point
    nthfpluss = []
    rpeak = [] # local time reference
    while True:
        # If a previous intersection was found, continue the analysis from there
        if nthfpluss:
            lastp = nthfpluss[-1] + 1
            y22 = y2[lastp:]
            # find intersection with Th
            try:
                nthfplus = scipy.intersect1d(pl.find(y22 > Th), pl.find(y22 < Th) - 1)[0]
            except IndexError:
                break
            # adjust index
            nthfplus += nthfpluss[-1] + 1
            # if a previous R peak was found: 
            if rpeak:
                # check if intersection is within the 200-1200 ms interval. Modification: 300 ms -> 200 bpm
                if nthfplus - rpeak[-1] > v300ms and nthfplus - rpeak[-1] < v1200ms:
                    pass
                # if new intersection is within the <200ms interval, skip it. Modification: 300 ms -> 200 bpm
                elif nthfplus - rpeak[-1] < v300ms:
                    nthfpluss += [nthfplus]
                    continue
        # no previous intersection, find the first one
        else:
            try:
                nthfplus = scipy.intersect1d(pl.find(y2 > Th), pl.find(y2 < Th) - 1)[0]
            except IndexError:
                break
        nthfpluss += [nthfplus]
        
        # Define 160 ms search region
        windowW = scipy.arange(nthfplus, nthfplus + v160ms)
        # "Check if the condition y2[n] < Th holds for a specified 
        # number of consecutive points (experimentally we found this number to be at least 10 points)"
        i,f = windowW[0], windowW[-1] if windowW[-1] < len(y2) else -1
        hold_points = scipy.diff(pl.find(y2[i:f] < ThNew))
        cont=0
        for hp in hold_points:
            if hp == 1:
                cont += 1
                if cont == int(scipy.ceil(SamplingRate / 100.)): # NOTE: NEEDS TO ADAPT TO SAMPLING RATE
                    # "Upon finding a candidate R peak, the original signal,
                    # x[n] is scanned inside the obtained window, 
                    # and the peak is determined as the time instant corresponding
                    # to the highest amplitude signal"
                    if all(MM > max(x[i:f])):
                        rpeak += [scipy.argmax(x[i:f]) + i]
                    break
            else:
                cont = 0
    
    # extract heartbeat segments
    rpeak_old = rpeak
    segs, rpeak = tools.extractHeartbeats(x, rpeak, SamplingRate, Rminus, Rplus)
    
    if len(rpeak) == 0:
        # no peaks detected
        R = []
        px = x
    else:
        # translate to global time reference
        R = [r + offset for r in rpeak]
        
        # update continuity signal and offset
        lr = rpeak[-1] + 1
        px = x[lr:]
        offset += lr
        
        prevR.extend(R)
    
    # compute average heart rate
    if len(prevR) < 2:
        heart_rate = 0
    else:
        heart_rate = int(scipy.mean(SamplingRate * (60. / scipy.diff(prevR)))) # bpm
    
    # Update buffer MM
    Mnew = mmth * max(y2)
    MM[MMidx] =  Mnew if Mnew <= 1.5 * MM[MMidx-1] else 1.1 * MM[MMidx-1]
    MMidx = scipy.mod(MMidx+1,len(MM))
    
    # output
    kwrvals = {}
    kwrvals['R'] = scipy.array(R)
    kwrvals['Segments'] = segs
    kwrvals['Params'] = {
                        # 'Segments': segs,
                        # 'RawSegments': RawSegments,
                        'CSignal': px,
                        'MM': MM,
                        'MMidx': MMidx,
                        'offset': offset,
                        'rpeak': rpeak,
                        'rpeak_old': rpeak_old,
                        'Rminus': Rminus,
                        'Rplus': Rplus,
                        'prevR': prevR,
                        'HR': heart_rate,
                        'nits': len(nthfpluss),
                        'nbeats': len(rpeak),
                        }
    return kwrvals


def batch_christov(Signal=None, SamplingRate=1000.0, debug=False, IF=True):
    if IF:
        fn = 301
        lpc, hpc = 1., 20.
        bfir1 = ss.firwin(fn,[2*lpc/SamplingRate, 2*hpc/SamplingRate], pass_zero=False)
        yfir = ss.lfilter(bfir1, [1], Signal)
        yfir = yfir[fn:]
        FIRParams = {'fn': fn, 'lp': lpc, 'hp': hpc}
    else:
        yfir = Signal
        FIRParams = None
    
    rpeaks, win, ld = [], 1.5*SamplingRate, len(yfir)
    b = win
    res = christov(Signal=yfir[b-win:b]-1.*scipy.mean(yfir[b-win:b]), SamplingRate=SamplingRate, Show=False)
    map(lambda i: rpeaks.append(i+b-win), res['R'])
    b += win
    while b<=ld:
        res = christov(Signal=yfir[b-win:b]-1.*scipy.mean(yfir[b-win:b]), SamplingRate=SamplingRate, Show=False)
        map(lambda i: rpeaks.append(i+b-win), res['R'])
        b += win

    try:
        res = christov(Signal=yfir[-win:-1]-1.*scipy.mean(yfir[-win:-1]), SamplingRate=SamplingRate, Show=False)
        map(lambda i: rpeaks.append(i+ld-win), res['R'])
    except:
        pass
    
    # ECG Segmentation
    length = len(yfir)
    segs = []
    for r in rpeaks:
        a = r - int(0.2 * SamplingRate) # 200 ms before R
        if a < 0:
            continue
        b = r + int(0.4 * SamplingRate) # 400 ms after R
        if b > length:
            continue
        segs.append(yfir[a:b])
    kwrvals = {}
    kwrvals['RawSignal'] = Signal
    kwrvals['SamplingRate'] = SamplingRate
    kwrvals['R'] = rpeaks
    kwrvals['Segments'] = segs
    kwrvals['FilteredSignal'] = yfir
    kwrvals['FIRParams'] = FIRParams
    return kwrvals

def christov(Signal=None, SamplingRate=1000., Filter={}, Params={}, Show=False):
    """

    Determine ECG signal information.

    Kwargs:
        Signal (array): input ECG signal.
        
        SamplingRate (float): Sampling frequency (Hz).
        
        Filter (dict): Filter parameters.

        Params (dict): Initial conditions.        

    Kwrvals:

    Configurable fields:{"name": "models.christov", "config": {"SamplingRate": "1000.", "Show": "False"}, "inputs": ["Signal", "Filter", "Params"], "outputs": []}

    See Also:

    Notes:

    Example:

    References:
        .. [1]    Ivaylo I Christov, Real time electrocardiogram QRS detection using combined adaptive threshold,
                BioMedical Engineering OnLine 2004, 3:28
                This article is available from: http://www.biomedical-engineering-online.com/content/3/1/28
                2004 Christov; licensee BioMed Central Ltd.
    """    
    # Check
    if Signal is None:
        raise TypeError, "An input signal is needed."
    if Filter:
        Filter.update({'Signal': Signal})
        if not Filter.has_key('SamplingRate'): Filter.update({'SamplingRate': SamplingRate})
        Signal = flt(**Filter)['Signal']    
    # Pre-processing 
    # 1. Moving averaging filter for power-line interference suppression:
    # averages samples in one period of the powerline
    # interference frequency with a first zero at this frequency.
    X = ss.filtfilt(scipy.ones(SamplingRate/50.)/50.,[1],Signal)
    # 2. Moving averaging of samples in 28 ms interval for electromyogram
    # noise suppression a filter with first zero at about 35 Hz.    
    X = ss.filtfilt(scipy.ones(SamplingRate/35.)/35.,[1],X)    
    X = flt.zpdfr(Signal=X, SamplingRate=SamplingRate, UpperCutoff=40., LowerCutoff=9., Order=7.)['Signal']
    k, Y, L = 1, [], len(X)
    for n in range(k+1, L-k): Y.append(X[n]**2-X[n-k]*X[n+k])
    Y = scipy.array(Y)
    Y[pl.find(Y < 0)] = 0
    # Complex lead
    # Y = abs(scipy.diff(X)) # 1-lead
    # 3. Moving averaging of a complex lead (the sintesis is
    # explained in the next section) in 40 ms intervals a filter
    # with first zero at about 25 Hz. It is suppressing the noise
    # magnified by the differentiation procedure used in the
    # process of the complex lead sintesis.        
    Y = ss.lfilter(scipy.ones(SamplingRate/25.)/25.,[1],Y)
    # Init    
    M_th = 0.4 # paper is 0.6
    MM = Params['M'] if Params.has_key('M') else M_th*max(Y[:5*SamplingRate])*scipy.ones(5) # 0.6
    MMidx = Params['MMidx'] if Params.has_key('MMidx') else 0
    M = scipy.mean(MM)
    slope = scipy.linspace(1.0,0.6,int(SamplingRate))
    slopeR = scipy.linspace(1.0,0.6*1.4,int(SamplingRate))
    Rdec = 0#(slopeR[1]-slopeR[0])/1000.
    R = Params['Rth'] if Params.has_key('Rth') else 0
    RR = Params['RR'] if Params.has_key('RR') else scipy.zeros(5)
    RRidx = Params['RRidx'] if Params.has_key('RRidx') else 0
    Rm = scipy.mean(RR)
    QRS = Params['QRS'] if Params.has_key('QRS') else []
    Rpeak = Params['R'] if Params.has_key('R') else []
    offset = Params['offset'] if Params.has_key('offset') else 0
    current_sample = 0
    skip = False
    if(Show): M_hist,F_hist, R_hist, MFR_hist = [], [], [], [] # to plot
    v50ms = int(0.050*SamplingRate)
    v300ms = int(0.300*SamplingRate)
    v350ms = int(0.350*SamplingRate)
    v200ms = 0.2*SamplingRate
    v1200ms = 1.2*SamplingRate
    F =  scipy.mean(Y[:v350ms])
    if Params.has_key('Y_350'): Y_temp = scipy.concatenate((Params['Y_350'], Y))
    nbeats = Params['nbeats'] if Params.has_key('nbeats') else 0
    # Go through each sample
    while current_sample < len(Y):
        if QRS:
            # No detection is allowed 200 ms after the current one. In
            # the interval QRS to QRS+200ms a new value of M5 is calculated: newM5 = 0.6*max(Yi)
            if current_sample <= QRS[-1]+v200ms-offset:
                Mnew = M_th*max(Y[QRS[-1]-offset:QRS[-1]+v200ms-offset]) # 0.6
                # The estimated newM5 value can become quite high, if
                # steep slope premature ventricular contraction or artifact
                # appeared, and for that reason it is limited to newM5 = 1.1*M5 if newM5 > 1.5* M5
                # The MM buffer is refreshed excluding the oldest component, and including M5 = newM5.
                Mnew = Mnew if Mnew <= 1.5*MM[MMidx-1] else 1.1*MM[MMidx-1]
                MM[MMidx] =  Mnew
                MMidx = scipy.mod(MMidx+1,5)
                # M is calculated as an average value of MM.
                Mtemp = scipy.mean(MM)
                M = Mtemp
                skip = True
            # M is decreased in an interval 200 to 1200 ms following
            # the last QRS detection at a low slope, reaching 60 % of its
            # refreshed value at 1200 ms.
            elif current_sample >= QRS[-1]+v200ms-offset and current_sample < QRS[-1]+v1200ms-offset:
                M = Mtemp*slope[current_sample - QRS[-1] - int(v200ms) + offset]
            # After 1200 ms M remains unchanged.
            # R = 0 V in the interval from the last detected QRS to 2/3 of the expected Rm.
            if current_sample >= QRS[-1]-offset and current_sample < QRS[-1]+(2/3.)*Rm-offset:
                R = 0
            # In the interval QRS + Rm * 2/3 to QRS + Rm, R decreases
            # 1.4 times slower then the decrease of the previously discussed
            # steep slope threshold (M in the 200 to 1200 ms interval).
            elif current_sample >= QRS[-1]+(2/3.)*Rm-offset and current_sample < QRS[-1]+Rm-offset:
                R += Rdec
            # After QRS + Rm the decrease of R is stopped
        # MFR = M + F + R
        MFR = M + F +  R
        # QRS or beat complex is detected if Yi = MFR
        if not skip and Y[current_sample] >= MFR:
            QRS +=[current_sample+offset]
            Rpeak += [QRS[-1]+scipy.argmax(Y[QRS[-1]: QRS[-1]+v300ms])]
            if len(QRS) >= 2:
                # A buffer with the 5 last RR intervals is updated at any new QRS detection.     
                RR[RRidx] = QRS[-1]-QRS[-2]
                RRidx = scipy.mod(RRidx+1,5)
        skip = False
        # With every signal sample, F is updated adding the maximum
        # of Y in the latest 50 ms of the 350 ms interval and
        # subtracting maxY in the earliest 50 ms of the interval.
        if current_sample >= v350ms-offset:
            if current_sample < v350ms and offset != 0:
                Y_latest50 = Y_temp[current_sample+v300ms:current_sample+v350ms]
                Y_earliest50 = Y_temp[current_sample:current_sample+v50ms]
            else: 
                Y_latest50 = Y[current_sample-v50ms:current_sample]
                Y_earliest50 = Y[current_sample-v350ms:current_sample-v300ms]
            F += (max(Y_latest50) - max(Y_earliest50))/1000.
        # Rm is the mean value of the buffer RR.
        Rm = scipy.mean(RR)
        current_sample += 1
        # Additional triggering of eventually 
        #missed heart beat in the last detected RR interval
        # if len(QRS) >= 3:
            # t1 = RR[RRidx-2]
            # t2 = RR[RRidx-1]
            # if (t1 > Rm or Rm-t1 < 0.12*Rm) and (abs(t2-2*Rm) < 0.5*Rm):
                # # A test is performed on each of the primary leads where a
                # # sharp peak is searched (defined as a product > 4 uV of two
                # # signal differences having one central and two lateral points 8 ms apart).
                # if ... :
                    # # If the test is passed, a second one is carried
                    # # out for the amplitude of the summary lead at that
                    # # point, which should be bigger then 1/3 of the mean value
                    # # of the buffer MM, in order to define this point as a missed QRS complex.
                    # if ... :
        if(Show):    
            M_hist += [M]
            F_hist += [F]
            R_hist += [R]
            MFR_hist += [MFR]
    if(Show):
        try:
            pl.figure(777)
            pl.figure(777).clf()
            t = scipy.linspace(0, len(Y)/SamplingRate, len(Y))
            t += offset/SamplingRate
            pl.plot(t, Y, 'b', lw=2)
            pl.plot(t, M_hist, 'r', lw=1)
            pl.plot(t, F_hist, 'g', lw=1)
            pl.plot(t, R_hist, 'y', lw=1)
            pl.plot(t, MFR_hist, 'k', lw=2)
            pl.plot(t[scipy.array(QRS[nbeats:])-offset], Y[scipy.array(QRS[nbeats:])-offset], 'ro', markersize=10)
            pl.legend(('Y', 'M', 'F', 'R', 'MFR', 'QRS'), 'best', shadow=True)
            pl.xlabel('Time (sec)')
            pl.grid('on')
            pl.show()
        except Exception:
            pass
    # Output
    # offset = QRS[-1]
    rpeaks = []
    for i in Rpeak:
        a, b = i-100, i+100
        if a < 0: a = 0
        if b > len(Signal): b = -1
        rpeaks.append(scipy.argmax(Signal[a:b])+a)
    kwrvals = {}
    kwrvals['R'] = rpeaks
    # kwrvals['Params'] = {
                        # 'R': Rpeak,
                        # 'QRS': QRS,
                        # 'M': MM, 
                        # 'MMidx': MMidx,
                        # 'Rth': R,
                        # 'RR': RR,
                        # 'RRidx': RRidx,
                        # 'F': F,
                        # 'offset': offset,
                        # 'Y_350': Y[offset-v350ms:],
                        # 'nbeats': len(QRS),
                        # 'X': X,
                        # 'Y': Y
                        # }
    return kwrvals

def gamboa(Signal=None, SamplingRate=1000., tol=0.002):
    """

    Gamboa's algorithm to detect ECG beat indexes.

    Kwargs:
        Signal (array): input filtered ECG signal.
        
        SamplingRate (float): Sampling frequency (Hz).
        
        tol (float): tolerance parameter.        

    Returns:
        rpeaks (array): R-peak indexes

    Configurable fields:{"name": "models.gamboa", "config": {"SamplingRate": "1000."}, "inputs": ["Signal"], "outputs": ["rpeaks"]}

    See Also:

    Notes:

    Example:

    References:
        .. [1] ...
        
    """
    # Check
    if Signal is None:
        raise TypeError, "An input signal is needed."    
    v_300ms = 0.3*SamplingRate
    
    nbins = 100
    (a,b,c) = pl.hist(Signal[::10],nbins)
    pl.close()
    
    TH = 0.01
    f = a.astype('d')
    F = scipy.cumsum(f/f.sum())
    v0 = b[pl.where(F>TH)[0][0]]
    v1 = b[pl.where(F<(1-TH))[0][-1]]
    if abs(v0) > v1: norm_signal = Signal/float(v0)
    else: norm_signal = Signal/float(v1)
    d2 = scipy.diff(norm_signal,2)
    # tol = 0.015# 
    # tol = 0.002 # CVP data 
    b = pl.find((pl.diff(pl.sign(pl.diff(-d2))))==-2)+2
    b = scipy.intersect1d(b, pl.find(-d2>tol))
    if (len(b)<3):
        raise ValueError, "Error Occured."
    else:
        b = b.astype('d')
        rpeaks = []
        previous = b[0]
        for i in b[1:]:
            if i-previous > v_300ms:
                previous = i
                rpeaks.append(scipy.argmax(Signal[i:i+100])+i)
        rpeaks = scipy.array(rpeaks)
    kwrvals = {}
    kwrvals['R'] = rpeaks
    return kwrvals

def ESSF(Signal=None, SamplingRate=1000., ESSF_params = {'s_win': 0.3, 's_amp': 70.}):
    """

    ECG Slope Sum Function: algorithm to detect ECG beat indexes.

    Kwargs:
        Signal (array): input filtered ECG signal.
        
        SamplingRate (float): Sampling frequency (Hz).
        
        ESSF_params (float): data dependent parameters.        

    Returns:
        rpeaks (array): R-peak indexes

    Configurable fields:{"name": "models.ESSF", "config": {"SamplingRate": "1000."}, "inputs": ["Signal"], "outputs": ["rpeaks"]}

    See Also:

    Notes:

    Example:

    References:
        .. [1] ...
        
    """
    # Check
    if Signal is None:
        raise TypeError, "An input signal is needed."      
    # ESSF_params depends on data !
    v_300ms = 0.3*SamplingRate
    dy = scipy.diff(Signal)
    dy[dy<0] = 0
    win = ESSF_params['s_win']*SamplingRate # 300.
    ssf = flt.smooth(dy, Window={'Length': win, 'Type':'boxcar','Parameters':None})['Signal']*ESSF_params['s_amp']
    ssf = flt.smooth(ssf, Window={'Length': 180, 'Type':'boxcar','Parameters':None})['Signal']
    # ssf[ssf<0.6*max(ssf)] = 0.6*max(ssf)
    ups = pl.find(Signal[1:]>=ssf)
    downs = pl.find(Signal[1:]<=ssf) + 1
    onset = scipy.intersect1d(ups, downs)
    rpeaks = []
    for i,on in enumerate(onset):
        if i == 0: rpeaks.append(on+scipy.argmax(Signal[on:on+100]))
        elif (on-rpeaks[-1]) > v_300ms: rpeaks.append(on+scipy.argmax(Signal[on:on+100]))
    
    kwrvals = {}
    kwrvals['R'] = scipy.array(rpeaks[1:-1])
    return kwrvals


def armSSF(Signal=None, SamplingRate=1000., threshold=20, winB=0.03, winA=0.01, Params=None):
    """
    Modified Slope Sum Function R peak detection algorithm for ARM board.
    
    Kwargs:
        Signal (array): input ECG signal.
        
        SamplingRate (float): Sampling frequency (Hz).
        
        threshold (int): threshold.
        
        winB (int): size of search window before candidate.
        
        winA (int): size of search window after candidate.
    
    Returns:
        rpeaks (array): R-peak indexes.
    
    Configurable fields:{}
    
    See Also:
    
    Notes:
    
    Example:
    
    References:
        .. [1] ...
        
    """
    
    # check inputs
    winB = int(SamplingRate * winB)
    winA = int(SamplingRate * winA)
    if Params is None:
        Params = {'R': [],
                  'x': [],
                  'sf': 0,
                  }
    
    Rset = set(Params['R'])
    
    # add continuity signal
    x = np.hstack((Params['x'], Signal))
    nb = len(x)
    sf = Params['sf']
    
    # diff
    dx = np.diff(x)
    dx[dx >= 0] = 0
    dx = dx ** 2
    
    # detection
    idx, = np.nonzero(dx > threshold)
    idx0 = np.hstack(([0], idx))
    didx = np.diff(idx0)
    
    # search
    nr = 0
    sidx = idx[didx > 1]
    for item in sidx:
        a = item - winB
        if a < 0:
            a = 0
        b = item + winA
        if b > nb:
            continue
        
        nr += 1
        r = np.argmax(x[a:b]) + a
        Rset.add(r + sf)
    
    # continuity
    if nr > 0:
        a = r + winA + 1
        Params['x'] = x[a:]
        Params['sf'] += a
    else:
        Params['x'] = x[0:]
    
    Params['R'] = list(Rset)
    Params['R'].sort()
    
    return Params


# @jit
def happee_variance(signal):
    """
    Compute HAPPEE variance [1].
    
    Kwargs:
        signal (array): input signal.
    
    Returns:
        variance (float): HAPPEE variance.
    
    Configurable fields:{}
    
    See Also:
    
    Notes:
    
    Example:
    
    References:
        .. [1] Kay, S.; Ding, Q.; Li, D., "On the Design and Implementation of A Highly Accurate
               Pulse Predictor for Exercise Equipment," IEEE Trans. on Biomedical Engineering,
               10.1109/TBME.2015.2407155
        
    """
    
    variance = 2. * (1. / signal.shape[0] * signal.sum()) ** 2
    
    return variance


def happee_grid(length, grid, M):
    """
    Generate the helper grid dictionary for the HAPPEE algorithm [1].
    
    Kwargs:
        length (int): size of the signal window.
        
        grid (array, list): search grid of pulse periods.
        
        M (int): Pulse length.
    
    Returns:
        gridMem (dict): helper grid dictionary.
        
    
    Configurable fields:{}
    
    See Also:
    
    Notes:
    
    Example:
    
    References:
        .. [1] Kay, S.; Ding, Q.; Li, D., "On the Design and Implementation of A Highly Accurate
               Pulse Predictor for Exercise Equipment," IEEE Trans. on Biomedical Engineering,
               10.1109/TBME.2015.2407155
        
    """
    
    length_f = float(length)
    
    gridMem = {'prototypes': {},
               'index': {},
               }
    
    for p in grid:
        indP = np.zeros(p, dtype='bool')
        indP[:M] = True
        indP = np.tile(indP, np.ceil(length_f / p))
        gridMem['prototypes'][p] = indP
        
        for n in xrange(0, p - M):
            indN = np.zeros(n, dtype='bool')
            ind2 = np.hstack((indN, indP))
            ind2 = ind2[:length]
            ind1 = np.logical_not(ind2)
            gridMem['index'][(p, n)] = (ind1, ind2)
    
    return gridMem


def happee_search(window, grid, M, gridMem):
    """
    Perform HAPPEE [1] search on a signal window.
    
    Kwargs:
        window (array): input signal window.
        
        grid (array, list): search grid of pulse periods.
        
        M (int): Pulse length.
        
        gridMem (dict): helper grid dictionary.
    
    Returns:
        T (float): Test statistic.
        
        n0 (int): Location of first pulse.
        
        P (int): Pulse period.
        
    
    Configurable fields:{}
    
    See Also:
    
    Notes:
    
    Example:
    
    References:
        .. [1] Kay, S.; Ding, Q.; Li, D., "On the Design and Implementation of A Highly Accurate
               Pulse Predictor for Exercise Equipment," IEEE Trans. on Biomedical Engineering,
               10.1109/TBME.2015.2407155
        
    """
    
    nb = window.shape[0]
    nb_f = float(nb)
    
    # only abs values are used
    window = np.abs(window)
    
    # estimate window variance
    S_global = happee_variance(window)
    
    # search grid
    T = 0
    P = 0
    n0 = 0
    
    for p in grid:
        try:
            indP = gridMem['prototypes'][p]
        except KeyError:
            indP = np.zeros(p, dtype='bool')
            indP[:M] = True
            indP = np.tile(indP, np.ceil(nb_f / p)) # repeats ind0
            gridMem['prototypes'][p] = indP
        
        for n in xrange(0, p - M):
            # select pulse elements
            try:
                ind1, ind2 = gridMem['index'][(p, n)]
            except KeyError:
                indN = np.zeros(n, dtype='bool')
                ind2 = np.hstack((indN, indP))
                ind2 = ind2[:nb]
                ind1 = np.logical_not(ind2)
                gridMem['index'][(p, n)] = (ind1, ind2)
            
            # compute statistic
            w1 = window[ind1]
            w2 = window[ind2]
            S_1 = happee_variance(w1)
            S_2 = happee_variance(w2)
            
            t = nb * np.log(S_global) - w1.shape[0] * np.log(S_1) - w2.shape[0] * np.log(S_2)
            
            # update maximum
            if t > T:
                T = t
                P = p
                n0 = n
    
    return T, n0, P


def happee(Signal=None, SamplingRate=1000., window=3., overlap=0.75, threshold=0., gridMem=None):
    """
    Implementation of the Highly Accurate Pulse Predictor for Exercise Equipment (HAPPEE) [1].
    
    Kwargs:
        Signal (array): input ECG signal.
        
        SamplingRate (float): Sampling frequency (Hz).
        
        window (float): Size (seconds) of the search window.
        
        overlap (float): Percentage of window overlap.
        
        threshold (float): Pulse detection threshold.
        
        gridMem (dict): helper grid dictionary.
    
    Returns:
        
    
    Configurable fields:{}
    
    See Also:
    
    Notes:
    
    Example:
    
    References:
        .. [1] Kay, S.; Ding, Q.; Li, D., "On the Design and Implementation of A Highly Accurate
               Pulse Predictor for Exercise Equipment," IEEE Trans. on Biomedical Engineering,
               10.1109/TBME.2015.2407155
        
    """
    
    # check inputs
    if Signal is None:
        raise TypeError("An input signal is needed.")
    
    Fs = float(SamplingRate)
    size = int(window * Fs)
    step = size - int(overlap * size)
    
    # HR search grid
    fb = Fs * 60.
    grid = fb / np.arange(40, 221, dtype='float')
    grid = grid.round()
    grid = np.unique(grid.astype('int'))
    M = int(Fs * 0.1)
    if gridMem is None:
        gridMem = {'prototypes': {},
                   'index': {},
                   }
    
    nb = 1 + (len(Signal) - size) / step
    HP = np.zeros(nb, dtype='float')
    t = np.zeros(nb, dtype='float')
    thr = np.zeros(nb, dtype='float')
    
    for i in xrange(nb):
        a = i * step
        t[i] = a
        wnd = Signal[a:a+size]
        
        T, n0, P = happee_search(wnd, grid, M, gridMem)
        
        thr[i] = T
        
        if T > threshold:
            HP[i] = P
        else:
            HP[i] = np.nan
    
    HR = fb / HP
    t /= Fs
    
    return t, HR, thr



if __name__=='__main__':
    pass
        
        