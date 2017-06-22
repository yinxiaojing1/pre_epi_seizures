import pylab as pl
import numpy as np
import scipy.signal as ss
import heapq
from math import pi, log

import sys
# sys.path.append('C:\\Users\\FrancisD23\\Documents\\Repositorios_tese\\BioSPPy')
# sys.path.append('/media/Data/repos/BioSSPy')			# for cardiocloud_server work
import tools
import peakd
import filt as flt

#BiometricsPyKit
sys.path.append('C:\\Users\\FrancisD23\\Documents\\Repositorios_tese\\BiometricsPyKit')
from preprocessing import preprocessing as prepro
from outlier import outlier


def hamilton(hand, Signal=None, SamplingRate=1000., Filter=True, init=(), Show=0, show2=0, show3=0, TH=None):
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

    if Filter:
        # 0.15 - Remove EMG, powerline and baseline shift
        emgsamples = 0.028*SamplingRate
        movemg = np.ones(emgsamples) / emgsamples
        rawbase = prepro.medFIR(Signal, SamplingRate)['Signal']
        rawend = ss.convolve(rawbase, movemg, mode='same')
        RawSignal = np.copy(rawend)
    else:
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
    Signal = moving_average(Signal, int(0.15*SamplingRate), cut=True)
    # 0.3 - Initialize Buffers
    if not init:
        init_ecg = 8
        if len(Signal)/(1.*SamplingRate) < init_ecg:
            init_ecg = int(len(Signal)/(1.*SamplingRate))
        qrspeakbuffer = np.zeros(init_ecg)
        noisepeakbuffer = np.zeros(init_ecg)
        print init_ecg
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
        if TH is None:
            TH = 0.45  # 0.45 for CVP, 0.475 for  ECGIDDB, 0.35 for PTB                    # 0.3125 - 0.475
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

        if Show:
            fig = pl.figure()
            mngr = pl.get_current_fig_manager()
            mngr.window.setGeometry(950, 50, 1000, 800)
            ax = fig.add_subplot(211)
            ax.plot(Signal, 'b', label='Signal')
            ax.grid('on')
            ax.axis('tight')
            ax.plot(all_peaks, Signal[all_peaks], 'ko', ms=10, label='peaks')
            if np.any(np.array(beats)):
                ax.plot(np.array(beats), Signal[np.array(beats)], 'g^', ms=10, label='rpeak')
            range_aid = range(len(Signal))
            ax.plot(range_aid, init['DT']*np.ones(len(range_aid)), 'r--', label='DT')
            ax.legend(('Processed Signal', 'all peaks', 'R-peaks', 'DT'), 'best', shadow=True)
            ax = fig.add_subplot(212)
            ax.plot(RawSignal, 'b', label='Signal')
            ax.grid('on')
            ax.axis('tight')
            ax.plot(all_peaks, RawSignal[all_peaks], 'ko', ms=10, label='peaks')
            if np.any(np.array(beats)):
                ax.plot(np.array(beats), RawSignal[np.array(beats)], 'g^', ms=10, label='rpeak')
            pl.show()
            if raw_input('_') == 'q':
                sys.exit()
            pl.close()

        # --> Update Detection Threshold
        ANP = np.median(init['noisepeakbuffer'])
        AQRSP = np.median(qrspeakbuffer)
        init['DT'] = ANP + TH*(AQRSP - ANP)

    if show3:
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
            if show2:
                print 'diffs, main', diff_f, max_f, '\nsec', diff_s, max_s
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

        if show2:
            fig = pl.figure()
            mngr = pl.get_current_fig_manager()
            mngr.window.setGeometry(950, 50, 1000, 800)
            ax = fig.add_subplot(111)
            ax.plot(rawwindow, 'b')
            for i in xrange(poslen):
                ax.plot(twopeaks[i][1], twopeaks[i][0], 'bo', markersize=10)
            ax.plot(rawbeats[b]-add, rawwindow[rawbeats[b]-add], 'yo', markersize=7)
            ax.grid('on')
            ax.axis('tight')
            pl.show()
            raw_input('---')
            pl.close()

    # kwrvals
    kwrvals = {}
    kwrvals['Signal'] = RawSignal
    kwrvals['init'] = init
    kwrvals['R'] = sorted(list(frozenset(rawbeats)))#/SamplingRate if SamplingRate else beats

    return kwrvals


def prefilt(raw, srate, show):
    med1 = ss.medfilt(raw, 201)
    med2 = ss.medfilt(med1, 601)
    rawbase = raw - med2
    powsamples = int(srate/50.)
    movpow = np.ones(powsamples) / powsamples
    rawpow = ss.convolve(rawbase, movpow, mode='same')
    emgsamples = int(0.028*srate)
    movemg = np.ones(emgsamples) / emgsamples
    rawemg = ss.convolve(rawpow, movemg, mode='same')
    # highfreq = 2.*1/srate
    # n, d = ss.butter(4, highfreq, btype='high')         # try before moving averages OU descartar inicio
    # rawbase = ss.filtfilt(n, d, rawemg)
    # rawend = np.copy(rawbase)
    rawbase2 = prepro.medFIR(raw, srate)['Signal']
    rawend = ss.convolve(rawbase2, movemg, mode='same')

    if show:
        fig = pl.figure()
        mngr = pl.get_current_fig_manager()
        mngr.window.setGeometry(950, 50, 1000, 800)
        ax = fig.add_subplot(411)
        ax.plot(raw)
        ax.set_title('raw signal')
        ax = fig.add_subplot(413)
        ax.plot(rawend)
        ax.set_title('all out v2')
        ax = fig.add_subplot(414)
        ax.plot(rawemg)
        ax.set_title('emg out')
        ax = fig.add_subplot(412)
        ax.plot(rawbase)
        ax.set_title('baseline out')
        pl.show()
        raw_input('cleaning')

    return rawend


def moving_average(a, n, cut=False):
    weigths = np.repeat(1.0, n)/n
    if not cut:
        smas = np.convolve(a, weigths, 'same')
    else:
        smas = np.convolve(a, weigths, 'valid')
    return smas


def definepeak(signal, srate, filter=False, show=0):

    if filter:
        print 'bla'
        signal = prefilt(signal, srate, 0)
    bin_nr = 50
    meankill = 1/3.5
    upmargin = 0.92
    w_peaks = peakd.sgndiff(Signal=signal)['Peak']
    w_negpeaks = peakd.sgndiff(Signal=signal, a=1)['Peak']
    zerdiffs = np.where(np.diff(signal) == 0)[0]
    w_peaks = np.concatenate((w_peaks, zerdiffs))
    w_negpeaks = np.concatenate((w_negpeaks, zerdiffs))

    poscounts, posedges = np.histogram(signal[w_peaks], bin_nr)
    negcounts, negedges = np.histogram(signal[w_negpeaks], bin_nr)

    poscond = (poscounts > 0) + (poscounts < max(poscounts))
    negcond = (negcounts > 0) + (negcounts < max(negcounts))
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

    if show:
        print 'meanup', meanpos
        print 'meandown', meanneg
        print derppos, '\n', derp_edgepos
        print derpneg, '\n', derp_edgeneg

    negmax = np.min(derp_edgeneg)
    posmax = np.max(derp_edgepos)

    if posmax >= upmargin*abs(negmax):
        print 'UP', posmax, upmargin*negmax, meanpos, meanneg
        up = True
    else:
        print 'DOWN', posmax, upmargin*negmax, meanpos, meanneg
        up = False

    if show:
        fig = pl.figure()
        mngr = pl.get_current_fig_manager()
        # mngr.window.showMaximized()
        mngr.window.setGeometry(950, 50, 1000, 800)
        ax = fig.add_subplot(311)
        ax.plot(signal)
        ax.plot(w_peaks, signal[w_peaks], 'go')
        ax.plot(w_negpeaks, signal[w_negpeaks], 'ro')
        ax = fig.add_subplot(312)
        ax.bar(posedges[:-1], poscounts, posedges[1]-posedges[0])
        ax = fig.add_subplot(313)
        ax.bar(negedges[:-1], negcounts, negedges[1]-negedges[0])
        raw_input('histograms')
        pl.close()

    return up


def monhe(raw, srate, show=0, show2=0, show3=1, filter=True):
    """
        GREAT but:
            discard crossings close to one another by less than 100 ms

    """
    # 0 - Remove EMG, powerline and baseline shift
    if filter:
        rawend = prefilt(raw, srate, show)
    else:
        rawend = np.copy(raw)

    # 0.5 - Choice sign of peaks (batch)
    up = definepeak(rawend, srate)

    # 1 - filter block (chebyshev 4th order 6-18 Hz)
    nyqfreq = srate/2.
    filtband = [6/nyqfreq, 18/nyqfreq]

    num, den = ss.cheby2(4, 40, filtband, btype='bandpass')
    # filtafter2 = ss.filtfilt(num, den, raw)
    filtafter = ss.filtfilt(num, den, rawend)

    if show:
            fig = pl.figure()
            mngr = pl.get_current_fig_manager()
            mngr.window.setGeometry(950, 50, 1000, 800)
            ax = fig.add_subplot(411)
            ax.plot(raw)
            ax.set_title('raw signal')
            ax = fig.add_subplot(412)
            ax.plot(rawend)
            ax.set_title('filtered from raw')
            ax = fig.add_subplot(413)
            # ax.plot(filtafter2)
            ax.plot(filtafter, 'r')
            ax.set_title('filtered for algorithm after preprocessing')
            ax = fig.add_subplot(414)
            ax.plot(filtafter)
            ax.set_title('filtered for algorithm')
            pl.show()
            raw_input('cleaning')

    # 2 - differentiate the signal and normalize according to max derivative in the signal
    diffsig = np.diff(filtafter)
    diffmax = np.max(np.abs(diffsig))
    dsignal = diffsig / diffmax

    # 3 - Get Shannon energy envelope
    diffsquare = dsignal**2
    logdiff = np.log(diffsquare)
    shannon = -1*diffsquare*logdiff

    # 4 - Two sided zero-phase filtering
    windowlen = 0.15*srate    # safe length of a QRS pulse
    rectangular = ss.boxcar(windowlen)
    smoothfirst = ss.convolve(shannon, rectangular, mode='same')
    revfirst = smoothfirst[::-1]
    smoothsec = ss.convolve(revfirst, rectangular, mode='same')
    smoothfinal = smoothsec[::-1]

    # 5 - Hilbert transform applied to the smoothed shannon energy envelope
    hilbbuff = ss.hilbert(smoothfinal)
    hilbsign = np.imag(hilbbuff)

    # 6 - Get moving average of the hilbert transform so as to subtract after
    n = int(2.5*srate)
    movwind = np.ones(n) / n
    movav = ss.convolve(hilbsign, movwind, mode='same')
    analyser = hilbsign - movav

    # 7 - Get zero crossings (from negative to positive) of the 'analyser' signal
    zero_crossings = np.where(np.diff(np.sign(analyser)))[0]
    zero_crossings = zero_crossings[zero_crossings > 0.05*srate]    # discard boundary effect that might appear at start
    crossers = analyser[zero_crossings]
    beats = zero_crossings[crossers < 0]
    crossdiffs = np.diff(beats)
    dangerous = crossdiffs < 0.15*srate          # to avoid stupid repetitions
    dangerous = np.nonzero(dangerous)[0]
    if len(dangerous):
        print 'DANGER', beats[dangerous]
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
            print '\tEVIL SLOPE', b, crossderivs[b], anti_fp*cross_med, reset_count
            evilbeats.append(b)
            beats.remove(b)
        if reset_count >= anti_badset:
            print '\tRESET'
            reset_count = 0
            cross_storer = np.zeros(store_size)
    beats = np.array(beats, dtype=int)
    evilbeats = np.array(evilbeats)

    # 8 ----------------------------------------- Find the R-peak exactly -----------------------------------------
    search = int(0.15*srate)
    adjacency = int(0.03*srate)
    diff_nr = int(0.01*srate)
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
        w_negpeaks = peakd.sgndiff(Signal=window, a=1)['Peak']
        zerdiffs = np.where(np.diff(rawwindow) == 0)[0]
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
            if show2:
                print 'diffs, main', diff_f, max_f, '\nsec', diff_s, max_s
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

        if show2:
            fig = pl.figure()
            mngr = pl.get_current_fig_manager()
            mngr.window.setGeometry(950, 50, 1000, 800)
            ax = fig.add_subplot(111)
            ax.plot(rawwindow, 'b')
            for i in xrange(poslen):
                ax.plot(twopeaks[i][1], twopeaks[i][0], 'bo', markersize=10)
            ax.plot(rawbeats[b]-add, rawwindow[rawbeats[b]-add], 'yo', markersize=7)
            ax.grid('on')
            ax.axis('tight')
            pl.show()
            raw_input('---')
            pl.close()

    # 8 ----------------------------------------- END OF POINT 8 -----------------------------------------

    if show3:
        fig = pl.figure()
        mngr = pl.get_current_fig_manager()
        mngr.window.setGeometry(950, 50, 1000, 800)
        ax = fig.add_subplot(412)
        ax.plot(rawend)
        if len(rawbeats):
            ax.plot(rawbeats, rawend[rawbeats], 'go')
        ax.set_title('end signal')
        ax = fig.add_subplot(411)
        ax.plot(raw)
        if beats.any():
            ax.plot(beats, raw[beats], 'go')
        ax.set_title('filtered from raw')
        ax = fig.add_subplot(413)
        ax.plot(smoothfinal)
        ax.set_title('smooth shannon')
        ax = fig.add_subplot(414)
        ax.plot(analyser)
        if beats.any():
            ax.plot(beats, analyser[beats], 'go')
        if evilbeats.any():
            ax.plot(evilbeats, analyser[evilbeats], 'ro')
        ax.plot(hilbsign, 'r')
        ax.set_title('analysed signal')
        pl.show()
        raw_input('shannon')

    # pl.close('all')

    # kwrvals
    kwrvals = {'Signal': rawend, 'R': sorted(list(frozenset(rawbeats)))}

    return kwrvals


def monhe2(raw, srate, show=0, show2=0, show3=0, filtered=None):
    """
        GREAT but:
            discard crossings close to one another by less than 100 ms

    """
    # 0 - Remove EMG, powerline and baseline shift
    if filtered is None:
        filtered = prefilt(raw, srate, show)

    # 0.5 - Choose sign of peaks (batch)
    up = definepeak(filtered, srate)

    # 1 - filter block (chebyshev 4th order 6-18 Hz)
    nyqfreq = srate/2.
    filtband = [6/nyqfreq, 18/nyqfreq]

    num, den = ss.cheby2(4, 40, filtband, btype='bandpass')
    filtafter = ss.filtfilt(num, den, raw)
    # filtafter = ss.filtfilt(num, den, rawend)

    if show:
            fig = pl.figure()
            mngr = pl.get_current_fig_manager()
            mngr.window.setGeometry(950, 50, 1000, 800)
            ax = fig.add_subplot(411)
            ax.plot(raw)
            ax.set_title('raw signal')
            ax = fig.add_subplot(412)
            ax.plot(filtered)
            ax.set_title('filtered from raw')
            ax = fig.add_subplot(413)
            # ax.plot(filtafter2)
            ax.plot(filtafter, 'r')
            ax.set_title('filtered for algorithm after preprocessing')
            ax = fig.add_subplot(414)
            ax.plot(filtafter)
            ax.set_title('filtered for algorithm')
            pl.show()
            raw_input('cleaning')

    # 2 - differentiate the signal and normalize according to max derivative in the signal
    diffsig = np.diff(filtafter)
    diffmax = np.max(np.abs(diffsig))
    dsignal = diffsig / diffmax

    # 3 - Get Shannon energy envelope
    diffsquare = dsignal**2
    logdiff = np.log(diffsquare)
    shannon = -1*diffsquare*logdiff

    # 4 - Two sided zero-phase filtering
    windowlen = int(0.15*srate)    # safe length of a QRS pulse
    rectangular = ss.boxcar(windowlen)
    smoothfirst = ss.convolve(shannon, rectangular, mode='same')
    revfirst = smoothfirst[::-1]
    smoothsec = ss.convolve(revfirst, rectangular, mode='same')
    smoothfinal = smoothsec[::-1]

    # 5 - Hilbert transform applied to the smoothed shannon energy envelope
    hilbbuff = ss.hilbert(smoothfinal)
    hilbsign = np.imag(hilbbuff)

    # 6 - Get moving average of the hilbert transform so as to subtract after
    n = int(2.5*srate)
    movav = moving_average(hilbsign, n)
    analyser = hilbsign - movav

    # 7 - Get zero crossings (from negative to positive) of the 'analyser' signal
    zero_crossings = np.where(np.diff(np.sign(analyser)))[0]
    zero_crossings = zero_crossings[zero_crossings > 0.05*srate]    # discard boundary effect that might appear at start
    crossers = analyser[zero_crossings]
    beats = zero_crossings[crossers < 0]
    crossdiffs = np.diff(beats)
    dangerous = crossdiffs < 0.15*srate          # to avoid stupid repetitions
    dangerous = np.nonzero(dangerous)[0]
    if len(dangerous):
        print 'DANGER', beats[dangerous]
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
            print '\tEVIL SLOPE', b, crossderivs[b], anti_fp*cross_med, reset_count
            evilbeats.append(b)
            beats.remove(b)
        if reset_count >= anti_badset:
            print '\tRESET'
            reset_count = 0
            cross_storer = np.zeros(store_size)
    beats = np.array(beats, dtype=int)
    evilbeats = np.array(evilbeats)

    # 8 ----------------------------------------- Find the R-peak exactly -----------------------------------------
    search = int(0.15*srate)
    adjacency = int(0.03*srate)
    diff_nr = int(0.01*srate)
    if diff_nr <= 1:
        diff_nr = 2
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
        zerdiffs = np.where(np.diff(rawwindow) == 0)[0]
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
            if show2:
                print 'diffs, main', diff_f, max_f, '\nsec', diff_s, max_s
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

        if show2:
            fig = pl.figure()
            mngr = pl.get_current_fig_manager()
            mngr.window.setGeometry(950, 50, 1000, 800)
            ax = fig.add_subplot(111)
            ax.plot(rawwindow, 'b')
            for i in xrange(poslen):
                ax.plot(twopeaks[i][1], twopeaks[i][0], 'bo', markersize=10)
            ax.plot(rawbeats[b]-add, rawwindow[rawbeats[b]-add], 'yo', markersize=7)
            ax.grid('on')
            ax.axis('tight')
            pl.show()
            raw_input('---')
            pl.close()

    # 8 ----------------------------------------- END OF POINT 8 -----------------------------------------

    if show3:
        fig = pl.figure()
        mngr = pl.get_current_fig_manager()
        mngr.window.setGeometry(950, 50, 1000, 800)
        ax = fig.add_subplot(412)
        ax.plot(filtered)
        if len(rawbeats):
            ax.plot(rawbeats, filtered[rawbeats], 'go')
        ax.set_title('end signal')
        ax = fig.add_subplot(411)
        ax.plot(raw)
        if beats.any():
            ax.plot(beats, raw[beats], 'go')
        ax.set_title('filtered from raw')
        ax = fig.add_subplot(413)
        ax.plot(smoothfinal)
        ax.set_title('smooth shannon')
        ax = fig.add_subplot(414)
        ax.plot(analyser)
        if beats.any():
            ax.plot(beats, analyser[beats], 'go')
        if evilbeats.any():
            ax.plot(evilbeats, analyser[evilbeats], 'ro')
        ax.plot(hilbsign, 'r')
        ax.set_title('analysed signal')
        pl.show()
        raw_input('shannon')

    hrate = np.diff(rawbeats)
    hrate = 60*srate/hrate


    pl.close('all')

    # kwrvals
    kwrvals = {'Signal': filtered, 'R': sorted(list(frozenset(rawbeats)))}

    return kwrvals, hrate