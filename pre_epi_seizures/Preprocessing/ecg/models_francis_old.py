import scipy
import pylab
import numpy
import scipy.signal as ss
import heapq

import sys
# sys.path.append('C:\\Users\\FrancisD23\\Documents\\Repositorios_tese\\BioSPPy')
# sys.path.append('/media/Data/repos/BioSSPy')			# for cardiocloud_server work
import tools
import peakd
import filt as flt

def hamilton(hand, Signal=None, SamplingRate=1000., Filter=True, init=(), Show=False):
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
    RawSignal = numpy.array(Signal[:])
    if Filter:
        UpperCutoff = 25.
        LowerCutoff = 3.
        Order = 4
        Signal = flt.zpdfr(Signal=Signal, SamplingRate=SamplingRate, UpperCutoff=UpperCutoff,LowerCutoff=LowerCutoff,Order=Order)['Signal']
        Signal = abs(scipy.diff(Signal,1)*SamplingRate)
        Signal = flt.smooth(Signal=Signal, Window={'Length':0.08*SamplingRate, 'Type':'hamming', 'Parameters':None})['Signal']
        # Filter.update({'Signal': Signal})
        # if not Filter.has_key('SamplingRate'): Filter.update({'SamplingRate': SamplingRate})
        # Signal = flt(**Filter)['Signal']
        # Init
    SamplingRate = float(SamplingRate)
    if not init:
        init_ecg = 8			#seconds for initialization
        if len(Signal)/SamplingRate < init_ecg:
            init_ecg = int(len(Signal)/SamplingRate)
        qrspeakbuffer = scipy.zeros(init_ecg)
        noisepeakbuffer = scipy.zeros(init_ecg)
        peak_idx_test = scipy.zeros(init_ecg)						#---zdec---
        noise_idx = scipy.zeros(init_ecg)
        rrinterval = SamplingRate*scipy.ones(init_ecg)
        # In order to make an initial estimate, we detect the maximum peaks in eight consecutive 1-second intervals.
        # These eight peaks are used as are initial eight values in the QRS peak buffer, 
        # we set the initial eight noise peaks to 0, and we set the initial threshold accordingly. 
        # We initially set the eight most recent R-to-R intervals to 1 second.
        # Init QRS buffer
        a,b = 0,SamplingRate
        all_peaks = numpy.array(peakd.sgndiff(Signal)['Peak'])#-1			#----REMOVE THE -1 WHEN LOCAL USE
        # print all_peaks, '\n'
        for i in range(0,init_ecg):
            peaks = peakd.sgndiff(Signal=Signal[a:b])['Peak']
            try:
                qrspeakbuffer[i] = max(Signal[a:b][peaks])   		#peak amplitude
                peak_idx_test[i] = peaks[scipy.argmax(Signal[a:b][peaks])] + a		#---zdec---
            except Exception as e:
                print e
            a += SamplingRate
            b += SamplingRate
        # Set Thresholds                                
        # Detection_Threshold = Average_Noise_Peak + TH*(Average_QRS_Peak-Average_Noise_Peak)
        ANP = scipy.median(noisepeakbuffer)
        AQRSP = scipy.median(qrspeakbuffer)
        TH = 0.475                                                 #0.3125 - 0.475
        DT = ANP + TH*(AQRSP - ANP)
        DT_vec = []
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
    lim = int(scipy.ceil(0.2*SamplingRate))
    diff_nr = int(scipy.ceil(0.045*SamplingRate))
    bpsi,bpe = int(init['offset']),0
    for f in all_peaks:
        # print '---', f, '---'
        DT_vec += [init['DT']]
        #1 - Checking if f-peak is larger than any peak following or preceding it by less than 200 ms
        peak_cond = numpy.array( (all_peaks > f - lim) * (all_peaks < f + lim) * (all_peaks != f) )
        peaks_within = all_peaks[peak_cond]
        if (peaks_within.any() and (max(Signal[peaks_within]) > Signal[f]) ):
            # print 'TINY '
            continue
        #4 - If the peak is larger than the detection threshold call it a QRS complex, otherwise call it noise
        if(Signal[f] > init['DT']):
            #---------------------FRANCIS---------------------
            #2 - look for both positive and negative slopes in raw signal
            if f < diff_nr:
                diff_now = scipy.diff(RawSignal[0:f+diff_nr])
            elif f + diff_nr >= len(RawSignal):
                diff_now = scipy.diff(RawSignal[f-diff_nr:len(Signal)])
            else:
                diff_now = scipy.diff(RawSignal[f-diff_nr:f+diff_nr])
            diff_signer = diff_now[ diff_now > 0]
            # print 'diff signs:', diff_signer, '\n', diff_now
            if len(diff_signer) == 0 or len(diff_signer) == len(diff_now):
                # print 'BASELINE SHIFT'
                continue
            #RR INTERVALS
            if(init['npeaks'] > 0):
                #---------------------FRANCIS---------------------
                #3 - in here we check point 3 of the Hamilton paper
                # that is, we check whether our current peak is a valid R-peak.
                # print 'info here, npeaks - ', init['npeaks'], ', beats - ', beats				#---zdec---
                prev_rpeak = beats[init['npeaks']-1]
                # print 'prev_rpeak - ', prev_rpeak												#---zdec---
                elapsed = f - prev_rpeak
                # print 'elapsed', elapsed
                # if the previous peak was within 360 ms interval
                if( elapsed < scipy.ceil(0.36*SamplingRate) ):
                    # check current and previous slopes
                    if prev_rpeak < diff_nr:
                        diff_prev = scipy.diff(RawSignal[0:prev_rpeak+diff_nr])
                    elif prev_rpeak+diff_nr >= len(RawSignal):
                        diff_prev = scipy.diff(RawSignal[prev_rpeak-diff_nr:len(Signal)])
                    else:
                        diff_prev = scipy.diff(RawSignal[prev_rpeak-diff_nr:prev_rpeak+diff_nr])
                    # print 'diff_now', diff_now
                    # print 'diff_prev', diff_prev
                    # print prev_rpeak
                    slope_now = max(diff_now)
                    slope_prev = max(diff_prev)
                    # print 'slopes: now - ', slope_now, 'prev - ', slope_prev					#---zdec---
                    # raw_input('_')
                    # Show = True
                    if(slope_now < 0.5*slope_prev):
                        # if current slope is smaller than half the previous one, then it is a T-wave
                        # print 'T-WAVE'
                        continue
                if not(hand) or Signal[f] < 3.*scipy.median(qrspeakbuffer):								# avoid retarded noise peaks
                    # print 'GOT IT GOOD'
                    beats += [int(f)+bpsi]
                else:
                    continue
                # print '+++++++++', beats[-1], '+++++++++'										#---zdec---
                #-----------------END_OF_FRANCIS------------------
                if(bpe == 0):
                    init['rrinterval'][init['indexrr']] = beats[init['npeaks']]-beats[init['npeaks']-1]
                    init['indexrr'] = init['indexrr']+1
                    if(init['indexrr'] == init_ecg): init['indexrr']=0
                else:
                    if(beats[init['npeaks']] > beats[bpe-1]+100):
                        init['rrinterval'][init['indexrr']] = beats[init['npeaks']]-beats[init['npeaks']-1]
                        init['indexrr'] = init['indexrr']+1
                        if(init['indexrr']==init_ecg): init['indexrr'] = 0
            #---------------------FRANCIS---------------------
            elif not(hand) or Signal[f] < 3.*scipy.median(qrspeakbuffer):
                # print 'GOT IT GOOD'
                beats += [int(f)+bpsi]
            else:
                continue
                # print '+++++++++', beats[-1], '+++++++++'									#---zdec---
            #-----------------END_OF_FRANCIS------------------
            init['npeaks'] += 1
            qrspeakbuffer[init['indexqrs']] = Signal[f]
            peak_idx_test[init['indexqrs']] = f
            init['indexqrs'] += 1
            if(init['indexqrs']==init_ecg): init['indexqrs'] = 0
        if(Signal[f] <= init['DT']):                    #4 - not valid
            # 5 - If no QRS has been detected within 1.5 R-to-R intervals, 
            # there was a peak that was larger than half the detection threshold, 
            # and the peak followed the preceding detection by at least 360 ms, 
            # classify that peak as a QRS complex
            tf = f+bpsi
            # RR interval median
            RRM = scipy.median(init['rrinterval'])            # initial values are good?
            # print 'RRM', RRM, init['rrinterval']								#---zdec---
            if(len(beats) >= 2):
                elapsed = tf-beats[init['npeaks']-1]
                # print '2nd elapsed', elapsed, 1.5*RRM				#---zdec---
                if(elapsed >= 1.5*RRM and elapsed > scipy.ceil(0.36*SamplingRate)):
                    if(Signal[f] > 0.5*init['DT']):
                        # print 'GOT IT RR'
                        beats+=[int(f)+int(init['offset'])]
                        #RR INTERVALS
                        if(init['npeaks']>0):
                            init['rrinterval'][init['indexrr']] = beats[init['npeaks']]-beats[init['npeaks']-1]
                            init['indexrr'] += 1
                            if(init['indexrr'] == init_ecg): init['indexrr'] = 0
                        init['npeaks'] += 1
                        qrspeakbuffer[init['indexqrs']] = Signal[f]
                        peak_idx_test[init['indexqrs']] = f
                        init['indexqrs'] += 1
                        if(init['indexqrs'] == init_ecg): init['indexqrs'] = 0
                else:
                    init['noisepeakbuffer'][init['indexnoise']] = Signal[f]
                    noise_idx[init['indexnoise']] =f
                    init['indexnoise'] += 1
                    # print 'NOISE'
                    if(init['indexnoise'] == init_ecg): init['indexnoise'] = 0
            else:
                init['noisepeakbuffer'][init['indexnoise']] = Signal[f]
                noise_idx[init['indexnoise']] =f
                init['indexnoise'] += 1
                # print 'NOISE'
                if(init['indexnoise'] == init_ecg): init['indexnoise'] = 0
        # derp = raw_input('PRESS ENTER\n')			#---zdec---
        #Update Detection Threshold
        ANP = scipy.median(init['noisepeakbuffer'])
        AQRSP = scipy.median(qrspeakbuffer)
        init['DT'] = ANP + 0.475*(AQRSP - ANP)
		
        if Show:
            fig = pylab.figure()
            mngr = pylab.get_current_fig_manager()
            mngr.window.setGeometry(950,50,1000,800)
            ax = fig.add_subplot(211)
            ax.plot(Signal, 'b', label='Signal')
            ax.grid('on')        
            ax.axis('tight')
            ax.plot(all_peaks, Signal[all_peaks], 'ko', ms=10, label='peaks')		
            if(scipy.any(scipy.array(beats))): ax.plot(scipy.array(beats)-bpsi, Signal[scipy.array(beats)-bpsi], 'g^', ms=10, label='rpeak')
            range_aid = range(len(Signal))
            ax.plot(range_aid, DT_vec[-1]*scipy.ones(len(range_aid)),'r--', label='DT')
            ax.legend(('Processed Signal', 'all peaks', 'R-peaks','DT'), 'best', shadow=True)
            ax = fig.add_subplot(212)
            ax.plot(RawSignal, 'b', label='Signal')
            ax.grid('on')        
            ax.axis('tight')
            ax.plot(all_peaks, RawSignal[all_peaks], 'ko', ms=10, label='peaks')
            if(scipy.any(scipy.array(beats))): ax.plot(scipy.array(beats)-bpsi, RawSignal[scipy.array(beats)-bpsi], 'g^', ms=10, label='rpeak')
            # ax.axes.yaxis.set_visible(False)
            # ax.axes.xaxis.set_visible(False)
            pylab.show()
            if raw_input('_')=='q': sys.exit()
            pylab.close()
            # Show = False
	
    beats = scipy.array(beats)
    # kwrvals
    kwrvals = {}
    kwrvals['Signal'] = Signal
    kwrvals['init'] = init

    lim = lim
    r_beats = []
    thres_ch = 0.85
    adjacency = 0.05*SamplingRate
    for i in beats:
        # print '----------', i, adjacency, '----------'
        error = [False, False]
        if i-lim < 0:
            window = RawSignal[0:i+lim]
            add = 0
        elif i+lim >= len(RawSignal):
            window = RawSignal[i-lim:len(RawSignal)]
            add = i-lim
        else:
            window = RawSignal[i-lim:i+lim]
            add = i-lim
        meanval = numpy.mean(window)
        w_peaks = peakd.sgndiff(Signal=window)['Peak']
        w_negpeaks = peakd.sgndiff(Signal=window, a=1)['Peak']
        zerdiffs = numpy.where(scipy.diff(window) == 0)[0]
        w_peaks = numpy.concatenate( (w_peaks, zerdiffs) )
        w_negpeaks = numpy.concatenate( (w_negpeaks, zerdiffs) )

        pospeaks = sorted(zip(window[w_peaks], w_peaks), reverse=True)
        negpeaks = sorted(zip(window[w_negpeaks], w_negpeaks))
        # print '\n peaksssss', pospeaks, negpeaks

        # print '\n diffs', zerdiffs
        try:
            twopeaks = [pospeaks[0]]
        except IndexError:
            pass
            # print 'diff pos----->', twopeaks
        try:
            twonegpeaks = [negpeaks[0]]
        except IndexError:
            pass
            # print 'diff neg----->', twonegpeaks
		
        # ----------- getting positive peaks -----------
        for i in xrange(len(pospeaks)-1):
            if abs(pospeaks[0][1] - pospeaks[i+1][1]) > adjacency:
                twopeaks.append(pospeaks[i+1])
                break
        try:
            # posdiv = (twopeaks[0][0]-meanval)/(1.*twopeaks[1][0]-meanval)
            posdiv = abs( twopeaks[0][0]-twopeaks[1][0] )
            # print 'peaks', twopeaks[0][0], twopeaks[1][0], posdiv
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
            # print 'negpeaks', twonegpeaks[0][0], twonegpeaks[1][0], thres_ch*negdiv
        except IndexError:
            error[1] = True
			
        # ----------- choosing type of R-peak -----------
        if not sum(error):
            if posdiv > thres_ch*negdiv:
                # print '\t pos noerr'
                r_beats.append(twopeaks[0][1] + add)
            else:
                # print '\t neg noerr'
                r_beats.append(twonegpeaks[0][1] + add)
        elif sum(error) == 2:
            if abs(twopeaks[0][1]) > abs(twonegpeaks[0][1]):
                # print '\t pos allerr'
                r_beats.append(twopeaks[0][1] + add)
            else:
                # print '\t neg allerr'
                r_beats.append(twonegpeaks[0][1] + add)
        elif error[0]: 
            # print '\t pos poserr'
            r_beats.append(twopeaks[0][1] + add)
        else: 
            # print '\t neg negerr'
            r_beats.append(twonegpeaks[0][1] + add)
			
        # print 'rbeats', r_beats
        # fig = pylab.figure()
        # mngr = pylab.get_current_fig_manager()
        # mngr.window.setGeometry(950,50,1000,800)
        # ax = fig.add_subplot(111)
        # ax.plot(window, 'b')
        # ax.grid('on')        
        # ax.axis('tight')
        # zerdiffs = numpy.array(zerdiffs)
        # for i in xrange(len(twopeaks)):
            # try:
                # ax.plot(twopeaks[i][1],twopeaks[i][0], 'bo', markersize = 10)
            # except IndexError:
                # pass
            # try:
                # ax.plot(twonegpeaks[i][1],twonegpeaks[i][0], 'ro', markersize = 10)
            # except IndexError:
                # pass
        # pylab.show()
        # derpp = raw_input('---')
        # if derpp is 'q':
            # sys.exit("quitting")
        # pylab.close()
			
    kwrvals['R'] = sorted(list(frozenset(r_beats)))#/SamplingRate if SamplingRate else beats
    
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
	res = engzee(Signal=yfir[:win]-1.*scipy.mean(yfir[:win]), SamplingRate=SamplingRate, initialfilter=False, Show=True)
	dlen = len(yfir)
	while ( mrkrend < dlen ):    
		mrkrend += win
		print mrkrend
		mrkr = res['Params']['offset'][-1]
		res = engzee(Signal=yfir[mrkr:mrkrend]-1.*scipy.mean(yfir[mrkr:mrkrend]), SamplingRate=SamplingRate, initialfilter=False, Params=res['Params'], Show=True)
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
	
def engzee(Signal=None, SamplingRate=1000., initialfilter=True, Filter={}, Params={}, Show=False):
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
	c = [1, 4, 6, 4, 1, -1, -4, -6, -4, -1]
	y2 = scipy.array(map(lambda n: scipy.dot(c,y1[n-9:n+1]), range(9,len(y1))))
	# Define threshold
	#--------------------------FRANCIS--------------------------
	changeM = 0.75*SamplingRate
	Miterate = 1.75*SamplingRate					# constant defining the multiplier to which max y2 will be checked
	inc = 1
	update = False
	#----------------------END OF FRANCIS-----------------------
	mmth=0.48			# 0.48 for Santa Marta // maybe change to 0.53 or 0.55 to avoid some FPs!
	mmp = 0.2
	MM = Params['MM'] if Params.has_key('MM') else mmth*max(y2[:Miterate])*scipy.ones(3)
	MMidx = Params['MMidx'] if Params.has_key('MMidx') else 0
	NN = Params['NN'] if Params.has_key('NN') else mmp*min(y2[:Miterate])*scipy.ones(2)
	NNidx = Params['NNidx'] if Params.has_key('NNidx') else 0
	Th = scipy.mean(MM) if Method is 'online' else 0.6*max(y2[:Miterate])            # paper: mmp=0.2, modification: mmp=0.7
	ThNew = scipy.mean(NN) if Method is 'online' else 0.7*min(y2[:Miterate])
	Th_iter = numpy.arange(0.0, len(y2), changeM)
	Th_vec = [Th]
	ThNew_vec = [ThNew]
	# due to iteration in windows, sometimes, a peak is not detected when lying in window borders. this sample-advance tries to deal with it
	err_kill = int(SamplingRate/100.)
	# Init variables
	v250ms = 0.25*SamplingRate
	v1200ms = 1.2*SamplingRate
	v180ms = int(0.180*SamplingRate)
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
		if update:
			if inc*changeM + Miterate < len(y2):
				Mnew = mmth*max(y2[(inc-1)*changeM:inc*changeM + Miterate])
				Nnew = mmp*min(y2[(inc-1)*changeM:inc*changeM + Miterate])
			elif len(y2) - (inc-1)*changeM > 1.5*SamplingRate :
				Mnew = mmth*max(y2[(inc-1)*changeM:len(y2)])
				Nnew = mmp*min(y2[(inc-1)*changeM:len(y2)])
			# print '-------------------------'								#---zdec---
			# print 'here', (inc-1)*changeM, inc*changeM+Miterate, len(y2)-(inc-1)*changeM,'\n'
			# print '-------------------------'
			if len(y2)-inc*changeM > Miterate:
				MM[MMidx] = Mnew if Mnew <= 1.5*MM[MMidx-1] else 1.1*MM[MMidx-1]
				NN[NNidx] = Nnew if abs(Nnew) <= 1.5*abs(NN[NNidx-1]) else 1.1*NN[NNidx-1]
			MMidx = scipy.mod(MMidx+1,len(MM))
			NNidx = scipy.mod(NNidx+1,len(NN))
			Th = scipy.mean(MM)
			ThNew = scipy.mean(NN)
			Th_vec += [Th]
			ThNew_vec += [ThNew]
			inc += 1
			update = False
		if nthfpluss:
			lastp = nthfpluss[-1]+1-offset[-1]
			if lastp < (inc-1)*changeM:
				lastp = (inc-1)*changeM
			y22 = y2[lastp:inc*changeM+err_kill]
			# print '++++++++++++++++++++'
			# if (inc-1)*changeM >= 3000 or (inc-1)*changeM <= 3375:
				# print 'signal: ', y2[3370:3380]
				# print 'thres:', Th
			# print lastp, ' + ', (inc-1)*changeM, ' + ', inc*changeM					#---zdec---
			# print 'vec', nthfpluss
			# raw_input('?')
			# find intersection with Th
			try:
				nthfplus = scipy.intersect1d(pylab.find(y22>Th),pylab.find(y22<Th)-1)[0]
			except IndexError:
				if inc*changeM > len(y2):
					break
				else:
					update = True
					continue	
			# adjust index
			# nthfplus += nthfpluss[-1]+1
			nthfplus += int(lastp)
			# if a previous R peak was found: 
			if rpeak and Method is 'online':
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
				nthfplus = int((inc-1)*changeM) + scipy.intersect1d(pylab.find(y2[ (inc-1)*changeM:inc*changeM+err_kill ] > Th),pylab.find(y2[ (inc-1)*changeM:inc*changeM + err_kill] < Th)-1)[0]
			except IndexError:
				if inc*changeM > len(y2):
					break
				else:
					update = True
					continue				
		nthfpluss += [nthfplus] # plot
	# Define 160ms search region
		windowW = scipy.arange(nthfplus, nthfplus+v180ms)
		windowW -= offset[-1]
		# "Check if the condition y2[n] < Th holds for a specified 
		# number of consecutive points (experimentally we found this number to be at least 10 points)"
		i,f = windowW[0], windowW[-1] if windowW[-1]<len(y2) else -1
		hold_points = scipy.diff(pylab.find(y2[i:f]<ThNew))
		cont=0
		for hp in hold_points:
			if hp == 1:
				cont+=1
				if cont == int(scipy.ceil(SamplingRate/100.))-1:                                # NOTE: NEEDS TO ADAPT TO SAMPLING RATE (-1 is because diff in line 936 eats a sample
					# "Upon finding a candidate R peak, the original signal,
					# x[n] is scanned inside the obtained window, 
					# and the peak is determined as the time instant corresponding
					# to the highest amplitude signal"
					#if all(MM>max(x[i:f])):
					max_shift = int(scipy.ceil(SamplingRate/50.))			# looks for X's max a bit to the right
					if nthfpluss[-1] - offset[-1] > max_shift:
						rpeak += [scipy.argmax(x[i-max_shift:f])+i-max_shift+offset[-1]]
					else:
						rpeak += [scipy.argmax(x[i:f])+i+offset[-1]]
					a,b = Rminus, Rplus
					cur_r_peak = rpeak[-1]
					# print 'peak', cur_r_peak					#---zdec---
					# print max(x[i:f]), scipy.mean(x), max(x[i:f])-min(x[i:f]), scipy.std(x), abs(scipy.mean(scipy.diff(x)))
					break
			else:
				cont=0
			# print hp, cont
	if Show:
		pylab.figure(787)
		pylab.figure(787).clf()
		pylab.plot(x, 'b', label='x')
		pylab.plot(y2, 'g', label='y2')
		pylab.grid('on')        
		pylab.axis('tight')
		pylab.plot(scipy.array(nthfpluss[nits:])-offset[-1], y2[scipy.array(nthfpluss[nits:])-offset[-1]],'ko', label='nthfplus')
		pylab.plot(scipy.arange(i,i+len(y2[i:f])), y2[i:f], 'm')
		if(scipy.any(scipy.array(rpeak[nbeats:])-offset[-1])): pylab.plot(scipy.array(rpeak[nbeats:])-offset[-1], x[scipy.array(rpeak[nbeats:])-offset[-1]], 'r^', ms=10, label='rpeak')
		for j in range(len(Th_iter)):
			if j < len(Th_iter)-1:
				range_aid = range(int(Th_iter[j]), int(Th_iter[j+1]))
				pylab.plot(range_aid, Th_vec[j]*scipy.ones(len(range_aid)),'r--', label='Th')
				pylab.plot(range_aid, ThNew_vec[j]*scipy.ones(len(range_aid)),'k--', label='ThNew')
			else:
				range_aid = range(int(Th_iter[j]), len(y2))
				pylab.plot(range_aid, Th_vec[j]*scipy.ones(len(range_aid)),'r--', label='Th')
				pylab.plot(range_aid, ThNew_vec[j]*scipy.ones(len(range_aid)),'k--', label='ThNew')
		pylab.legend(('X', 'Y1', 'Intersects', 'Window', 'R-peaks', 'M', 'N'), 'best', shadow=True)
		pylab.xlabel('Time (samples)')
		ax = pylab.figure(787).add_subplot(111)
		# ax.axes.yaxis.set_visible(False)
		# ax.axes.xaxis.set_visible(False)
		pylab.show()
		if raw_input('_')=='q': herf=1

	# pylab.figure(987).clf()
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
						'NN': NN,
						'NNidx': NNidx,
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

def christov(Signal=None, hand=False, SamplingRate=1000., Filter={}, Params={}, Show=False, dec_R=3., M_th=0.48, dec_F=350., L=1):
	"""

	Determine ECG signal information.

	Kwargs:
		Signal (array): input ECG signal.
		
		SamplingRate (float): Sampling frequency (Hz).
		
		Filter (dict): Filter parameters.

		Params (dict): Initial conditions. 

		dec_R (float): tuning constant for adaptive threshold's R component
		
		dec_F (float): tuning constant for adaptive threshold's F component
		
		M_th (float): tuning constant for adaptive threshold's M_th component
		
		L (int): number of leads

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
	# for n in range(k+1, L-k): Y.append(X[n]**2-X[n-k]*X[n+k])				#---zdec---	
	for n in range(k+1, L-k): Y.append( abs(X[n+k]-X[n-k]) )				# this is the expression in the paper. above one unknown
	Y = scipy.array(Y)
	# Y[pylab.find(Y < 0)] = 0					#---zdec---
	# Complex lead
	# Y = abs(scipy.diff(X)) # 1-lead
	# 3. Moving averaging of a complex lead (the sintesis is
	# explained in the next section) in 40 ms intervals a filter
	# with first zero at about 25 Hz. It is suppressing the noise
	# magnified by the differentiation procedure used in the
	# process of the complex lead sintesis.        
	Y = ss.lfilter(scipy.ones(SamplingRate/25.)/25.,[1],Y)
	# Init 
	if hand:
		start = 2.
	else:
		start = 0
	init_ecg = 6                                                #seconds for initialization
	qrspeakbuffer = scipy.zeros(init_ecg)
	indexqrs = 0
	a,b = start*SamplingRate, (start+1)*SamplingRate
	# print all_peaks
	for i in range(0,init_ecg):
		peaks = peakd.sgndiff(Y[a:b])['Peak']
		try:
			qrspeakbuffer[i] = max(Y[a:b][peaks])   		#peak amplitude
		except Exception as e:
			print e
		a += SamplingRate
		b += SamplingRate
	v50ms = int(0.050*SamplingRate)
	v300ms = int(0.300*SamplingRate)
	v350ms = int(0.350*SamplingRate)
	v200ms = 0.2*SamplingRate
	v1200ms = 1.2*SamplingRate
	# M_th = 0.6 # paper is 0.6
	# MM = Params['M'] if Params.has_key('M') else M_th*max(Y[:5*SamplingRate])*scipy.ones(5) # 0.6
	MM = Params['M'] if Params.has_key('M') else M_th*scipy.median(qrspeakbuffer)*scipy.ones(5) # for hand signals
	# MMidx = Params['MMidx'] if Params.has_key('MMidx') else 0
	MMidx = Params['MMidx'] if Params.has_key('MMidx') else 0
	M = scipy.mean(MM)
	slope = scipy.linspace(1.0,0.6,int(SamplingRate))
	slopeR = scipy.linspace(1.0,0.6*1.4,int(SamplingRate))
	Rdec = (slopeR[1]-slopeR[0])*dec_R
	R = Params['Rth'] if Params.has_key('Rth') else 0
	RR = Params['RR'] if Params.has_key('RR') else SamplingRate/2*scipy.ones(5)
	RRidx = Params['RRidx'] if Params.has_key('RRidx') else 0
	Rm = scipy.mean(RR)
	QRS = Params['QRS'] if Params.has_key('QRS') else []
	Rpeak = Params['R'] if Params.has_key('R') else []
	offset = Params['offset'] if Params.has_key('offset') else 0
	current_sample = int(start*SamplingRate)					# for hand signals
	# current_sample = 0
	skip = False
	repetition = False
	crap = -(v200ms-offset)-1
	if(Show): M_hist,F_hist, R_hist, MFR_hist = [], [], [], [] # to plot
	F =  scipy.mean(Y[int(start*SamplingRate):int(start*SamplingRate + v350ms)])
	if Params.has_key('Y_350'): Y_temp = scipy.concatenate((Params['Y_350'], Y))
	nbeats = Params['nbeats'] if Params.has_key('nbeats') else 0
	# Go through each sample
	# print qrspeakbuffer, 2.25*scipy.median(qrspeakbuffer)
	while current_sample < len(Y):
		if current_sample <= crap+v200ms-offset:
				skip = True
		if QRS:
			# No detection is allowed 200 ms after the current one. In
			# the interval QRS to QRS+200ms a new value of M5 is calculated: newM5 = 0.6*max(Yi)
			if current_sample <= QRS[-1]+v200ms-offset:
				Mnew = M_th*max(Y[QRS[-1]-offset:QRS[-1]+v200ms-offset]) # 0.6
				# The estimated newM5 value can become quite high, if
				# steep slope premature ventricular contraction or artifact
				# appeared, and for that reason it is limited to newM5 = 1.1*M5 if newM5 > 1.5* M5
				# The MM buffer is refreshed excluding the oldest component, and including M5 = newM5.
				Mnew = Mnew if Mnew <= 1.5*MM[MMidx] else 1.1*MM[MMidx]
				MM[MMidx] =  Mnew
				MMidx = scipy.mod(MMidx+1,5)
				# M is calculated as an average value of MM.
				M = scipy.mean(MM)
				M_ref = M
				skip = True
			# M is decreased in an interval 200 to 1200 ms following
			# the last QRS detection at a low slope, reaching 60 % of its
			# refreshed value at 1200 ms.
			elif current_sample >= QRS[-1]+v200ms-offset and current_sample < QRS[-1]+v1200ms-offset:
				M = M_ref*slope[current_sample - QRS[-1] - int(v200ms) + offset]
			# After 1200 ms M remains unchanged.
			# R = 0 V in the interval from the last detected QRS to 2/3 of the expected Rm.
				if current_sample >= QRS[-1]-offset and current_sample < QRS[-1]+(2/3.)*Rm-offset:
					R = 0
				# In the interval QRS + Rm * 2/3 to QRS + Rm, R decreases
				# 1.4 times slower then the decrease of the previously discussed
				# steep slope threshold (M in the 200 to 1200 ms interval).
				elif current_sample >= QRS[-1]+(2/3.)*Rm-offset and current_sample < QRS[-1]+Rm-offset:
					R += Rdec*M_ref
				# After QRS + Rm the decrease of R is stopped
		elif current_sample >= (2/3.)*Rm-offset and current_sample < Rm-offset:				#---francis---
			R += Rdec*M
		# MFR = M + F + R
		MFR = M + F +  R
		# QRS or beat complex is detected if Yi = MFR
		if not skip and Y[current_sample] >= MFR:
			QRS +=[current_sample+offset]
			# print QRS
			Rpeak += [QRS[-1]+scipy.argmax(Y[QRS[-1]: QRS[-1]+v200ms])]	
			# print Rpeak, Y[Rpeak[-1]]
			if not(hand) or Y[Rpeak[-1]] < 2.25*scipy.median(qrspeakbuffer):
				if len(Rpeak) >= 2:
					# print Rpeak																#---zdec---
					#----------------------FRANCIS----------------------
					if(Rpeak[-1] - Rpeak[-2] < 0.25*SamplingRate):
						if(Y[Rpeak[-1]] > Y[Rpeak[-2]]): Rpeak.pop(len(Rpeak)-2)
						else: Rpeak.pop(len(Rpeak)-1)
						repetition = True
					#------------------END OF FRANCIS-------------------
					# A buffer with the 5 last RR intervals is updated at any new QRS detection.     
					# print Rpeak
					if len(Rpeak) >= 2:
						RR[RRidx] = Rpeak[-1]-Rpeak[-2]
						RRidx = scipy.mod(RRidx+1,5)
				if hand and not(repetition):
					# print current_sample
					# print 2.25*scipy.median(qrspeakbuffer)
					# print qrspeakbuffer
					# print'--------------------'
					qrspeakbuffer[indexqrs] = Y[Rpeak[-1]]
					indexqrs += 1
					if(indexqrs == init_ecg): indexqrs = 0
			else:
				Rpeak.pop()
				crap = QRS.pop()
				# print crap
		skip = False
		repetition = False
		# With every signal sample, F is updated adding the maximum
		# of Y in the latest 50 ms of the 350 ms interval and
		# subtracting maxY in the earliest 50 ms of the interval.
		if current_sample >= start*SamplingRate + v350ms-offset:
			if current_sample < start*SamplingRate + v350ms and offset != 0:
				Y_latest50 = Y_temp[current_sample+v300ms:current_sample+v350ms]
				Y_earliest50 = Y_temp[current_sample:current_sample+v50ms]
			else: 
				Y_latest50 = Y[current_sample-v50ms:current_sample]
				Y_earliest50 = Y[current_sample-v350ms:current_sample-v300ms]
			F += (max(Y_latest50) - max(Y_earliest50))/dec_F
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
		# if(current_sample >= 5.2*SamplingRate and current_sample <= 5.25*SamplingRate):
			# print '--------------', current_sample-1,'--------------'
			# print MM
			# print 'M', M
		if(Show):    
			M_hist += [M]
			F_hist += [F]
			R_hist += [R]
			MFR_hist += [MFR]
	if(Show):
		range_help = range(int(start*SamplingRate),len(Y))
		fig = pylab.figure(777)
		pylab.plot(range(len(Y)), Y, 'b', lw=2)
		# print len(t[int(start*SamplingRate):]), len(M_hist), len(F_hist), len(R_hist), len(MFR_hist)
		pylab.plot(range_help, M_hist, 'r', lw=1)
		pylab.plot(range_help, F_hist, 'g', lw=1)
		pylab.plot(range_help, R_hist, 'y', lw=1)
		pylab.plot(range_help, MFR_hist, 'k', lw=2)
		pylab.plot(scipy.array(QRS[nbeats:])-offset, Y[scipy.array(QRS[nbeats:])-offset], 'bo', markersize=5)
		pylab.plot(scipy.array(Rpeak[nbeats:])-offset, Y[scipy.array(Rpeak[nbeats:])-offset], 'ro', markersize=10)
		pylab.legend(('Y', 'M', 'F', 'R', 'MFR', 'Intersects', 'R-peaks'), 'best', shadow=True)
		pylab.xlabel('Time (samples)')
		pylab.grid('on')
		pylab.show()
		derp = raw_input('PRESS ENTER')
		pylab.close(fig)
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

def test_real_time():
	# Get Data
	from database import mongoH5
	
	config = {
			  'dbName': 'CruzVermelhaPortuguesa',
			  'host': '193.136.222.234',
			  'port': 27017,
			  'path': r'\\193.136.222.220\cybh\data\CVP\hdf5'}
	db = mongoH5.bioDB(**config)
	recs = mongoH5.records(**db)
	ids_recs = recs.getAll()['idList']
	
	raw = recs.getData(ids_recs[1], 'ECG/hand/raw', 0)['signal']
	
	step = 150
	overlap = 300
	SamplingRate = 1000.
	bfir1 = ss.firwin(overlap+1,[2*1./SamplingRate, 2*20./SamplingRate], pass_zero=False)
	y = scipy.zeros(overlap)
	
	pylab.figure(1)
	
	for i in xrange(100):
	
		frameArray = scipy.copy(raw[i*step:(i+1)*step])
		frameArray -= scipy.mean(frameArray)
		ffr = scipy.convolve(frameArray, bfir1)
		y[-overlap:] += ffr[:overlap]
		y = scipy.concatenate((y,ffr[overlap:]))
		
		# y = y[step:]
		
		# ysc = scipy.copy(y)
		# ysc = ysc[:step]
		
		pylab.cla()
		pylab.plot(y)
		pylab.show()
		
	return


if __name__=='__main__':
	
	option = ''
	
	while option != 'q':
	
		# Menu
		option = raw_input('Choose an algorithm:\n\t1:%s\n\t2:%s\n\t3:%s\n\t4:%s\n\t5:%s\n\n\tq:%s\n>'%('Hamilton', 'EngZee', 'Christov', 'ESSF', 'Gamboa', 'quit'))
		
		if option is '1':
			algorithm = 'Hamilton'
		elif option is '2':
			algorithm = 'EngZee'
		elif option is '3':
			algorithm = 'Christov'
		elif option is '4':
			algorithm = 'ESSF'
		elif option is '5':
			algorithm = 'Gamboa'
		elif option is 'q':
			break    
		else:
			raise TypeError, "Not an option."
				
		# Get Data
		from database import mongoH5
		
		config = {'dbName': 'CruzVermelhaPortuguesa',
				  'host': '193.136.222.234',
				  'port': 27017,
				  'path': r'\\193.136.222.220\cybh\data\CVP\hdf5'}
		db = mongoH5.bioDB(**config)
		recs = mongoH5.records(**db)
		ids_recs = recs.getAll()['idList']
		raw = recs.getData(ids_recs[1], 'ECG/hand/raw', 0)['signal'][:10000]
		
		
		# Filter
		SamplingRate = 1000.
		fn, lpc, hpc = 301, 5.0, 20.
		bfir1 = ss.firwin(fn,[2*lpc/SamplingRate, 2*hpc/SamplingRate], pass_zero=False)
		filteredsignal = ss.lfilter(bfir1, [1], raw)
		filteredsignal = filteredsignal[fn:]
		filteredsignal -= scipy.mean(filteredsignal)
	 
		# Process
		if algorithm is 'Hamilton':
			# Hamilton algorithm
			res = hamilton(Signal = filteredsignal, SamplingRate=SamplingRate)
			rpeaks = res['R']
		elif algorithm is 'EngZee':
			res = batch_engzee(Signal=filteredsignal, SamplingRate=SamplingRate, debug=False, IF=False)
			segs = res['Segments']
			rpeaks = res['R']
		elif algorithm is 'Christov':
			res = batch_christov(Signal=filteredsignal, SamplingRate=SamplingRate, debug=False, IF=False)
			rpeaks = res['R']
		elif algorithm is 'ESSF':
			rpeaks = ESSF(filteredsignal, SamplingRate, ESSF_params = {'s_win': 0.3, 's_amp': 70.})
		elif algorithm is 'Gamboa':
			rpeaks = gamboa(filteredsignal, SamplingRate, 0.002)
	 
		# Plot
		segs = []
		for rpeak in rpeaks:
			a = rpeak-200
			b = rpeak+400
			if a>=0 and b<=len(filteredsignal) and b-a==600:
				seg = filteredsignal[a:b]
				if len(seg) == 600: segs.append(seg)
		segs = scipy.array(segs)     
		fig = pylab.figure(1)
		ax = fig.add_subplot(211)
		ax.cla()
		ax.plot(filteredsignal)
		ax.vlines(rpeaks, min(filteredsignal), max(filteredsignal), 'r', lw=3)
		ax.axis('tight')
		ax.set_title('R-Peak Detection: %s algorithm'%(algorithm))
		pylab.grid()
		ax.legend(('ECG', 'R'))
		ax = fig.add_subplot(212)
		ax.cla()
		ax.plot(segs.T, 'k')
		ax.axis('tight')
		ax.set_xlabel('Time (ms)')
		ax.set_title('ECG Heartbeats')
		pylab.grid()
		
		# Save to temp
		pylab.savefig('../temp/fig1.png')
		
		print "Done. Results saved in /temp/fig1.png"
		
		