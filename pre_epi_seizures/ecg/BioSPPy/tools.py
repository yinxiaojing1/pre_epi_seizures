'''
Created on May 28, 2013

@author: Carlos
'''


import numpy as np
import pylab as pl
import scipy.signal as ss


def synchronize(a, v, flag=False):
	# returns the delay (number of samples) of a in relation to v
	# delay < 0 => a is ahead in relation to v
	# delay > 0 => a is delayed in relation to v
	
	corr = np.correlate(a, v, mode='full')
	# corr = np.abs(corr)
	x = np.arange(-len(v) + 1, len(a))
	
	if flag:
		return x[np.argmax(corr)], x, corr
	else:
		return x[np.argmax(corr)]


def getSyncSlices(d, a, v):
	# get sync slices
	
	if d < 0:
		c = min([len(a), len(v[-d:])])
		sla = slice(0, c, None)
		slv = slice(-d, -d + c)
	elif d > 0:
		c = min([len(v), len(a[d:])])
		slv = slice(0, c, None)
		sla = slice(d, d + c)
	else:
		c = min([len(a), len(v)])
		sla = slice(0, c, None)
		slv = slice(0, c, None)
	
	return sla, slv


def syncOverlap(a, v):
	# return the slices of each input signal in which there is overlap
	
	# get the delay
	d = synchronize(a, v)
	
	return getSyncSlices(d, a, v)


def PLFSynchronize(a, v, flag=False, minOverlap=1):
	# synchronization using the Phase-Locking Factor
	
	# lengths
	la = len(a)
	lv = len(v)
	
	# check compatibility
	minL = min([la, lv])
	if minOverlap > minL:
		raise ValueError("Input minOverlap must be <= min([len(a), len(v)]) = %d." % minL)
	
	# delays
	x = np.arange(-len(v) + minOverlap, len(a) - minOverlap + 1)
	lx = len(x)
	ra = np.arange(la, dtype='int')
	rv = np.arange(lv, dtype='int')
	
	# analytic signals
	aa = ss.hilbert(a)
	av = ss.hilbert(v)
	
	# phase
	pa = np.angle(aa)
	pv = np.angle(av)
	
	plf = np.zeros(lx)
	for i in xrange(lx):
		ina = np.intersect1d(ra, rv + x[i], assume_unique=True)
		inv = ina - x[i]
		plf[i] = np.absolute(np.mean(np.exp(1j * (pa[ina] - pv[inv]))))
	
	if flag:
		return x[np.argmax(plf)], x, plf
	else:
		return x[np.argmax(plf)]


def PLFCorrSynchronize(a, v, flag=False):
	# synchronization using the Phase-Locking Factor and correlation
	
	# PLF
	_, x, plf = PLFSynchronize(a, v, flag=True, minOverlap=1)
	
	corr = np.correlate(a, v, mode='full') * plf
	
	if flag:
		return x[np.argmax(corr)], x, corr
	else:
		return x[np.argmax(corr)]


def _extractHeartbeats(signal, R, before, after):
	# private beat extractor
	
	R = np.sort(R)
	length = len(signal)
	segs = []
	newR = []
	
	for r in R:
		a = r - before
		if a < 0:
			continue
		b = r + after
		if b > length:
			break
		segs.append(signal[a:b])
		newR.append(r)
	
	segs = np.array(segs)
	newR = np.array(newR, dtype='int')
	
	return segs, newR


def extractHeartbeats(signal, R, sampleRate, before=0.2, after=0.4):
	# extract the desired heartbeats from the signal
	
	# check inputs
	if before < 0:
		raise ValueError, "Please specify a non-negative 'before' value."
	if after < 0:
		raise ValueError, "Please specify a non-negative 'after' value."
	
	# convert delimiters to samples
	before = int(before * sampleRate)
	after = int(after * sampleRate)
	
	return _extractHeartbeats(signal, R, before, after)


def checkECG(data):
	#returns True if the ecg is correct or False if it is upside down
	#data must be filtered and align in zero
	
	hist,bins = np.histogram(data, bins = 100)
	mx = np.argmax(hist)
	m1 = mx - pl.find(hist[:mx] >=10)[0]
	m2 = pl.find(hist[mx:] <=10)
	if len(m2) == 0:
		m2 = len(hist[mx:])
	else:
		m2 = m2[0]
	if (m1-m2) > 0:
		check = False
	else: check = True

	return check, hist, bins


def compareSegmentation(referenceR=None, testR=None, SamplingRate=None, offset=0, minRR=None, tol=0.05):
	# compare the segmentation performance of a list of R positions against a reference list
	# R lists are array indexes => both sets acquired at the same sampling frequency
	
	# check inputs
	if referenceR is None:
		raise TypeError, "Please provide the reference set of R positions."
	if testR is None:
		raise TypeError, "Please provide the test set of R positions."
	if SamplingRate is None:
		raise TypeError, "Please provide the samplig rate."
	if minRR is None:
		minRR = np.inf
	
	# ensure numpy
	referenceR = np.array(referenceR)
	testR = np.array(testR)
	
	# convert to samples
	minRR = minRR * SamplingRate
	tol = tol * SamplingRate
	
	TP = 0
	FP = 0
	
	matchIdx = []
	dev = []
	
	for i, r in enumerate(testR):
		# deviation to closest R in reference
		error = np.abs(referenceR[np.argmin(np.abs(referenceR - (r + offset)))] - (r + offset))
		
		if error < tol:
			TP += 1
			matchIdx.append(i)
			dev.append(error)
		else:
			if len(matchIdx) > 0:
				bdf = r - testR[matchIdx[-1]]
				if bdf < minRR:
					# false positive, but removable with RR interval check
					pass
				else:
					FP += 1
			else:
				FP += 1
	
	# convert deviations to time
	dev = np.array(dev, dtype='float')
	dev /= SamplingRate
	nd = len(dev)
	if nd == 0:
		mdev = np.nan
		sdev = np.nan
	elif nd == 1:
		mdev = np.mean(dev)
		sdev = 0.
	else:
		mdev = np.mean(dev)
		sdev = np.std(dev, ddof=1)
	
	# interbeat interval
	th1 = 1.5
	th2 = 0.3
	
	rIBI = np.diff(referenceR)
	rIBI = np.array(rIBI, dtype='float')
	rIBI /= SamplingRate
	
	good = np.nonzero((rIBI < th1) & (rIBI > th2))[0]
	rIBI = rIBI[good]
	
	nr = len(rIBI)
	if nr == 0:
		rIBIm = np.nan
		rIBIs = np.nan
	elif nr == 1:
		rIBIm = np.mean(rIBI)
		rIBIs = 0.
	else:
		rIBIm = np.mean(rIBI)
		rIBIs = np.std(rIBI, ddof=1)
	
	tIBI = np.diff(testR[matchIdx])
	tIBI = np.array(tIBI, dtype='float')
	tIBI /= SamplingRate
	
	good = np.nonzero((tIBI < th1) & (tIBI > th2))[0]
	tIBI = tIBI[good]
	
	nt = len(tIBI)
	if nt == 0:
		tIBIm = np.nan
		tIBIs = np.nan
	elif nt == 1:
		tIBIm = np.mean(tIBI)
		tIBIs = 0.
	else:
		tIBIm = np.mean(tIBI)
		tIBIs = np.std(tIBI, ddof=1)
	
	# output
	kwrvals = {}
	kwrvals['TP'] = TP
	kwrvals['FP'] = FP
	kwrvals['Performance'] = float(TP) / len(referenceR)
	kwrvals['Acc'] = float(TP) / (TP + FP)
	kwrvals['Err'] = float(FP) / (TP + FP)
	kwrvals['match'] = matchIdx
	kwrvals['deviations'] = dev
	kwrvals['meanDeviation'] = mdev
	kwrvals['stdDeviation'] =  sdev
	kwrvals['meanRefIBI'] = rIBIm
	kwrvals['stdRefIBI'] = rIBIs
	kwrvals['meanTestIBI'] = tIBIm
	kwrvals['stdTestIBI'] =tIBIs
	
	return kwrvals

