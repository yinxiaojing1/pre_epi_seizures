"""
.. module:: bitalinoProcess
   :platform: Windows
   :synopsis: BITalino acquisition process.

.. moduleauthor:: Filipe Canento, Carlos Carreiras


"""

# Imports
# built-in
import collections
import time
from multiprocessing import Event, Pipe, Process, Queue

# 3rd party
import numpy as np

# BioSPPy
import filt
from ecg import models

# BITalino
from bitalino import BITalino

# BiometricsPyKit
from misc import misc



class BITalinoProcess(Process):
    """
    BITalino acquisition process.
    """
    
    def __init__(self, outQueue, goFlag, acqFlag, mac=None,
                 channels=[0], step=100, SamplingRate=1000,
                 timeout=5, bitCls=None):
        # run parent __init__
        super(BITalinoProcess, self).__init__()
        
        # synchronization inputs
        self.queue = outQueue
        self.go = goFlag
        self.acq = acqFlag
        
        # BITalino settings
        self.mac = mac
        self.channels = channels
        self.step = step
        self.SamplingRate = int(SamplingRate)
        if bitCls is None:
            self.device = BITalino()
        else:
            self.device = bitCls()
        
        # trigger stuff
        c1, c2 = Pipe()
        self._outerPipe = c1
        self._innerPipe = c2
        self._trigout = 2 * self.step / float(self.SamplingRate)
        
        # timeout
        self.timeout = timeout
    
    @classmethod
    def instaStart(cls, *args, **kwargs):
        # do some class method magic here to instantiate and start the process
        
        outQueue = Queue()
        goFlag = Event()
        goFlag.set()
        acqFlag = Event()
        
        p = cls(outQueue, goFlag, acqFlag, *args, **kwargs)
        p.start()
        
        return p, outQueue, goFlag, acqFlag
    
    def _checkTrigger(self):
        # check if there is a trigger to act on
        
        if self._innerPipe.poll():
            data = self._innerPipe.recv()
            
            try:
                mask = data['Mask']
            except KeyError:
                # report failure
                self._innerPipe.send({'Ack': False})
            else:
            # trigger
                self.device.trigger(mask)
                
                # report success
                self._innerPipe.send({'Ack': True})
    
    def trigger(self, data=[0, 0, 0, 0]):
        # act on digital outputs
        
        if not self.acq.is_set():
            return False
        
        # send mask
        self._outerPipe.send({'Mask': data})
        
        # wait for ack
        
        if self._outerPipe.poll(self._trigout):
            out = self._outerPipe.recv()
            return out['Ack']
        else:
            return False
    
    def connect(self):
        # connect to BITalino
        
        if self.device.open(self.mac, self.SamplingRate):
            time.sleep(1.5)
            print "successful"
            return True
        else:
            print "failed"
            return False
    
    def setup(self):
        # setup bitalino
        
        print "connecting to %s: " % self.mac
        
        while self.go.is_set():
            if self.connect():
                return True
            time.sleep(1.5)
        
        return False
    
    def on_start(self):
        # run immediately after acquisition starts
        
        pass
    
    def exit(self):
        # exit process
        
        self.acq.clear()
        self.go.clear()
    
    def processData(self, data):
        # process received data
        
        # send data
        self.queue.put(data)
    
    def on_stop(self):
        # run immediately before acquisition stops
        
        pass
    
    def run(self):
        # main loop
        
        # connect to device, proceed if successful
        if self.setup():
            while self.go.is_set():
                # wait for acquisition flag
                print "Waiting"
                self.acq.wait(timeout=self.timeout)
                
                if self.acq.is_set():
                    # start acquisition
                    self.device.start(self.channels)
                    # time.sleep(1)
                    print "acquiring"
                    
                    self.on_start()
                    
                    while self.acq.is_set():
                        # read data
                        data = self.device.read(self.step)
                        
                        # process data
                        self.processData(data)
                        
                        # check trigger
                        self._checkTrigger()
                    
                    # clean up
                    self.on_stop()
                    self.queue.put('')
                    
                    # stop acquisition
                    self.device.stop()
                    # print "blocked"
                    print "acquisition stopped"
            
            # disconnect device
            self.device.close()
            print "disconnected"


class CheckHands(object):
    """
    Class to check if hands are on the ECG electrodes.
    """
    
    def __init__(self, step=150, size=300, pad=256, signalFc=None, highFc=None,
                 SamplingRate=1000., nbits=10, tol=24, delay=3):
        
        # check inputs
        if highFc is None:
            highFc = [45, 65]
        if signalFc is None:
            signalFc = [5, 20]
        
        # self inputs
        self.step = step
        self.size = size
        self.pad = pad
        self.SamplingRate = SamplingRate
        self.nints = 2 ** nbits
        self.tol = tol
        if delay >= 0:
            self.delay = [0] + delay * [-1] + [1]
        else:
            self.delay = [0, 1]
        
        # frequencies
        freqs = np.linspace(0, self.SamplingRate/2, self.pad + 1)
        self.highNoiseBand = (highFc[0] <= freqs) * (freqs <= highFc[1])
        self.signalBand = (signalFc[0] <= freqs) * (freqs <= signalFc[1])
        
        # reset buffer
        self.reset()
    
    def reset(self):
        # reset buffer
        
        self.handsOn = False
        self.handValue = misc.BDIterator(self.delay)
        self.buffer = np.array([], dtype='int')
    
    def check(self, signal):
        # check if hands are on with given signal fragment
        
        # previous value
        prev = self.handsOn
        
        # make sure we have ints
        signal = signal.astype('int')
        
        # add to buffer
        self.buffer = np.hstack((self.buffer, signal[-self.step:]))
        
        if len(self.buffer) >= self.size:
            aux = self.buffer[:]
            
            counts = np.bincount(aux, minlength=self.nints) / float(self.size)
            bc = np.sum(counts[:self.tol])
            tc = np.sum(counts[self.nints-self.tol:self.nints])
            
            if bc > 0.3:
                # print "hands: bottom limit"
                res = False
            elif tc > 0.3:
                # print "hands: top limit"
                res = False
            else:
                bux = aux - aux.mean()
                ft = np.fft.fft(bux, n=2*self.pad)
                ft = np.abs(ft[:self.pad + 1]) / (self.pad + 1)
                
                sft = np.sum(ft)
                # print "hands: total power", sft
                if sft > 10:
                    ft /= sft
                    pS = np.sum(ft[self.signalBand])
                    # print "hands: signal power", pS
                    if pS > 0:
                        vH = np.sum(ft[self.highNoiseBand]) / pS
                        # print "hands: noise power", pS
                        
                        if vH < 1.0:
                            res = True
                        else:
                            res = False
                        
                    else:
                        # print "hands: no signal power"
                        res = False
                else:
                    # print "hands: no power"
                    res = False
            
            if res:
                val = self.handValue.next()
            else:
                val = self.handValue.prev()
            
            if val == 1:
                self.handsOn = True
            elif val == 0:
                self.handsOn = False
            
            # roll
            self.buffer = self.buffer[-self.step:]
        
        return self.handsOn != prev


class CheckHands2(object):
    """
    Class to check if hands are on the ECG electrodes.
    """
    
    HF = [45, 65]
    PAD = 512
    
    TH_BV = 0.0234375
    TH_BP = 0.3
    TH_TL = 0.00002
    TH_TH = 0.02
    TH_HF = 0.6
    
    def __init__(self, step=150, size=300, SamplingRate=1000., nbits=10, delay=3):
        # init
        
        # self inputs
        self.step = step
        self.size = size
        self.SamplingRate = SamplingRate
        if delay >= 0:
            self.delay = [0] + delay * [-1] + [1]
        else:
            self.delay = [0, 1]
        
        # histogram params
        self.nints = 2 ** nbits
        self.lower = int(self.TH_BV * self.nints)
        self.upper = self.nints - self.lower
        self.TH_BP *= self.size
        
        # frequency params
        afreqs = np.abs(self.SamplingRate * np.fft.fftfreq(self.PAD))
        self.hfband = (self.HF[0] <= afreqs) * (afreqs <= self.HF[1])
        norm = float(self.PAD * self.size * ((self.nints - 1) ** 2))
        self.TH_TL *= norm
        self.TH_TH *= norm
        
        # reset buffer
        self.reset()
    
    def reset(self):
        # reset buffer
        
        self.handsOn = False
        self.handValue = misc.BDIterator(self.delay)
        self.buffer = np.array([], dtype='int')
    
    def check(self, signal):
        # check if hands are on with given signal fragment
        
        # previous value
        prev = self.handsOn
        
        # make sure we have ints
        signal = signal.astype('int')
        
        # add to buffer
        self.buffer = np.hstack((self.buffer, signal[-self.step:]))
        
        if len(self.buffer) >= self.size:
            aux = self.buffer[:]
            
            counts = np.bincount(aux, minlength=self.nints)
            bc = np.sum(counts[:self.lower])
            tc = np.sum(counts[self.upper:self.nints])
            
            if bc > self.TH_BP:
                # print "hands: bottom limit"
                res = False
            elif tc > self.TH_BP:
                # print "hands: top limit"
                res = False
            else:
                bux = aux - aux.mean()
                ft = np.abs(np.fft.fft(bux, n=self.PAD)) ** 2
                sft = np.sum(ft)
                
                if sft < self.TH_TL:
                    res = False
                elif sft > self.TH_TH:
                    res = False
                else:
                    hfpow = np.sum(ft[self.hfband]) / sft
                    if hfpow > self.TH_HF:
                        res = False
                    else:
                        res = True
            
            if res:
                val = self.handValue.next()
            else:
                val = self.handValue.prev()
            
            if val == 1:
                self.handsOn = True
            elif val == 0:
                self.handsOn = False
            
            # roll
            self.buffer = self.buffer[-self.step:]
        
        return self.handsOn != prev


class CheckHandsNotch(object):
    """
    Class to check if hands are on the ECG electrodes.
    """
    
    HF = [45, 65]
    PAD = 512
    
    TH_BV = 0.0234375
    TH_BP = 0.15
    # TH_BP = 0.3
    TH_TL = 0.00015529624809234089
    TH_TH = 0.015529624809234088
    # TH_TH = 0.007764812404617044
    # TH_TL = 0.00002
    # TH_TH = 0.02
    # TH_HF = 0.6
    
    def __init__(self, step=150, size=300, SamplingRate=1000., nbits=10, delay=3):
        # init
        
        # self inputs
        self.step = step
        self.size = size
        self.SamplingRate = SamplingRate
        if delay > 0:
            self.delay = [0] + delay * [-1] + [1]
        else:
            self.delay = [0, 1]
        
        # histogram params
        self.nints = 2 ** nbits
        self.lower = int(self.TH_BV * self.nints)
        self.upper = self.nints - self.lower
        self.TH_BP *= self.size
        
        # frequency params
        afreqs = np.abs(self.SamplingRate * np.fft.fftfreq(self.PAD))
        self.hfband = (self.HF[0] <= afreqs) * (afreqs <= self.HF[1])
        norm = float(self.PAD * self.size * ((self.nints - 1) ** 2))
        self.TH_TL *= norm
        self.TH_TH *= norm
        
        # notch filter at 50 Hz
        wn = 50. / self.SamplingRate
        r = 0.99
        b = np.array([1.0, -2.0 * np.cos(2*np.pi * wn), 1.0])
        a = np.array([1.0, -2.0*r * np.cos(2*np.pi * wn), r*r])
        self.notch = OnlineFilterBA(b, a, truncate=False)
        
        # reset buffer
        self.reset()
    
    def reset(self):
        # reset buffer
        
        self.handsOn = False
        self.handValue = misc.BDIterator(self.delay)
        self.buffer = np.array([], dtype='int')
        self.filter_buffer = np.array([], dtype='float')
        self.notch.reset()
    
    def test(self, signal):
        
        # filter
        out = self.notch.filter(signal)
        self.filter_buffer = np.hstack((self.filter_buffer, out[-self.step:]))
        
        # make sure we have ints
        signal = signal.astype('int')
        
        # add to buffer
        self.buffer = np.hstack((self.buffer, signal[-self.step:]))
        
        if len(self.buffer) >= self.size:
            aux = self.buffer[:]
            
            counts = np.bincount(aux, minlength=self.nints)
            bc = np.sum(counts[:self.lower])
            tc = np.sum(counts[self.upper:self.nints])
            
            filtered = self.filter_buffer[:]
            bux = filtered - filtered.mean()
            ft = np.abs(np.fft.fft(bux, n=self.PAD)) ** 2
            sft = np.sum(ft)
            
            # roll
            self.buffer = self.buffer[-self.step:]
            self.filter_buffer = self.filter_buffer[-self.step:]
            
            return [bc, tc, sft]
    
    def check(self, signal):
        # check if hands are on with given signal fragment
        
        # previous value
        prev = self.handsOn
        
        # filter
        out = self.notch.filter(signal)
        self.filter_buffer = np.hstack((self.filter_buffer, out[-self.step:]))
        
        # make sure we have ints
        signal = signal.astype('int')
        
        # add to buffer
        self.buffer = np.hstack((self.buffer, signal[-self.step:]))
        
        if len(self.buffer) >= self.size:
            aux = self.buffer[:]
            
            counts = np.bincount(aux, minlength=self.nints)
            bc = np.sum(counts[:self.lower])
            tc = np.sum(counts[self.upper:self.nints])
            
            if bc > self.TH_BP:
                # print "hands: bottom limit"
                self.handValue.index = 0
                self.handsOn = False
            elif tc > self.TH_BP:
                # print "hands: top limit"
                self.handValue.index = 0
                self.handsOn = False
            else:
                filtered = self.filter_buffer[:]
                bux = filtered - filtered.mean()
                ft = np.abs(np.fft.fft(bux, n=self.PAD)) ** 2
                sft = np.sum(ft)
                
                if sft < self.TH_TL:
                    val = self.handValue.prev()
                elif sft > self.TH_TH:
                    val = self.handValue.prev()
                else:
                    val = self.handValue.next()
                
                if val == 1:
                    self.handsOn = True
                elif val == 0:
                    self.handsOn = False
            
            # roll
            self.buffer = self.buffer[-self.step:]
            self.filter_buffer = self.filter_buffer[-self.step:]
        
        return self.handsOn != prev


class CheckHands3(object):
    """
    Class to check if hands are on the ECG electrodes.
    """
    
    HF = [1.953125, 3.90625]
    PAD = 512
    
    TH_BV = 0.0234375
    THRL = 1.
    THRH = 100.
    
    def __init__(self, step=150, size=500, SamplingRate=1000., nbits=10, delay=3, bsize=5):
        # init
        
        # self inputs
        self.step = step
        self.size = size
        self.SamplingRate = SamplingRate
        if delay >= 0:
            self.delay = [0] + delay * [-1] + [1]
        else:
            self.delay = [0, 1]
        
        # histogram params
        self.nints = 2 ** nbits
        self.lower = int(self.TH_BV * self.nints)
        self.upper = self.nints - self.lower
        
        # frequency params
        self.bsize = bsize
        afreqs = np.abs(self.SamplingRate * np.fft.fftfreq(self.PAD))
        self.hfband = (self.HF[0] <= afreqs) * (afreqs <= self.HF[1])
        
        # reset buffer
        self.reset()
    
    def reset(self):
        # reset buffer
        
        self.handsOn = False
        self.handValue = misc.BDIterator(self.delay)
        self.buffer = np.array([], dtype='int')
        self.powBuffer = collections.deque(self.bsize * [0.], maxlen=self.bsize)
    
    def check(self, signal):
        # check if hands are on with given signal fragment
        
        # previous value
        prev = self.handsOn
        
        # make sure we have ints
        signal = signal.astype('int')
        
        # add to buffer
        self.buffer = np.hstack((self.buffer, signal[-self.step:]))
        if len(self.buffer) >= self.size:
            aux = self.buffer[:]
            
            # power computation
            bux = aux - aux.mean()
            power = np.fft.fft(bux, n=self.PAD) / self.PAD
            power = np.sum(np.abs(power[self.hfband]))
            self.powBuffer.append(power)
            
            if np.any(aux < self.lower):
                res = False
            elif np.any(aux > self.upper):
                res = False
            else:
                val = np.mean(self.powBuffer)
                if val < self.THRL:
                    res = False
                elif val > self.THRH:
                    res = False
                else:
                    res = True
            
            if res:
                val = self.handValue.next()
            else:
                val = self.handValue.prev()
            
            if val == 1:
                self.handsOn = True
            elif val == 0:
                self.handsOn = False
            
            # roll
            self.buffer = self.buffer[-self.step:]
        
        return self.handsOn != prev


class OnlineFilter(object):
    """
    Online Filtering class.
    """
    
    def __init__(self, truncate=False, *args, **kwargs):
        
        # get filter coefficients
        b, a = filt._getFilter(*args, **kwargs)
        
        # self things
        self.b = b
        self.a = a
        self.truncate = truncate
        
        # reset
        self.reset()
    
    def reset(self):
        # reset the filter state
        
        self.zi = filt._getInititalState(self.b, self.a)
        
        if self.truncate:
            self.size = len(self.zi)
        else:
            self.size = 0
    
    def filter(self, signal):
        # filter the signal fragment
        
        out, zf = filt._filterSignal(self.b, self.a, signal, self.zi, checkPhase=False)
        
        self.zi = zf
        
        if self.size > 0:
            # truncate filter delay
            out = out[self.size:]
            self.size -= len(signal)
        
        return out


class OnlineFilterBA(OnlineFilter):
    """OnlineFilter with explicitly given coefficients."""
    
    def __init__(self, b, a, truncate=False):
        # self things
        self.b = b
        self.a = a
        self.truncate = truncate
        
        # reset
        self.reset()


class SegmentECG(object):
    """
    Online ECG segmentation.
    """
    
    def __init__(self, size=1500, SamplingRate=1000.):
        
        # self inputs
        self.size = size
        self.SamplingRate = SamplingRate
        
        # reset
        self.reset()
    
    def reset(self):
        # reset HR, continuity parameters, buffer
        
        self.HR = 0
        self.params = {}
        self.resetBuffer()
    
    def resetBuffer(self):
        # signal bugger
        
        self.buffer = np.array([], dtype='float')
    
    @staticmethod
    def concatenateSegments(*args):
        # helper method to concatenate array of segments
        
        # filter out length 0
        aux = filter(lambda item: len(item) > 0, args)
        
        # concatenate
        try:
            out = np.vstack(aux)
        except ValueError:
            out = np.array([])
        
        return out
    
    def segment(self, signal):
        # segment the ECG signal fragment
        
        # add to buffer
        self.buffer = np.hstack((self.buffer, signal))
        
        if len(self.buffer) >= self.size:
            aux = self.buffer[:]
            
            # segment
            res = models.engzee_incomplete(Signal=aux, SamplingRate=self.SamplingRate, Params=self.params)
            res['Signal'] = aux
            
            # update HR
            self.HR = res['Params']['HR']
            
            # update continuity parameters
            self.params = res.pop('Params')
            
            # reset buffer
            self.resetBuffer()
        else:
            res = {}
        
        return res


class SSFSegmenter(object):
    """
    Online ECG segmentation using the SSF algorithm.
    """
    
    def __init__(self, SamplingRate=1000., size=1.5, threshold=20, before=0.03, after=0.01):
        # self inputs
        self.SamplingRate = SamplingRate
        self.size = int(SamplingRate * size)
        self.before = int(SamplingRate * before)
        self.after = int(SamplingRate * after)
        self.threshold = threshold
        
        # reset
        self.reset()
    
    def reset(self):
        # reset HR, continuity parameters, buffer
        
        self.HR = 0
        self.params = {}
        self.resetBuffer()
    
    def resetBuffer(self):
        # signal bugger
        
        self.buffer = np.array([], dtype='float')
    
    @staticmethod
    def concatenateSegments(*args):
        # helper method to concatenate array of segments
        
        # filter out length 0
        aux = filter(lambda item: len(item) > 0, args)
        
        # concatenate
        try:
            out = np.vstack(aux)
        except ValueError:
            out = np.array([])
        
        return out
    
    def segment(self, signal):
        # segment the ECG signal fragment
        
        # add to buffer
        self.buffer = np.hstack((self.buffer, signal))
        
        if len(self.buffer) >= self.size:
            aux = self.buffer[:]
            
            # segment
            res = models.engzee_incomplete(Signal=aux, SamplingRate=self.SamplingRate, Params=self.params)
            res['Signal'] = aux
            
            # update HR
            self.HR = res['Params']['HR']
            
            # update continuity parameters
            self.params = res.pop('Params')
            
            # reset buffer
            self.resetBuffer()
        else:
            res = {}
        
        return res


class WSPlotter(object):
    """
    Prepare the signal plot for communication to the webpage via websockets.
    """
    
    def __init__(self, screenSize=1280, decimation=1, scale=None):
        
        # check inputs
        if scale is None:
            scale = (600., 0.35)
        
        # self inputs
        self.screenSize = screenSize
        self.decimation = decimation
        self.scale = scale
        
        # string template
        self.tmpl = "[" + ",".join(["[%d,%s]" % (i, '%s') for i in range(screenSize)]) + "]"
        
        # reset buffer
        self.reset()
    
    def reset(self):
        # reset buffer
        
        self.signalBuffer = np.zeros(self.screenSize)
    
    def prepare(self, signal):
        # send signal
        
        # decimate and scale signal
        ysc = signal[::self.decimation] / self.scale[0] + self.scale[1]
        
        # prepare string
        ysc =  ysc[-self.screenSize:]
        nb = len(ysc)
        
        self.signalBuffer = np.concatenate((self.signalBuffer[nb:], ysc))
        res = self.tmpl % tuple(self.signalBuffer)
        
        return res

