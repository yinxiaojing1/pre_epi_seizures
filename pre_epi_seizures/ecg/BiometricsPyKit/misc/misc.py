"""
.. module:: misc
   :platform: Unix, Windows
   :synopsis: This module provides miscellaneous tools.

.. moduleauthor:: Filipe Canento


"""

# Imports
# built/in
import cPickle
import ctypes
import gzip
import itertools
import logging
import time

# 3rd party
from scipy import linalg
import scipy.spatial.distance
import numpy
import pylab

# BiometricsPyKit



def getLogger(name, logPath=None, level='debug'):
    # create a logger
    logger = logging.getLogger(name)
    
    if len(logger.handlers) == 0:
        # logger was not yet configured
        # create handler
        if logPath is None:
            handler = logging.StreamHandler()
        else:
            handler = logging.FileHandler(filename=logPath)
        
        # create formatter
        formatter = logging.Formatter(fmt='%(levelname)s - %(asctime)s - %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
        
        # set level
        if level == 'debug':
            level = logging.DEBUG
        elif level == 'info':
            level = logging.INFO
        elif level == 'warning':
            level = logging.WARNING
        elif level == 'error':
            level = logging.ERROR
        elif level == 'critical':
            level = logging.CRITICAL
        else:
            raise ValueError, "Unknown logging level (%s)" % str(level)
        
        logger.setLevel(level)
        handler.setLevel(level)
        
        # add formatter to handler
        handler.setFormatter(formatter)
        
        # add handler to logger
        logger.addHandler(handler)
    
    return logger


def slasherDict(d, path):
    # get items from a nested dictionary with a unix-like path
    
    bits = path.split('/')
    if path[0] == '/':
        bits = bits[1:]
    
    aux = d
    for item in bits:
        try:
            aux = aux[item]
        except KeyError:
            aux = None
            break
    
    return aux


class FiniteCycle(object):
    """
    Cycles the given finite iterable indefinitely. 
    Subclasses ``itertools.cycle`` and adds pickle support.
    """
    def __init__(self, finite_iterable):
        self._index = 0
        self._iterable = tuple(finite_iterable)
        self._iterable_len = len(self._iterable)
        self._iter = itertools.cycle(finite_iterable)
        # super(FiniteCycle, self).__init__(self._iterable)

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index):
        """
        Sets the current index into the iterable. 
        Keeps the underlying cycle in sync.

        Negative indexing supported (will be converted to a positive index).
        """
        index = int(index)
        if index < 0:
            index = self._iterable_len + index
            if index < 0:
                raise ValueError('Negative index is larger than the iterable length.')

        if index > self._iterable_len - 1:
            raise IndexError('Index is too high for the iterable. Tried %s, iterable '
                             'length %s.' % (index, self._iterable_len))

        # calculate the positive number of times the iterable will need to be moved
        # forward to get to the desired index
        delta = (index + self._iterable_len - self.index) % (self._iterable_len)

        # move the finite cycle on ``delta`` times.
        for _ in xrange(delta):
            self.next()

    def next(self):
        self._index += 1
        if self._index >= self._iterable_len:
            self._index = 0
        return self._iter.next()

    def peek(self):
        """
        Return the next value in the cycle without moving the iterable forward.
        """
        return self._iterable[self.index]

    def __reduce__(self):
        return (FiniteCycle, (self._iterable, ), {'index': self.index})

    def __setstate__(self, state):
        self.index = state.pop('index')


class BDIterator(object):
    def __init__(self, collection, prev=0):
        self.collection = collection
        self.index = prev

    def next(self):
        i = self.index + 1
        
        try:
            result = self.collection[i]
        except IndexError:
            result = self.collection[self.index]
        else:
            self.index = i
        
        return result

    def prev(self):
        i = self.index - 1
        if i < 0:
            i = 0
        self.index = i
        
        return self.collection[self.index]
    
    def item(self):
        return self.collection[self.index]

    def __iter__(self):
        return iter(self.collection)


class InfiniteBiterator(object):
    """
    This infinite iterator swings both ways.
    """
    def __init__(self, collection):
        """
        The iterator infinitely loops over the given collection, either forwards or backwards.
        """
        
        self._collection = collection
        self._length = len(collection)
        self._index = self._length - 1
    
    @property
    def index(self):
        """
        Returns current index.
        """
        return self._index

    @index.setter
    def index(self, index):
        """
        Sets the current index.
        """
        
        index = int(index)
        
        if index < 0:
            index = abs(index)
            if index > self._length:
                self._index = 0
            else:
                self._index = self._length - index
        else:
            if index >= self._length:
                self._index = self._length - 1
            else:
                self._index = index
    
    def next(self):
        """
        Get the next element from the collection.
        """
        
        self._index += 1
        
        if self._index >= self._length:
            self._index = 0
        
        return self._collection[self._index]

    def prev(self):
        """
        Get the previous element from the collection. 
        """
        self._index -= 1
        
        if self._index < 0:
            self._index = self._length - 1
        
        return self._collection[self._index]
    
    def cur(self):
        """
        Return the current element.
        """
        
        return self._collection[self._index]


# from http://msdn.microsoft.com/en-us/library/windows/desktop/dd375731(v=vs.85).aspx
WIN_KEYS = {'BACKSPACE': 0x08, 'TAB': 0x09, 'ENTER': 0x0D, 'SHIFT': 0x10,
            'CTRL': 0x11, 'ALT': 0x12, 'ESC': 0x1B, 'SPACE': 0x20,
            'LEFT': 0x25, 'UP': 0x26, 'RIGHT': 0x27, 'DOWN': 0x28,
            'DEL': 0x2E, '0': 0x30, '1': 0x31, '2': 0x32, '3': 0x33, '4': 0x34,
            '5': 0x35, '6': 0x36, '7': 0x37, '8': 0x38, '9': 0x39, 'A': 0x41,
            'B': 0x42, 'C': 0x43, 'D': 0x44, 'E': 0x45, 'F': 0x46, 'G': 0x47,
            'H': 0x48, 'I': 0x49, 'J': 0x4A, 'K': 0x4B, 'L': 0x4C, 'M': 0x4D,
            'N': 0x4E, 'O': 0x4F, 'P': 0x50, 'Q': 0x51, 'R': 0x52, 'S': 0x53,
            'T': 0x54, 'U': 0x55, 'V': 0x56, 'W': 0x57, 'X': 0x58, 'Y': 0x59,
            'Z': 0x5A, 'F1': 0x70, 'F2': 0x71, 'F3': 0x72, 'F4': 0x73,
            'F5': 0x74, 'F6': 0x75, 'F7': 0x76, 'F8': 0x77,  'F9': 0x78,
            'F10': 0x79, 'F11': 0x7A, 'F12': 0x7B, 'F13': 0x7C, 'F14': 0x7D,
            'F15': 0x7E, 'F16': 0x7F, 'F17': 0x80, 'F18': 0x81, 'F19': 0x82,
            'F20': 0x83, 'F21': 0x84, 'F22': 0x85, 'F23': 0x86, 'F24': 0x87,
            'NUM_0': 0x60, 'NUM_1': 0x61, 'NUM_2': 0x62, 'NUM_3': 0x63,
            'NUM_4': 0x64, 'NUM_5': 0x65, 'NUM_6': 0x66, 'NUM_7': 0x67,
            'NUM_8': 0x68, 'NUM_9': 0x69, '+': 0xBB, ',': 0xBC, '-': 0xBD,
            '.': 0xBE, '*': 0x6A, '/': 0x6F,
            }


PUL = ctypes.POINTER(ctypes.c_ulong)
try:
    SendInput = ctypes.windll.user32.SendInput
except AttributeError:
    SendInput = None

class KeyBdInput(ctypes.Structure):
    # Windows Keyboard input emulator class
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class HardwareInput(ctypes.Structure):
    # Windows hardware input emulator class
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]


class MouseInput(ctypes.Structure):
    # Windows mouse input emulator class
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class Input_I(ctypes.Union):
    # Windows input union emulator class
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]


class Input(ctypes.Structure):
    # Windows input emulator class
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]


def pressKey(key):
    # press a key
    
    # translate key
    hexKeyCode = WIN_KEYS[key]
    
    # send input
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(hexKeyCode, 0x48, 0, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def releaseKey(key):
    # release a key
    
    # translate key
    hexKeyCode = WIN_KEYS[key]
    
    # send input
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(hexKeyCode, 0x48, 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def checkMouseMov(d):
    # check the mouse distance
    
    if d > 65535:
        d = 65535
    
    if d < 0:
        d = 0
    
    return d


def moveMouse(dx, dy, absolute=True):
    # move the mouse
    
    if absolute:
        dwf = 0x8000
        # check displacement values
        dx = checkMouseMov(dx)
        dy = checkMouseMov(dy)
    else:
        dwf = 0
    
    # movement ocurred flag
    dwf += 0x0001
    
    # convert to ctype
    dx = ctypes.c_long(dx)
    dy = ctypes.c_long(dy)
    
    # send input
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(dx, dy, 0, dwf, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def clickMouse(key):
    # mouse click
    
    if key == 'LEFT':
        dwf = 0x0002
    elif key == 'RIGHT':
        dwf = 0x0008
    elif key == 'MIDDLE':
        dwf = 0x0020
    else:
        raise ValueError, "Unknown mouse key (%s)." % str(key)
    
    # send input
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(0, 0, 0, dwf, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def unclickMouse(key):
    # mouse unclick
    
    if key == 'LEFT':
        dwf = 0x0004
    elif key == 'RIGHT':
        dwf = 0x0010
    elif key == 'MIDDLE':
        dwf = 0x0040
    else:
        raise ValueError, "Unknown mouse key (%s)." % str(key)
    
    # send input
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(0, 0, 0, dwf, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def keyboardPress(key, duration=0):
    # press individual keyboard keys or combinations of keys
    # windows only (for now)
    
    # single key, or combination?
    if isinstance(key, basestring):
        key = [key]
    
    assert isinstance(key, list)
    
    # press
    for item in key:
        pressKey(item)
    
    # wait
    time.sleep(duration)
    
    # release
    for item in key[::-1]:
        releaseKey(item)


def get_subgroup_by_tags(st, tags):
    info = {
        'train_set': tags,
        'test_set': tags,
        'train_time': (0, 0),
        'test_time': (0, 0),} 
    # get subject info
    subjects = st.subjectTTSets(info) # name, train record ids, test record ids
    out = {}
    for s in subjects:
        for rid in s[1]:
            out[rid] = s[0]
    return out

def load_information(filename):
    fd = gzip.open(filename, 'rb')
    data = cPickle.load(fd)
    fd.close()
    return data

def save_information(info, dformat, filename=None):
    # save information from dict info in specified format (gzip, hdf5, ...)
    if dformat == 'dict':
        if filename == None: raise TypeError, "Specify output filename"
        fd = gzip.open(filename, 'wb')
        cPickle.dump(info, fd)
        fd.close()
    elif dformat == 'hdf5':
        print "not yet..."
    else:
        raise TypeError, "Save format %s not implemented."%dformat

def merge_clusters(data):
    """
    Merge information from different clusters. Ignores cluster -1 (noise). 

    Input:
    
        data (dict): input data. 
                     example: data = {1: {'segments': record 1 ECG segments, 'R': record 1 r peaks}, 
                                     2: {'segments': record 2 ECG segments, 'R': record 2 r peaks},
                                     ...}
        
        data_type (string): data type to be analyzed
        
        parameters (dict): filter parameters.
                           Example: parameters = {'method': 'dbscan', ...}
                                                  
    Output:
        
        output (dict): output data where keys correspond to record id numbers.
                       example: output = { 1: {-1: record 1 outlier indexes, '0': record 1 cluster 0 indexes},
                                           2: {-1: record 2 outlier indexes, '0': record 2 cluster 0 indexes},
                                           ...}

    Configurable fields:{"name": "??.??", "config": {}, "inputs": ["data", "data_type", "parameters"], "outputs": ["output"]}

    See Also:

    Notes:

    Example:
       
    """      
    # receives dict {'-1': outlier indexes list, '0': cluster 0 indexes list, '1': cluster 1 indexes list, ..., 'n': cluster n indexes list}
    # returns array with indexes from all clusters except -1
    res = []
    for cluster in data:
        if cluster == '-1': continue
        res = scipy.hstack((res, data[cluster]))
    return res

def quantize(signal, levels=256):
    
    signal_min = numpy.min(signal,1) #scipy.array(map(lambda i: min(i), signal))
    signal_max = numpy.max(signal,1) #scipy.array(map(lambda i: max(i), signal))
    signal_range = signal_max - signal_min 
    signal_invrange = 1./signal_range
    
    signal_min_matrix = []
    for i in xrange(len(signal)):
        signal_min_matrix.append(signal_min[i]*scipy.ones(len(signal[i])))
    signal_min_matrix = scipy.array(signal_min_matrix)
    
    signal_invrange_matrix = scipy.diag(signal_invrange)
    
    signal_q = scipy.matrix(signal_invrange_matrix)*scipy.matrix((signal - signal_min_matrix))*(levels-1)
    signal_q = scipy.around(scipy.array(signal_q))
    
    return signal_q

def chunks(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

def cosv(a,b):
    return (scipy.dot(a,b)/(linalg.norm(a)*linalg.norm(b)))

def msedistance(s1, s2):
    return scipy.spatial.distance.euclidean(s1, s2)

def cosdistance(s1,s2):
    return 1-abs(cosv(s1, s2))

def dtwdistance(s1, s2):
    return

def wavedistance(testwave, trainwaves, fdistance):
    return scipy.array([fdistance(wave, testwave) for wave in trainwaves])

def mean_waves(segments, nwaves, jump=0):
    # no superposition, remainder is discarded
    jump = nwaves if jump == 0 else jump
    res = []
    for mean_wave in map(lambda i: 1.*scipy.sum(map(lambda j: 1.*segments[j], scipy.arange(i,i+nwaves)), axis=0)/nwaves, scipy.arange(0,len(segments)-(nwaves-1), jump)):
        res.append(mean_wave)
        
    return scipy.array(res)

def median_waves(data, nwaves, jump=0):
    jump = nwaves if jump == 0 else jump
    
    out = []
    for i in xrange(0, len(data), jump):
        out.append(scipy.median(data[i:i+jump], 0))
        
    return scipy.array(out)       

def random_idx(low, high, size, exclude=[]):
    res = []
    while len(res) < size:
        r = scipy.random.randint(low,high)
        if r not in res and r not in exclude: res.append(r)
    return res

def plot_data(data, ax, title):
    sd_th = 1.0
    ax.cla()
    l, c = 5., [.75,.75,.75]
    map(lambda i: ax.plot(data[i,:], color=c, alpha=.5, linewidth=l), scipy.arange(0, scipy.shape(data)[0], 1))     
    # Mean Curve
    mean_curve = scipy.mean(data,0)
    sdplus, sdminus = mean_curve+sd_th*scipy.std(data,0), mean_curve-sd_th*scipy.std(data,0)
    l, c = 2., [.35,.35,.35]
    ax.plot(sdminus,'--',color=c, linewidth=l)
    ax.plot(sdplus,'--',color=c, linewidth=l)
    ax.plot(mean_curve,'k', linewidth=5.)
    ax.grid() 
    ax.autoscale(enable=True, axis='both', tight=True)  
    ax.set_title(title)
    
    return mean_curve, sdplus, sdminus

def plot_mean_curve(data):
    
    pylab.figure()
    
    l=5.
    c=[.75,.75,.75]
    map(lambda i: pylab.plot(data[i,:],color=c,alpha=.05,linewidth=l), scipy.arange(0,scipy.shape(data)[0],1))
    
    mw = scipy.mean(data,0)
    sd = scipy.std(data,0)

    l=2.
    c=[.35,.35,.35]
    pylab.plot(mw-sd,'--',color=c, linewidth=l)
    pylab.plot(mw+sd,'--',color=c, linewidth=l)
    pylab.plot(mw,'k', linewidth=5.)
    pylab.axis('tight')
    pylab.grid()
    
    return mw, sd

def ar(x, M):
    rxx = []
    lx = len(x)
    for m in xrange(0,M):
        rxx.append(0)
        for i in xrange(0, lx-m-1):
            rxx[-1] += x[i]*x[i+m]
    return rxx/rxx[0]

def centeredFFT(x, fs, oneside=False, ax=None):
    X = scipy.fft(x)/len(x)
    X = numpy.fft.helper.fftshift(X)
    X = abs(X)
    
    if(oneside):
        X=X[len(X)*0.5:]*2
        freq=scipy.linspace(0,0.5*fs,len(X))
    else:
        freq=scipy.linspace(-0.5*fs,0.5*fs,len(X))
        
    if ax is not None:    
        ax.plot(freq,X)
        ax.axis([0,max(freq),0,max(X)*1.1])
        ax.grid('on')
        
    return freq,X

def test_real_time():
    import scipy.signal as ss
    from database import mongoH5
    # Get Data
    config = {'dbName': 'CruzVermelhaPortuguesa', 'host': '193.136.222.234', 'port': 27017, 'path': r'\\193.136.222.220\cybh\data\CVP\hdf5'}
    db = mongoH5.bioDB(**config)
    recs = mongoH5.records(**db)
    ids_recs = recs.getAll()['idList']
    raw = recs.getData(ids_recs[1], 'ECG/hand/raw', 0)['signal']
    # Parameters
    step = 150
    overlap = 300
    SamplingRate = 1000.
    y_b = scipy.zeros(overlap)
    y_f = scipy.zeros(overlap)
    # filter_t = 'firwin'#'butter'
    # FIR filter
    bfir1 = ss.firwin(overlap+1,[2*1./SamplingRate, 2*40./SamplingRate], pass_zero=False)
    # Butterworth
    [b, a] = ss.butter(4, [2.*1./SamplingRate, 2.*40./SamplingRate], 'band')
    zf= ss.lfilter_zi(b, a)
    # Plot
    fig = pylab.figure()
    for i in xrange(100):
        frameArray = scipy.copy(raw[i*step:(i+1)*step])
        frameArray -= scipy.mean(frameArray)
        
        # Butterworth filtering with initial conditions
        ffr, zf = ss.lfilter(b=b, a=a, x=frameArray, zi=zf)
        y_b = scipy.concatenate((y_b,ffr))
        # Overlap & add fir filtering      
        ffr = scipy.convolve(frameArray, bfir1)
        y_f[-overlap:] += ffr[:overlap]
        y_f = scipy.concatenate((y_f,ffr[overlap:]))
            
        ax = fig.add_subplot(311)
        ax.cla()
        pylab.plot(raw[:(i+1)*step], 'g', label='raw')
        ax.set_title('raw')
        ax = fig.add_subplot(312)
        ax.cla()
        pylab.plot(y_b, 'b', label='butter')
        ax.set_title('butter')
        ax = fig.add_subplot(313)
        ax.cla()
        pylab.plot(y_b, 'b', label='firwin')
        ax.set_title('firwin')
        pylab.show()
        
    return