"""
.. module:: featureextraction
   :platform: Unix, Windows
   :synopsis: This module provides various functions to...

.. moduleauthor:: Filipe Canento

"""

import scipy
import pylab
import numpy
import traceback
import math
import h5py



def selector(method):
    """
    Selector for the feature extraction functions and methods.
    
    Input:
        method (str): The desired function or method.
    
    Output:
        fcn (function): The function pointer.
    
    Configurable fields:
    
    See Also:
    
    
    Example:
    
    
    References:
        .. [1]
    """
    
    if method == 'odinaka':
        fcn = odinakaSTFT
    elif method == 'STFT':
        fcn = shortTimeFT
    else:
        raise TypeError, "Method %s not implemented." % method
    
    return fcn


def subpattern(patterns, cols, lines='all'):
    # patterns is list of patterns
    if lines == 'all':
        out = scipy.array(patterns)[:, cols]
    else:
        out = scipy.array(patterns)[lines, cols]
    return out


def getWindow(name, n):
    # get an n sized window function
    
    if name == 'rectangular':
        return numpy.ones(n, dtype='float')
    elif name == 'bartlett':
        return numpy.bartlett(n)
    elif name == 'blackman':
        return numpy.blackman(n)
    elif name == 'hamming':
        return numpy.hamming(n)
    elif name == 'hanning':
        return numpy.hanning(n)
    else:
        raise TypeError, "Unknown window function (%s)." % str(name)


def shortTimeFT(signal, n, overlap, nfft, window, scale=False):
    # compute the Short Time Fourier Transform
    
    # check if window is string
    if isinstance(window, basestring):
        window = getWindow(window, n)
    
    X = numpy.array([numpy.fft.fft(window * signal[i: i + n], n=nfft)
                     for i in xrange(0, len(signal) - n + 1, n - overlap)])
    
    if scale:
        # to compensate for windowing loss
        X /= numpy.sum(numpy.abs(window)**2)
    
    return X


def odinakaSTFT(data, n, overlap, nfft, window, mask=None, scale=False):
    # compute STFT for each template in data and extract spectral power according to mask
    
    # check data representation
    try:
        data = numpy.array(data['templates'])
    except ValueError:
        # data is already numpy?
        if not isinstance(data, numpy.ndarray):
            raise
    
    # check mask
    if mask is None:
        mask = range(int(nfft / 2))
    elif isinstance(mask, list) or isinstance(mask, tuple):
        mask = range(mask[0], mask[1])
    
    X = []
    for tpl in data:
        # compute the short time FT
        aux = shortTimeFT(tpl, n, overlap, nfft, window, scale)
        # transform
        aux = numpy.log(numpy.abs(aux[:, mask])**2)
        X.append(aux)
    
    X = numpy.array(X)
    
    return {'templates': X}


def all_indices(value, qlist):
    indices = []
    idx = -1
    for a in qlist:
        idx = idx + 1
        if a == value:
            indices.append(idx)
    return indices


def stft(x, fs, framesz, hop, nfft, ftype): #0-stft0, 1-stft1
    if not ftype:
        X = stft0(x, fs, framesz, hop)
    else:
        X = stft1(x, fs, framesz, hop, nfft)
    return X
        

def stft0(x, fs, framesz, hop):
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    w = scipy.hamming(framesamp)
    #print len(w), len(x[0:0+framesamp]), framesamp
    #tmp = w*x[0:0+framesamp]
    X = scipy.array([scipy.fft(w*x[i:i+framesamp]) 
                     for i in range(0, len(x)-framesamp, hopsamp)])
    return X
    
def stft1(x, fs, framesz, hop, nfft):
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    w = scipy.hamming(framesamp)
    X = scipy.array([(scipy.fft(w*x[i:i+framesamp],nfft))
                     for i in range(0, len(x)-framesamp, hopsamp)])
    return X

class odinaka(object):
    
    def __init__(self, fs , framesz , hop , sinal_length):
        
        # sampled at 1 kHz
        # with a frame size of 64 milliseconds
        # and hop size of 10 milliseconds.
        # ECG heart beat time 600ms
    
        self.is_trained = False
        self.global_train = []
        self.users_train = []
        self.users_distance = []
        
        self.fs = fs           # sampled at 1 kHz
        self.framesz = framesz     # with a frame size of 64 milliseconds
        self.hop = hop         # and hop size of 10 milliseconds.
        self.sinal_length = sinal_length  # ECG heart beat time 600ms     
        self.classify_k = 0   
        
        self.n_stft = None
        self.stft_size = None
        
    def train(self, data=None, data_idx=None, label=None):
        
        # Check inputs
        #if data is None:
        #    raise TypeError, "Please provide input data."
        #if label is None:
        #    raise TypeError, "Please provide input data label."    
        #if 0 not in label:
        #    raise TypeError, "Label must start in 0 (e.g., label = [0,0,0,1,2] for 5 samples)."
        success = False
        try:
            
            self.n_stft = int((int(self.sinal_length*self.fs) - int(self.framesz*self.fs)) / int(self.hop*self.fs))+1
            self.stft_size = int(self.framesz*self.fs)/2
            
            user = 1
            
            while all_indices(user, data_idx):
                
                
            
                media = scipy.zeros([self.n_stft,self.stft_size])
                m_total = scipy.zeros(self.n_stft*self.stft_size)
                variancia = scipy.zeros([self.n_stft,self.stft_size])
                v_total = scipy.zeros(self.n_stft*self.stft_size)
                
                aux_t = 0
                #print user, data_idx, all_indices(user, data_idx), data[0]
                for heartbeat in data[all_indices(user, data_idx)]:
                    aux_t = aux_t + 1
                    X = stft0(heartbeat, self.fs, self.framesz, self.hop)
                    i=0                
                    for laps in X:
                        aux = scipy.log(pow(abs(laps[0:self.stft_size]),2))
                        media[i] = media[i] + aux
                        i = i + 1
                media = media / aux_t
                
                for heartbeat in data[all_indices(user, data_idx)]:
                    
                    X = stft0(heartbeat, self.fs, self.framesz, self.hop)
                    i=0                
                    for laps in X:
                        aux = scipy.log(pow(abs(laps[0:self.stft_size]),2))
                        variancia[i] = variancia[i] + ((aux - media[i])**2)
                        i = i + 1
                    
                variancia = variancia / aux_t
                
                media = scipy.reshape(media, self.n_stft*self.stft_size)
                variancia = scipy.reshape(variancia, self.n_stft*self.stft_size)
                
                self.users_train.append([user, media,variancia])
                
                m_total = (m_total + media)/2
                v_total = (v_total + variancia)/2
                
                user = user + 1
   
            self.global_train.append(m_total)
            self.global_train.append(v_total)
            
            for users in self.users_train:
                d = ((users[2] + ((users[1]-m_total)**2))/(2*v_total)) + ((v_total + ((users[1]-m_total)**2))/(2*users[2])) - 1
                self.users_distance.append([users[0], d])
            
               
            success = True
            self.is_trained = True
        except Exception as e:
            print e
            print traceback.format_exc()
        return success
    
    def classify(self, data=None):
        #    self.global_train[0] - total media
        #    self.global_train[1] - total variancia
        #
        #    users[0] - id user
        #    users[1] - media
        #    users[2] - variancia
        max_score = -numpy.inf
        max_score_user = None
        
        try:
            
            #pylab.figure()
            #pylab.plot(data)
            #pylab.show()
            
            if self.is_trained:
                
                cmp_user = scipy.zeros([self.n_stft,self.stft_size])
                X = stft0(data, self.fs, self.framesz, self.hop)
                i=0                
                for laps in X:
                    aux = scipy.log(pow(abs(laps[0:self.stft_size]),2))
                    cmp_user[i] = cmp_user[i] + aux
                    i = i + 1
                cmp_user = scipy.reshape(cmp_user, self.n_stft*self.stft_size)
                
                #pylab.figure()
                #pylab.plot(cmp_user)
                #pylab.show()
                
                constB = 1.0/(numpy.sqrt(2.0*math.pi*self.global_train[1])) 
                
                score_array = numpy.zeros((5,2))
                #user_array = numpy.zeros(24)
                
                for users in self.users_train:
                    
                    constA = 1.0/(numpy.sqrt(2.0*math.pi*users[2])) 
                    
                    
                    
                    A = constA * numpy.exp(-(((cmp_user-users[1])**2)/(2.0*users[2])))+1
                    B = constB * numpy.exp(-(((cmp_user-self.global_train[0])**2)/(2.0*self.global_train[1])))+1
                    
                    #pylab.figure()
                    #pylab.plot(A)
                    #pylab.plot(B)
                    #pylab.show()
                    
                    #ood=0
                    #for a in B:
                    #    if not a:
                    #        good = 1
                    #if good:
                    #    continue
                    
                    score = numpy.sum(numpy.log(A/B)*(self.users_distance[users[0]-1]>self.classify_k))
                    
                    if score >= numpy.Inf or score <= -numpy.Inf:
                        continue
                    #print users[0],"->>",score
                    
                    if score>score_array[0,0]:
                        score_array[0, 0] = score
                        score_array[0, 1] = users[0]
                        score_array = numpy.array(sorted(score_array, key=lambda score: score[0]))
                    
                    #print score_array
                    #print users[0]
                    
                    if score>max_score:
                        max_score = score
                        max_score_user = users[0]
                        

                #return max_score_user, max_score
                    
            else:
                print "Please perform the train method before classification."
        except Exception as e:
            print e
            print traceback.format_exc()
        return max_score_user, max_score, score_array
        

if __name__ == '__main__':
    
    #idx_user = scipy.random.randint(0,5,size=20)
    idx_user = [0,0,1,1,2,2,3,3,4,4] 
    #idx_user = numpy.array(idx_user)
    
    h5file = h5py.File('dataset_IST.hdf5', 'r')
    
    users = []
    
    users.append(h5file['ECG']['AL']['REST']['segments'][0])
    users.append(h5file['ECG']['AL']['REST']['segments'][1])
    users.append(h5file['ECG']['CC']['REST']['segments'][2])
    users.append(h5file['ECG']['CC']['REST']['segments'][3])
    users.append(h5file['ECG']['FC']['REST']['segments'][4])
    users.append(h5file['ECG']['FC']['REST']['segments'][5])
    users.append(h5file['ECG']['MC']['REST']['segments'][6])
    users.append(h5file['ECG']['MC']['REST']['segments'][7])
    users.append(h5file['ECG']['PA']['REST']['segments'][8])
    users.append(h5file['ECG']['PA']['REST']['segments'][9])
    
    mytest = (h5file['ECG']['MC']['REST']['segments'][22])
    
    
    h5file.close()
    
    users = numpy.array(users)
    

    
    odi = odinaka(fs = 1000, framesz = 0.064, hop = 0.010, sinal_length = 0.7)
    
    
    
    print odi.is_trained
    odi.train(users, idx_user)
    print odi.is_trained
    

    
    
    
    
    #idx = 6
    print odi.classify(mytest)
    
    #print odi.users_train
    #print odi.users_distance
    
    #print odi.users_train[0][0]
    
    #pylab.figure()
    #pylab.plot(odi.global_train[0])
    #pylab.figure()
    #pylab.plot(odi.global_train[1])
    #pylab.show()
    
    

    
