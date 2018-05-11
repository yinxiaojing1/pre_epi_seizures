'''
Created on 9 de Out de 2012

@author: Filipe
'''

from database import mongoH5
from preprocessing import preprocessing

import pylab
import scipy
import numpy
import scipy.signal as ss

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

if __name__ == '__main__':
    
    # ECG BASED BIOMETRICS
    
    # Get Raw Files
    
    # Filter
    
        # 0.5 to 40 Hz 4th order butterworth
        
        # 1.0 to 40 Hz FIR
        
        # 5.0 to 20 Hz FIR
    
        # Save filtered signal
    
    # Segment
    
        # engzee
        # ...
    
        # Save segments and corresponding r-peak indexes

    # Outlier Detection
    
        # method 1
        
        # method 2
        
        # save indexes
                     
    # TEST ECG ED Method
    
    # Initialization
    
    # number of runs
    number_runs = 10
    # segments type (filter and segmentation method used)
    filter_type = '5TO20'
    segmentation_method = 'ENGZEE'
    #number of users, database, experiment (e.g, 30 users from CYBH A1, A2)
    nusers = 30
    experiment = 'CYBH' 
    # train set and test set
    train_set, test_set = 'A1', 'A2'
    number_train_patterns, number_test_patterns = 10, 10  
    # outlier Detection method
    outlier_method = 'DBSCAN'
    # number of mean waves to be used:
    #    = 1 is single ECG Segment
    #    = 3 is mean wave of 3 consecutive ECG Segments
    #    = 5 is mean wave of 5 consecutive ECG Segments
    number_mean_waves = 5
    
    # get subjects
    subjects = get_subjects(nusers, experiment)
    # train and test patterns
    train_patterns, test_patterns = {}, {}
    
    # loop that creates train and test patterns 
    for run in xrange(number_runs):
        
        train_patterns[run] = {}
        test_patterns[run] = {}
        
        for subject in subjects:
            # get train data
            train_data = get_data_indexes(number_train_patterns, subject, train_set, experiment, outlier_method)
            # determine mean waves
            if number_mean_waves > 1: train_data = misc.mean_waves(train_data, number_mean_waves)
            # get test data
            test_data = get_data_indexes(number_test_patterns, subject, test_set, experiment, outlier_method)
            --- > get_data must assure that train and test patterns are mutually exclusive

            # determine mean waves
            if number_mean_waves > 1: test_data = misc.mean_waves(test_data, number_mean_waves)            
            
            train_patterns[run][subject] = train_data   # indexes of segments
            test_patterns[run][subject] = test_data     # indexes of segments
    # save 
    save following information about trial
    
    number_runs
    filter_type
    segmentation_method
    nusers
    experiment 
    train_set
    test_set
    number_train_patterns
    number_test_patterns  
    outlier_method
    number_mean_waves
    subjects
    train_patterns
    test_patterns
    
    
    # Load information
    
    # rejection threshold
    thresholds = 
    # classifier
    ...
                  
    #    authentication evaluation
    for test_run in xrange(number_runs):
        for test_user in test_patterns[test_run]:
            for train_run in xrange(number_runs):
                for train_user in train_patterns[train_run]:
                    # create classifier
                    
                    # train train_patterns[train_run][train_user]
                    
                    # classify test_patterns[test_run][test_user]
                    
                    for threshold in thresholds:
                        # save number of false acceptances, false rejections
                        
    #    identification
    for test_run in xrange(number_runs):
        for test_user in test_patterns[test_run]:
            for train_run in xrange(number_runs):
                # create classifier
                
                # train train_patterns[train_run]
                
                # classify test_patterns[test_run][test_user]
                
                for threshold in thresholds:
                    # save number of times the system fails
                        # a) user is in database but is rejected by system
                        # b) user is identified as other user       
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #===========================================================================
    # 
    # # butterworth filtering example
    # fs = 1000.  # sampling rate
    # x = scipy.arange(-2*scipy.pi, 2*scipy.pi, 1/fs)
    # y = scipy.cos(2*scipy.pi*5*x) + scipy.cos(2*scipy.pi*10*x) + scipy.cos(2*scipy.pi*45*x) + scipy.cos(2*scipy.pi*100*x)
    # [b, a] = ss.butter(4, 2.*50./fs, 'low')
    # y_lowf = ss.filtfilt(b, a, y)
    # [b, a] = ss.butter(4, [2.*45./fs, 2.*150./fs], 'band')
    # y_bandf = ss.filtfilt(b, a, y)
    # fig = pylab.figure()
    # ax = fig.add_subplot(111)
    # ax.set_xlabel('Frequency (Hz)')
    # ax.set_ylabel('|FFT(x)/N|')
    # f,X = centeredFFT(y,fs,False,ax)
    # f,X = centeredFFT(y_lowf,fs,False,ax)
    # f,X = centeredFFT(y_bandf,fs,False,ax)
    # pylab.show()
    # 
    #===========================================================================
    
    
    # test_real_time()
    
    
    
    #===========================================================================
    # # Data Info
    # config = {'dbName': 'CruzVermelhaPortuguesa', 'host': '193.136.222.234', 'port': 27017, 'path': r'\\193.136.222.220\cybh\data\CVP\hdf5'}
    # # config = {'dbName': 'CYBH', 'host': '193.136.222.234', 'port': 27017, 'path': r'\\193.136.222.220\cybh\mongo-hdf5\hdf5'}  
    # experiments = ['Fisioterapia']
    # db = mongoH5.bioDB(**config)
    # recs = mongoH5.records(**db)
    # rec_list = recs.get()['docList']
    # raw_signal = recs.getData(1, '/ECG/hand/raw', 0)['signal']
    # fs = 1000.
    # 
    # q = ''
    # while q != 'q':
    #    q = raw_input('>')
    #    n, l, h = q.split(' ') 
    #    n, l, h = int(n), float(l), float(h)
    #    
    #    [b, a] = ss.butter(n, [2.*l/fs, 2.*h/fs], 'band')
    #    filtered_signal = ss.filtfilt(b, a, raw_signal)
    #    
    #    pylab.figure()
    #    pylab.plot(filtered_signal, label='butter')        
    #    # pylab.plot(filt.zpdfr(raw_signal, 1000.,  h, l, n)['Signal'], label='zpdfr')
    #    # pylab.plot(ecg.filt(raw_signal, 1000., h, l, n)['Signal'], label='filt')
    #    # pylab.plot(preprocessing.firfilt(raw_signal, n, l, h, 1000.), label='fir')
    #    pylab.legend()
    #    pylab.title('%d %0.2f %0.2f'%(n,l,h))
    #    pylab.show()
    #    
    #    
    #    
    #===========================================================================
        
        
        
    
    #===========================================================================
    # import os
    # os.chdir('C:\Users\Filipe\Desktop\cardioprintdata')
    # 
    # import glob
    # import pylab
    # from preprocessing import preprocessing    
    # 
    # fig = pylab.figure(1)
    # for fname in glob.glob(os.path.join('.','*.raw')):
    #    signal = preprocessing.firfilt(pylab.loadtxt(fname), 300, 1., 20., 1000.)
    #    fig.clear()
    #    pylab.plot(signal[20000:30000], 'k')
    #    pylab.axis('off')
    #    pylab.title(fname.split('-')[-1])
    #    pylab.savefig(fname.replace('.raw','.png'))
    # print "done"
    # #===========================================================================
    # # 
    # # from reportlab.lib.pagesizes import letter
    # # from reportlab.platypus import SimpleDocTemplate, Image
    # # doc = SimpleDocTemplate("image.pdf", pagesize=letter)
    # # parts = []
    # # for fname in glob.glob(os.path.join('.','*.png')):
    # #    parts.append(Image(fname))
    # # doc.build(parts)    
    # #===========================================================================
    #===========================================================================
        
        
    #===========================================================================
    # ECG BASED BIOMETRICS
    # 
    # TEST ECG ED Method
    # 
    #    Load ECG Segments
    #     
    #        specify segments type (filter and segmentation method used)
    #        specify number of users, database, experiment (e.g, 30 users from CYBH)
    #        specify train set (example A1 activity)
    #        specify test set (example A2 activity)
    #        
    #        data = {
    #                'train': {
    #                          'user1': [seg1,
    #                                    seg2,
    #                                    ...],
    #                          'user2': [seg1,
    #                                    seg2,
    #                                    ...],
    #                          '...': ...}
    #                'test': {
    #                          'user1': [seg1,
    #                                    seg2,
    #                                    ...],
    #                          'user2': [seg1,
    #                                    seg2,
    #                                    ...],
    #                          '...': ...}
    #                }
    #        
    #    Outlier Detection
    #        specify method
    #        
    #        call outlier detection method
    #        
    #        non_outliers = {
    #                    'train': {
    #                              'user1': user1 non_outlier indexes,
    #                              'user2': user2 non_outlier indexes,
    #                              '...': ...}
    #                    'test': {
    #                             'user1': user1 non_outlier indexes,
    #                             'user2': user2 non_outlier indexes,
    #                             '...': ...}
    #                    }
    #        
    #    Specify number of mean waves to be used:
    #        = 1 is single ECG Segment
    #        = 3 is mean wave of 3 consecutive ECG Segments
    #        = 5 is mean wave of 5 consecutive ECG Segments
    #        
    #    for each run
    #    
    #        for each user
    #        
    #            aux = mean_waves ( data['train'][user][non_outliers['train'][user]] )
    #            train_set[run][user] = select ntrain random patterns from aux
    #            
    #            aux = mean_waves ( data['test'][user][non_outliers['test'][user]] )
    #            test_set[run][user] = select ntest random patterns from aux
    #                (assure that train and test patterns are mutually exclusive.
    #                 No worries if train experiment is different from test experiment (eg. A1 Vs A2 ))
    #            
    #        
    #            train_set= {
    #                        'run1': {
    #                                 'user1': [meanwave1,
    #                                           meanwave2,
    #                                           ...],
    #                                 'user2': [meanwave1,
    #                                           meanwave2,
    #                                           ...],
    #                                 '...': ...                                    
    #                                 }
    #                        'run2': {
    #                                 'user1': [meanwave1,
    #                                           meanwave2,
    #                                           ...],
    #                                 'user2': [meanwave1,
    #                                           meanwave2,
    #                                           ...],
    #                                 '...': ...                                    
    #                                 }
    #                        ...
    #                        'runn': {
    #                                 'user1': [meanwave1,
    #                                           meanwave2,
    #                                           ...],
    #                                 'user2': [meanwave1,
    #                                           meanwave2,
    #                                           ...],
    #                                 '...': ...                                    
    #                                 }                                                        
    #                        }
    #            
    #    authentication
    #    
    #        for each test_run
    #       
    #           for each test_user in test_set[test_run]:
    #               
    #               for each train_run
    #               
    #                   create classifier
    #               
    #                   train with train_set[train_run][test_user]
    #               
    #                   classify( test_set[test_run][test_user] )
    #               
    #                   apply rejection threshold
    #               
    #                   save number of false acceptances, false rejections
    #                   
    #        TOTAL OF NRUNS^2 
    #               
    #       identification
    #       
    #        for each test_run
    #       
    #           for each test_user in test_set[test_run]:
    #               
    #               for each train_run
    #               
    #                   create classifier
    #               
    #                   train with train_set[train_run]
    #               
    #                   classify( test_set[test_run][test_user] )
    #               
    #                   apply rejection threshold
    #               
    #                   save number of times the system fails
    #                       a) user is in database but is rejected by system
    #                       b) user is identified as other user    
    # 
    # TEST AC/LDA Method
    # 
    #    Load ECG filtered signals
    # 
    #        specify filter used
    #        specify number of users, database, experiment (e.g, 30 users from CYBH)
    #        specify train set (example A1 activity)
    #        specify test set (example A2 activity)
    #    
    #     for a given number of runs (nruns):
    #       create random train patterns (ntrain)
    #           save dict where each key is a user and has train segments (train_set)
    #           
    #       create random test patterns (ntest) (assure that train and test patterns are mutually exclusive)
    #            dict where each key is a user and has test segments (test_set)        
        
        
        