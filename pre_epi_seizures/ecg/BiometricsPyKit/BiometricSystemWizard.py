"""
.. module:: BiometricSystemWizard
   :platform: Unix, Windows
   :synopsis: This module provides various functions to...

.. moduleauthor:: Filipe Canento


"""

import sys
sys.path.append('C:\work\python\BioSPPy')
sys.path.append('C:\work\python\Ziv-Merhav-Cross-Parsing\src')
import numpy
#python BiometricSystemWizard.py

# Imports
import os, glob
import scipy
import pylab
import gzip
import cPickle
from itertools import izip
from multiprocessing import Process, Queue, Manager
from Queue import Empty

# Biometrics
from misc import misc
from outlier import outlier
from classifiers import classifiers
reload(classifiers)
from datamanager import datamanager
from evaluation import evaluation
from preprocessing import preprocessing
from featureextraction import featureextraction
# BioSPPy
import ecg.models as ecgmodule

# Global Variables
global NUMBER_PROCESSES
NUMBER_PROCESSES = 11
      
#def compute_distances(work_queue):
#    while 1:
#        try:
#            q = work_queue.get(block=False)
#            fn1 = q['fn1']
#            fn2 = q['fn2']
#            out_file = q['outfile']
#            function_to_use = q['function_to_use']
#            
#            u1, u2 = fn1.split('-')[-1], fn2.split('-')[-1]
#            
#            fd = gzip.open(fn1, 'rb')
#            test = scipy.array(cPickle.load(fd)['segments'])
#            fd.close()
#            
#            # normalize test
#            test /= float(scipy.mean(numpy.max(test,1)))
#            
#            if u1 != u2:
#                fd = gzip.open(fn2, 'rb')
#                train = scipy.array(cPickle.load(fd)['segments'])
#                fd.close()
#                
#                # normalize train
#                train  /= float(scipy.mean(numpy.max(train,1)))
#                
#            else:
#                train = test
#                
#                
#            
#            res = []
#            resi = []
#            
#            for tp in test:
#                dis = misc.wavedistance(tp, train, function_to_use)
#                
#                idxs =[]
#                map(lambda i: idxs.append([dis[i], i]), range(len(dis)))
#  
#                idxs.sort()
#                idxs = scipy.array(idxs)
#                
#                res.append(idxs[:10, 0])
#                resi.append(idxs[:10, 1])
#
#
#                
#            save_information({'distances': res}, 'dict', out_file%(u1, u2))
#                        
#        except Empty:
#            break       

def do_work(work_queue, output):
    while 1:
        try:
            q = work_queue.get(block=False)
            function = q['function']
            data = q['data']
            parameters = q['parameters']
            recid = q['recid']
            output[recid] = function(data, **parameters)
            
            fd = gzip.open('falc_temp/output-%d'%recid, 'wb')
            cPickle.dump(output[recid], fd)
            fd.close()
            
        except Empty:
            break
        
def segmentation_work(work_queue, output):
    while 1:
        try:
            q = work_queue.get(block=False)
            function = q['function']
            data = q['data']
            parameters = q['parameters']
            recid = q['recid']
            res = function(data, **parameters)
            output[recid] = {'segments': res['Segments'], 'R': res['R']}
            
            
#            fd = gzip.open('falc_temp/segmentation/output-%d'%recid, 'wb')
#            cPickle.dump(output[recid], fd)
#            fd.close()
                    
        except Empty:
            break
                
def ecg_segmentation(data, parameters):
    """
    Performs ECG segmentation on data according to specified parameters.

    Input:
    
        data (dict): input data. 
                     example: data = {1: record 1 data, 2: record 2 data}
        
        parameters (dict): filter parameters.
                           Example: parameters = {'model': 'engzee'}
                                                  
    Output:
        
        output (dict): output data where keys correspond to record id numbers.
                       example: output = { 1: {'segments': record 1 signal segments, 'R': record 1 r peak indexes},
                                           2: {'segments': record 2 signal segments, 'R': record 2 r peak indexes}}

    Configurable fields:{"name": "??.??", "config": {}, "inputs": ["data", "parameters"], "outputs": ["output"]}

    See Also:

    Notes:

    Example:
       
    """
    if parameters['model'] == 'engzee':
        model = ecgmodule.batch_engzee
    else:
        raise TypeError, "Model %s not implemented."%parameters['model']
    # create work queue
    work_queue = Queue()
    manager = Manager()
    output = manager.dict()
    output['info'] = parameters
    parameters.pop('model')
    # fill queue
    for recid in data.keys():
        if recid == 'info': continue
        work_queue.put({'function': model, 'data': data.get(recid), 'parameters': parameters, 'recid': recid})
    # create N processes and associate them with the work_queue and do_work function
    processes = [Process(target=segmentation_work, args=(work_queue,output,)) for _ in range(NUMBER_PROCESSES)]
    # lauch processes
    for p in processes: p.start()
    # wait for processes to finish
    print "waiting ..."
    for p in processes: p.join()
    print "waiting is over"
    for p in processes: p.terminate()
        
    return output

def outlier_detection(data, data_type, parameters):
    """
    Performs outlier detection on data of given data_type according to specified parameters.

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
    if parameters['method'] == 'dbscan':
        method = outlier.outliers_dbscan
    else:
        raise TypeError, "Method %s not implemented."%parameters['method']
    # create work queue
    work_queue = Queue()
    manager = Manager()
    output = manager.dict()
    output['info'] = parameters
    parameters.pop('method')
    # fill queue
    for recid in data.keys():
        if recid == 'info': continue
        work_queue.put({'function': method, 'data': data.get(recid).get('segments'), 'parameters': parameters, 'recid': recid})
    # create N processes and associate them with the work_queue and do_work function
    processes = [Process(target=do_work, args=(work_queue,output,)) for _ in range(NUMBER_PROCESSES)]
    # lauch processes
    for p in processes: p.start()
    # wait for processes to finish
    print "waiting ..."
    for p in processes: p.join()
    print "wait is over ..."
    for p in processes: p.terminate()
        
    return output

def get_data(info, data_type, label):
    
    
    #sampling_rate = info['filter_paramenters']['sampling_rate'] ##
    
    out = {}
    for run in info['%s_patterns'%label]:
        out[run] = {}
        for name in info['%s_patterns'%label][run]:
            # get data corresponding to info['label_patterns']
            recid = info['%s_patterns'%label][run][name][1]
            good_idxs = []
            map(lambda i: good_idxs.append(int(i)), info['%s_patterns'%label][run][name][0]) 
            
            # temp
            fd = gzip.open('D:/experiments/CVP/data/segmentation/engzee/output-%d'%recid, 'rb')
            data = scipy.array(cPickle.load(fd)[data_type])##
            fd.close()
            
#            a = pylab.find(data['R'] <= info['%s_time'%label][1]*sampling_rate)##
#            b = pylab.find(data['R'] >= info['%s_time'%label][0]*sampling_rate)##
#            idxs_time = scipy.intersect1d(a,b) ##
#            data_idxs = scipy.intersect1d(idxs_time, good_idxs) ##  
            
            data_idxs = good_idxs
            data = data[data_idxs] ##
            #data = db2data(data_type, info['database'], refine={'_id': recid})[data_idxs]
            
            
            if info['subtractmean']:
                data -= scipy.mean(data)
                
            if info['normalization'][0]:
                if info['normalization'][1] == 'meanofmaxs':
                    data /= scipy.mean(numpy.max(data, 1))
                elif info['normalization'][1] == 'medianofmaxs':
                    data /= scipy.median(numpy.max(data, 1))
            
            if info['subpattern'][0]:
                data = featureextraction.subpattern(data, range(info['subpattern'][1], info['subpattern'][2]))
            
            # determine mean waves
            if info['number_mean_waves_%s'%label] == 'all':
                data = misc.mean_waves(data, len(data))
            elif info['number_mean_waves_%s'%label] > 1:
                data = misc.mean_waves(data, info['number_mean_waves_%s'%label])
            # determine median waves
            elif info['number_median_waves_%s'%label] == 'all':
                data = misc.median_waves(data, len(data))
            elif info['number_median_waves_%s'%label] > 1:
                data = misc.median_waves(data, info['number_median_waves_%s'%label])
            
            # get n random indexes  
            # ...
            
            # quantization
            if info['quantization'] > 0:
                data = misc.quantize(data, levels=info['quantization'])
                
            if info['patterns2strings']:
                res = []
                for i, si in enumerate(data):
                    line = ''.join('%d'%i for i in si)
                    res.append(line)
                data = res
                
            out[run][name] = data
    return out

def plot_biometric_results(ths, results, mode, outfile=None):
    pylab.figure()
    if mode == 'FAR-FRR':
        EERi = scipy.argmin(abs(results['FAR']-results['FRR']))
        pylab.plot(ths, results['FAR'], label='FAR')
        pylab.plot(ths, results['FRR'], label='FRR')
        pylab.vlines(ths[EERi], 0, 1, 'r')
        pylab.text(ths[EERi], 0.5, '%0.3f'%results['FAR'][EERi])
        pylab.xlabel('Threshold')
        pylab.grid()
        pylab.legend()
    if outfile: pylab.savefig(outfile)
    else: pylab.show()
    pylab.close()
    
def plot_data(data, ax, title):
    sd_th = 2.0
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
        
if __name__ == '__main__':
    
    LOAD = True
    FILTER = True
    SEGMENT = True
    OUTLIER = True
    PREPARE = True
    COMPUTE_DISTANCES = True
    
    RESULTS = False ###
    
    if RESULTS: LOAD, FILTER, SEGMENT, OUTLIER, PREPARE, COMPUTE_DISTANCES = False,False,False,False,False,False
    
    SAVE2DB = False
    
    # ----------------------------------------------------------------------------------------------------------
    # ECG BASED BIOMETRICS SCRIPT 1 - PREPARE DATA
    # Database
    config = {'source': 'HDF5',
              'path': r'\\193.136.222.220\cybh\data\CVP\hdf5',
              'experiments': ['T1-Sitting', 'T2-Sitting', 'T1-Recumbent', 'T2-Recumbent'],
              'mapper': {'raw': 'signals/ECG/hand/raw',
                         'filtered': 'signals/ECG/hand/filtered/fir5to20',
                         'segments': 'signals/ECG/hand/segments/engzee',
                         'R': 'events/ECG/hand/R',
                         'outlier': 'events/ECG/hand/outlier/dbscan'}
              }
    st = datamanager.Store(config)
    
    if LOAD:
        # Get Raw Files
        print "loading raw ...",
        res = st.db2data('raw', refine={'_id': [0]})
        sampling_rate = float(res[0][0]['mdata']['sampleRate'])
#        raw_data = {}
#        for rid in res: raw_data[rid] = res[rid][0]['signal']
#        print "done."
        
    if FILTER:
        # Filter
        filter_parameters = {
                         'filter': 'fir',
                         'order': 300,
                         'sampling_rate': sampling_rate,
                         'low_cutoff': 5.,
                         'high_cutoff': 20.,
                         }
#        print "filtering ...",
#        filtered_data = preprocessing.filter_data(data=raw_data, parameters=filter_parameters)
#        print "done."
##        Add filtered data to database
#        filtered_data = {'info': filter_parameters}
#        for i in xrange(252):
#            fd = gzip.open('falc_temp/filtered/zee5to20/output-%d'%i, 'rb')
#            d = cPickle.load(fd)
#            fd.close()
#            filtered_data[i] = {'filtered': d}
#            if SAVE2DB: st.data2db(res, 'filtered')
    
    if SEGMENT:
        # ECG Segmentation
        segmentation_parameters = {
                                   'model': 'engzee',
                                   'SamplingRate': sampling_rate,
                                   'debug': False,
                                   'IF': False}
#        print "segmentation ...",
#        ecg_segments = ecg_segmentation(data=filtered_data, parameters=segmentation_parameters)
#        print "done."
##        Add segment to database
#        ecg_segments = {'info': {'segmentation': segmentation_parameters, 'filter': filter_parameters}}
#        for i in xrange(252):
#            fd = gzip.open('falc_temp/segmentation/engzee/output-%d'%i, 'rb')
#            d = cPickle.load(fd)
#            fd.close()
#            ecg_segments[i] = d
#            if SAVE2DB: st.data2db(res, ['segments', 'R'])
   
    if OUTLIER:
        # Outlier Detection
        outlier_parameters = {
                              'method': 'rcosdmean', #'480rmmin', #'dbscan',
                              'min_samples': 10, 
                              'eps': 0.95, 
                              'metric': 'euclidean'}
        print "outlier detection ...",
        # outlier_results = outlier_detection(data=ecg_segments, data_type='segments', parameters=outlier_parameters)
        print "done."
        # Add outlier results to database
        outlier_results = {'info': outlier_parameters}
        res = {'info': {'outlier': outlier_results, 'segmentation': segmentation_parameters, 'filter': filter_parameters}}
        for i in xrange(252):
            fd = gzip.open('D:/experiments/CVP/data/outlier/%s/output-%d'%(outlier_parameters['method'], i), 'rb')
            outlier_results[i] = cPickle.load(fd)
            fd.close()
            res[i] = {'outlier': outlier_results[i]}
            if SAVE2DB: st.data2db(res, 'outlier')
    # ----------------------------------------------------------------------------------------------------------
    
    # ----------------------------------------------------------------------------------------------------------                 
    # ECG BASED BIOMETRICS SCRIPT 2 - PREPARE TO TEST 
    # Initialization
    if PREPARE:
        info = {
                'database': config,                               # database
                #'users': 23,                                        # number of users or list of users
                'filter': filter_parameters,                        # filter
                'segmentation_method': segmentation_parameters,     # segmentation method used
                'outlier_detection': outlier_parameters,            # outlier detection method
                'number_train_runs': 1,                            # number of train runs
                'number_test_runs': 1,                             # number of test runs
                
                #'train_set': ['Enfermagem', config['experiments'][0]],                              # train set
                #'test_set': ['Enfermagem', config['experiments'][1]],                               # test set
                
                'train_set': ['T1-Sitting'],                              # train set
                'test_set': ['T2-Sitting'],                               # test set
                
                'train_time': (0, 10),                                   # train time window in seconds
                'test_time': (0, 10),                                    # test time window in seconds                               # 
                'number_mean_waves_train': 0,
                'number_mean_waves_test': 0,                        # number of mean waves to be used:
                                                                    #    = 1 is single ECG Segment
                                                                    #    = 3 is mean wave of 3 consecutive ECG Segments
                                                                    #    = 5 is mean wave of 5 consecutive ECG Segments
                'number_median_waves_train': 5,
                'number_median_waves_test': 5,
                
                'quantization': -1,
                'patterns2strings': False,
                
                'subtractmean': False,
                'normalization': [False, ''],
                'subpattern': [False, -1, -1]
                }
        
        data_type = 'segments' # type of data that is going to be verified below according to info['train_time' and 'test_time'] to determine pattern indexes 
        # get subject info
        subjects = st.subjectTTSets(info) # name, train record ids, test record ids
        
        print len(subjects)
        
        # train and test patterns
        print "creating train and test patterns ...",
        train_patterns_idxs, test_patterns_idxs = evaluation.create_patterns(st, info, outlier_results)
        print "done."
        # save information about trial:
        info['subjects'] = subjects 
        info['train_patterns'] = train_patterns_idxs
        info['test_patterns'] = test_patterns_idxs
        
        test_name = raw_input('>')
        misc.save_information(info, 'dict', 'falc_temp/tests/%s/info.dict'%test_name)
    # ----------------------------------------------------------------------------------------------------------
    
    # ----------------------------------------------------------------------------------------------------------
    # ECG BASED BIOMETRICS SCRIPT 3 A) - TEST (LOAD)
    # Load information
    test_name = raw_input('>')
    
    source = 'falc_temp/tests/%s/info.dict'%test_name # 'information source ... file saved above
     
    # load data
    fd = gzip.open(source, 'rb')
    info = cPickle.load(fd)
    fd.close()

    out_file = 'falc_temp/distances/%d-%d-%s-%s.dict'
    
    data_type = 'segments'
    train_patterns = get_data(info, data_type, 'train')    # get train patterns corresponding to info['train_patterns'] indexes
    test_patterns = get_data(info, data_type, 'test')      # get test patterns corresponding to info['test_patterns'] indexes
    
    # classifier and parameters
    classifier_parameters = {'classifier': 'knn', 'parameters': {'k': 3, 'metric': 'euclidean'}}
    # set rejection thresholds
    rejection_thresholds = scipy.arange(1.0, 2.1, 0.01)     # ed norm
    rejection_thresholds = scipy.arange(0.0, 900.25, 5)     # ed
    rejection_thresholds = scipy.arange(0.0, 0.1, 0.001)    # cosine
    
    rejection_thresholds = scipy.arange(0.0, 0.04, 0.0001)    # cosine
    
        
    # ----------------------------------------------------------------------------------------------------------
    
    # ----------------------------------------------------------------------------------------------------------
    # ECG BASED BIOMETRICS SCRIPT 3 B) - Compute distances
    #COMPUTE_DISTANCES = True
    if COMPUTE_DISTANCES:
        evaluation.run_classifier(info, test_patterns, train_patterns, classifier_parameters, rejection_thresholds, out_file)   
    # ----------------------------------------------------------------------------------------------------------          
    # RESULTS
    restrict2users = []
    
    # MEDIAN WAVES
    #all-all-all
    #aut
    #restrict2users = [30,29,60,61,35,23,24,25,27,20,49,46,47,44,45,43,40,1,3,5,4,6,9,8,13,12,17,15,32,58,10,59,22,14,16,19,54,31,56,51,36,53,52,33,55,37,18,57,50]
    #id
    
    #all-all-5#
    #aut
    #restrict2users = [30,29,61,35,23,24,27,20,46,47,44,43,40,1,3,5,8,13,12,17,15,32,58,10,59,22,19,54,31,56,51,36,53,52,33,55,37,18,57,50]
    #id#
    
    #all-5-5#
    # aut
    #restrict2users = [30,29,61,35,23,24,27,20,46,47,44,43,40,1,3,2,5,4,6,8,13,12,17,15,32,58,10,59,22,19,54,31,56,51,36,53,52,33,55,37,18,57,50]
    # id
    #restrict2users = [1,3,4,11,13,15,17,18,23,30,31,35,36,40,43,47,50,51,52,54,55,57,58,59,60,61]
    
    #all-5-5 QRS#
    # aut 
    #restrict2users = [30,34,23,24,20,43,40,1,3,5,6,17,58,11,10,59,54,56,51,36,53,52,33,18]
    # id
    #restrict2users = [1, 11, 23, 24, 35, 40, 51, 52, 59] 
    
    # MEAN WAVES
    #all-5-5#
    # aut
    #restrict2users = [30,29,61,35,23,20,46,44,43,40,1,3,2,5,4,6,8,13,12,17,15,32,58,10,59,22,19,54,31,56,51,36,53,52,33,55,37,18,57,50]
    # id
    #restrict2users = [1, 3, 4, 11, 13, 15, 17, 18, 22, 23, 30, 31, 35, 36, 40, 43, 47, 50, 51, 54, 55, 57, 58, 59, 60, 61]
    
    report_title = ''
    #dpath = 'C:/work/python/Cardioprint/BiometricsPyKit/falc_temp/distances/'
    dpath = 'C:/work/python/Cardioprint/BiometricsPyKit/falc_temp/classification/'
    
    os.chdir(dpath)
    outfile = 'results-%s-%s.png'
    if RESULTS:    
        results = {0: {}}
        for fn in glob.glob(os.path.join(dpath, '*.dict')):
            print fn
            fd = gzip.open(fn, 'rb')
            d = cPickle.load(fd)
            fd.close()
            test_user = int(fn.split('-')[-2])
            train_user = int(fn.split('-')[-1].replace('.dict', ''))
            
            if restrict2users != [] and test_user not in restrict2users: continue
            
            out = {
                   'distances': d['distances'],
                   'labels': d['labels'], 
                   'true_labels': d['true_labels'],
                   'rejection_thresholds': rejection_thresholds}
            try: results[0][test_user][0][train_user] = out
            except: results[0][test_user]= {0: {train_user: out}}
        
        aut_res = evaluation.authentication_results(results)
        misc.save_information(aut_res, 'dict', 'results/aut_res.dict')
        id_res = evaluation.identification_results(results)
        print id_res
        misc.save_information(id_res, 'dict', 'results/id_res.dict')
        
        plot_biometric_results(rejection_thresholds, misc.load_information('results/aut_res.dict'), 'FAR-FRR', outfile%'authentication')
        
        #misc.save_information(results, 'dict', 'C:/work/python/resultsd')
        
#        aut_res = evaluation.authentication_results(results)
#        misc.save_information(aut_res, 'dict', 'aut_res-%s'%classifier_parameters['parameters']['metric'])
#        plot_biometric_results(rejection_thresholds, misc.load_information('aut_res-%s'%classifier_parameters['parameters']['metric']), 'FAR-FRR', outfile%('authentication',classifier_parameters['parameters']['metric']))
#        
#        id_res = evaluation.identification_results(results)
#        misc.save_information(id_res, 'dict', 'id_res-%s'%classifier_parameters['parameters']['metric'])        
#        plot_biometric_results(rejection_thresholds, misc.load_information('id_res-%s'%classifier_parameters['parameters']['metric']), 'FAR-FRR', outfile%('id',classifier_parameters['parameters']['metric']))        
#        
        ##evaluation.results2report(report_title, info, dpath, 'report.pdf')
