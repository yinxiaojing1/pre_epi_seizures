"""
.. module:: BiometricSystemWizard
   :platform: Unix, Windows
   :synopsis: This module provides various functions to...

.. moduleauthor:: Carlos Carreiras


"""


# Imports
import numpy
import os, glob
import scipy
import pylab
import gzip
import cPickle
from itertools import izip
from multiprocessing import Process, Queue, Manager
from Queue import Empty
# Reportlab
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, PageBreak, Table, TableStyle
# Biometrics
import config
from misc import misc
from outlier import outlier
from classifiers import classifiers
from datamanager import datamanager
from evaluation import evaluation
from preprocessing import preprocessing
from featureextraction import featureextraction
from wavelets import wavelets
# BioSPPy
import ecg.models as ecgmodule

## Global Variables
#global NUMBER_PROCESSES
#NUMBER_PROCESSES = 3



def do_work(work_queue, output):
    while 1:
        try:
            q = work_queue.get(timeout=config.queueTimeOut)
            function = q['function']
            data = q['data']
            parameters = q['parameters']
            recid = q['recid']
            
            print recid
            
            output[recid] = function(data, **parameters)
            
            fd = gzip.open('capc_temp/outlier/output-%d'%recid, 'wb')
            cPickle.dump(output[recid], fd)
            fd.close()
            
        except Empty:
            break
        
def segmentation_work(work_queue, output):
    while 1:
        try:
            q = work_queue.get(timeout=config.queueTimeOut)
            function = q['function']
            data = q['data']
            parameters = q['parameters']
            recid = q['recid']
            res = function(data, **parameters)
            output[recid] = {'segments': res['Segments'], 'R': res['R']}
            
            
#            fd = gzip.open('capc_temp/segmentation/output-%d'%recid, 'wb')
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
    elif parameters['model'] == 'wavelet':
        model = wavelets.waveletSegments
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
    processes = [Process(target=segmentation_work, args=(work_queue,output,)) for _ in range(config.numberProcesses)]
    # lauch processes
    for p in processes: p.start()
    # wait for processes to finish
    print "waiting ..."
    for p in processes: p.join()
    print "waiting is over"
    for p in processes: p.terminate()
        
    return output

def outliers_dist2mean(data= None, metric='euclidean', alpha = 1.):
    """
    Detect outliers using distance to mean wave (using wavelets)

    Input:
        data (array): input data (number of observations x number of features).
        
        alpha (float): the threshold is mean + alpha * std.
    
        
    Output:
        res (dict): output dict with keys "0":normal and "-1":outliers and the corresponding data indexes.

    Configurable fields:{"name": "outliers_dist2mean", "config": {"alpha": 1.}, "inputs": ["data"]}

    See Also:
        

    Notes:
    

    Example:
       
    """    
    
    if metric == 'euclidean':
        # mean wave
        meanWave = numpy.mean(data, axis=0)
        # compute distances to mean wave
        dists = numpy.zeros(data.shape[0])
        for i in xrange(data.shape[0]):
            dists[i] = numpy.sqrt(numpy.sum((data[i, :] - meanWave)**2))
    elif metric == 'wavelet':
        # mean wave
        meanWave = wavelets.meanWave(data, axis=0)
        # compute distances to mean wave
        dists = numpy.zeros(data.shape[0])
        for i in xrange(data.shape[0]):
            dists[i] = wavelets.waveDist(data[i, :, :], meanWave)
    else:
        raise ValueError, "Metric not implmented."
    
    # rejection threshold
    lim = dists.mean() + alpha * dists.std(ddof=1)
    good = pylab.find(dists < lim)
    bad = pylab.find(dists >= lim)
    
    res = {'0': good, '-1': bad}
    
    return res

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
    elif parameters['method'] == 'dist2mean':
        method = outliers_dist2mean
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
#    # create N processes and associate them with the work_queue and do_work function
#    processes = [Process(target=do_work, args=(work_queue,output,)) for _ in range(NUMBER_PROCESSES)]
#    # lauch processes
#    for p in processes: p.start()
#    # wait for processes to finish
#    print "waiting ..."
#    for p in processes: p.join()
#    print "wait is over ..."
#    for p in processes: p.terminate()
    do_work(work_queue, output)
        
    return output

def get_data(st, info, data_type, label):
    out = {}
    for run in info['%s_patterns'%label]:
        out[run] = {}
        for name in info['%s_patterns'%label][run]:
            # get data corresponding to info['label_patterns']
            recid = info['%s_patterns'%label][run][name][1]
            data_idxs = []
            map(lambda i: data_idxs.append(int(i)), info['%s_patterns'%label][run][name][0]) 
            data = st.db2data(data_type, refine={'_id': [recid]})[recid][0]['signal'][:, :, 4:7]
            # data = st.db2data(data_type, refine={'_id': [recid]})[recid][0]['signal'].swapaxes(0, 1)
            
            # temp
            R = st.db.records.getEvent(recid, '/ECG/hand/zee5to20/R/engzee', 0)['timeStamps']
            filt = st.db.records.getSignal(recid, '/ECG/hand/zee5to20/Segments/engzee', 0)['signal']
            if R[0] < 200:
                # remove first segment
                data = data[1:]
            elif len(filt) - R[-1] < 400:
                # remove last segment
                data = data[:-1]
            
            
            # outliers
            data = data[data_idxs]
            
            # normalization
            if info['normalization'] == 'mean':
                data /= scipy.mean(numpy.max(data, 0))
            
            
            if info['subpattern']:
                data = featureextraction.subpattern(data, range(165,238))
            
            # determine mean waves
            if info['number_mean_waves_%s'%label] > 1:
                data = misc.mean_waves(data, info['number_mean_waves_%s'%label])
                
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
    
    # configurations for wavelets
    mainConfig = {'database': {
                               'source': 'StorageBIT',
                               'DBConnection': {
                                                'dbName': 'CVP',
                                                'host': '193.136.222.234',
                                                'dstPath': 'D:/BioMESH',
#                                                'sync': False
                                                },
                               'experiments': [
                                               'T1-Sitting',
                                               'T2-Sitting',
                                               'T1-Recumbent',
                                               'T2-Recumbent'
                                               ],
                               'mapper': {
                                          'raw': 'signals/ECG/hand/raw',
                                          'filtered': 'signals/ECG/hand/Wavelets/rbio3-3',
                                          'segments': 'signals/ECG/hand/Wavelets/rbio3-3/Segments/engzee',
                                          'R': 'events/ECG/hand/Wavelets/rbio3-3/R/engzee',
                                          'outlier': 'events/ECG/hand/Wavelets/rbio3-3/outlier/dbscan'
                                          }
                               },
                  
                  'filter': {
                             'filter': 'wavelet',
                             'wavelet': 'rbio3.3',
                             'level': 8,
                             'matrix': True
                             },
                  
                  'segmentation_method': {
                                          'model': 'engzee'
                                          },
                  
                  'outlier_detection': {
#                                        'method': 'dist2mean',
#                                        'alpha': 0.1,
#                                        'metric': 'euclidean'
                                        'method': 'rcosdmean',
#                                        'metric': 'wavelet',
#                                        'min_samples': 10, 
#                                        'eps': 0.7
                                        },
                  
                  'cluestering_method': {},
                  
                  'classification_method': {
                                            'classifier': 'knn',
                                            'parameters': {
                                                           'k': 3,
                                                           'metric': 'wavelet'
                                                           }
                                            },
                  
                  'data_type': 'segments',
                  
                  'number_train_runs': 1,
                  'number_test_runs': 1,
                  
                  'train_set': ['Enfermagem', 'Sitting', 'T1'],
                  'test_set': ['Enfermagem', 'Sitting', 'T2'],
                  
                  'train_time': (0, 10),
                  'test_time': (0, 10),
                  
                  'number_mean_waves_train': 0,
                  'number_mean_waves_test': 0,
                  
                  'number_median_waves_train': 5,
                  'number_median_waves_test': 5,
                  
                  'quantization': -1,
                  'patterns2strings': False,
                  'normalization': None,
                  'subpattern': False,
                  'rejection_thresholds': scipy.arange(0., 1.005, 0.001), # wavelets
                  'rejection_thresholds': scipy.arange(0., 2001, 1.), # segments
                  }
    
    LOAD = False
    FILTER = False
    SEGMENT = False
    OUTLIER = False
    PREPARE = False
    COMPUTE_DISTANCES = False
    
    RESULTS = True
    
#    if RESULTS: LOAD, FILTER, SEGMENT, OUTLIER, PREPARE, COMPUTE_DISTANCES = False, False, False, False, False, False
    
    SAVE2DB = False
    
    # ----------------------------------------------------------------------------------------------------------
    # ECG BASED BIOMETRICS SCRIPT 1 - PREPARE DATA
    # Database
    st = datamanager.Store(mainConfig['database'])
    
    if LOAD:
        # Get Raw Files
        print "loading raw ...",
        res = st.db2data('raw')
        sampling_rate = float(res[0][0]['mdata']['sampleRate'])
        raw_data = {}
        for rid in res: raw_data[rid] = res[rid][0]['signal']
        print "done."
        
    if FILTER:
        # Filter
        filter_parameters = mainConfig['filter']
        ################ onde vou buscar a sampleRate???????????
        filter_parameters.update({'sampleRate': sampling_rate})
        print "filtering ...",
        filtered_data = preprocessing.filter_data(data=raw_data, parameters=filter_parameters)
        print "done."
        # Add filtered data to database
        filtered_data = {'info': filter_parameters}
        for i in xrange(252):
            fd = gzip.open('capc_temp/filtered/zee5to20/output-%d'%i, 'rb')
            d = cPickle.load(fd)
            fd.close()
            filtered_data[i] = {'filtered': d}
            if SAVE2DB: st.data2db(res, 'filtered')
    
    if SEGMENT:
        # ECG Segmentation
        segmentation_parameters = mainConfig['segmentation_method']
        ################ onde vou buscar a sampleRate???????????
        segmentation_parameters.update({'sampleRate': sampling_rate})
        print "segmentation ...",
        ecg_segments = ecg_segmentation(data=filtered_data, parameters=segmentation_parameters)
        print "done."
#        Add segment to database
        ecg_segments = {'info': {'segmentation': segmentation_parameters, 'filter': filter_parameters}}
        for i in xrange(252):
            fd = gzip.open('capc_temp/segmentation/engzee/output-%d'%i, 'rb')
            d = cPickle.load(fd)
            fd.close()
            ecg_segments[i] = d
            if SAVE2DB: st.data2db(res, ['segments', 'R'])
   
    if OUTLIER:
        # Outlier Detection
        outlier_parameters = mainConfig['outlier_detection']
        print "outlier detection ...",
#        # load ECG segments
#        res = st.db2data('segments')
#        ecg_segments = {}
#        for rid in res: ecg_segments[rid] = {'segments': res[rid][0]['signal'][:, :, 4:7]}
#        # for rid in res: ecg_segments[rid] = {'segments': res[rid][0]['signal'].swapaxes(0, 1)}
#        outlier_results = outlier_detection(data=ecg_segments, data_type='segments', parameters=outlier_parameters)
#        print outlier_results.keys()
        print "done."
        # Add outlier results to database
        outlier_results = {'info': outlier_parameters}
        res = {'info': {'outlier': outlier_results, 'segmentation': mainConfig['segmentation_method'], 'filter': mainConfig['filter']}}
        for i in xrange(252):
            fd = gzip.open('capc_temp/outlier/output-%d'%i, 'rb')
            outlier_results[i] = cPickle.load(fd)
            fd.close()
            res[i] = {'outlier': outlier_results[i]}
            if SAVE2DB: st.data2db(res, 'outlier')
    # ----------------------------------------------------------------------------------------------------------
    
    # ----------------------------------------------------------------------------------------------------------                 
    # ECG BASED BIOMETRICS SCRIPT 2 - PREPARE TO TEST 
    # Initialization
    if PREPARE:
        data_type = 'segments' # type of data that is going to be verified below according to info['train_time' and 'test_time'] to determine pattern indexes 
        # get subject info
        subjects = st.subjectTTSets(mainConfig) # name, train record ids, test record ids
        # train and test patterns
        print "creating train and test patterns ...",
        
        train_patterns_idxs, test_patterns_idxs = {}, {}
        # loop that creates train and test patterns
        for run in xrange(mainConfig['number_train_runs']):
            train_patterns_idxs[run] = {}
            test_patterns_idxs[run] = {}
            for subject_info in subjects:
                for label, subidx, var in izip(['train', 'test'], [1, 2],[train_patterns_idxs, test_patterns_idxs]):
                    data = st.db2data(data_type, refine={'_id': [subject_info[subidx][0]]})
                    data = data[subject_info[subidx][0]][0]['signal']
                    
                    # temp
                    R = st.db.records.getEvent(subject_info[subidx][0], '/ECG/hand/zee5to20/R/engzee', 0)['timeStamps']
                    filt = st.db.records.getSignal(subject_info[subidx][0], '/ECG/hand/zee5to20/Segments/engzee', 0)['signal']
                    if R[0] < 200:
                        # remove first segment
                        data = data[1:]
                    elif len(filt) - R[-1] < 400:
                        # remove last segment
                        data = data[:-1]
                    
                    # remove outliers    
                    normal_ecg_segments_idxs = misc.merge_clusters(outlier_results[subject_info[subidx][0]])
                    set_idxs = scipy.intersect1d(range(len(data)), normal_ecg_segments_idxs)
                    
                    #set_idxs = range(len(data))
                    
                    # put in dict
                    var[run][subject_info[0]] = (set_idxs, subject_info[subidx][0]) 
        
        # train_patterns_idxs, test_patterns_idxs = evaluation.create_patterns(st, mainConfig, outlier_results)
        print "done."
        # save information about trial:
        mainConfig['subjects'] = subjects 
        mainConfig['train_patterns'] = train_patterns_idxs
        mainConfig['test_patterns'] = test_patterns_idxs
        
        misc.save_information(mainConfig, 'dict', 'capc_temp/mainConfig.dict')
    # ----------------------------------------------------------------------------------------------------------
    
    
    # ----------------------------------------------------------------------------------------------------------
    # ECG BASED BIOMETRICS SCRIPT 3 A) - TEST (LOAD)
    # Load information
    source = 'capc_temp/mainConfig.dict' # 'information source ... file saved above'
    out_file = 'capc_temp/classification/distances/%d-%d-%s-%s.dict' 
    
    # load data
    fd = gzip.open(source, 'rb')
    info = cPickle.load(fd)
    fd.close()
    
    data_type = 'segments'
    
#    # temp
#    info['subpattern'] = False
#    info['number_mean_waves_train'] = 0
#    info['number_mean_waves_test'] = 0
#    info['number_median_waves_train'] = 0
#    info['number_median_waves_test'] = 0
    
    
    train_patterns = get_data(st, info, data_type, 'train')    # get train patterns corresponding to info['train_patterns'] indexes
    test_patterns = get_data(st, info, data_type, 'test')      # get test patterns corresponding to info['test_patterns'] indexes
    # classifier and parameters
    classifier_parameters = mainConfig['classification_method']
    # set rejection thresholds
    rejection_thresholds = mainConfig['rejection_thresholds']
    
    # ----------------------------------------------------------------------------------------------------------
    
    print "Train"
    for key in train_patterns[0]:
        print key, train_patterns[0][key].shape
    
    print "Test"
    for key in test_patterns[0]:
        print key, test_patterns[0][key].shape
    
    # raise KeyboardInterrupt
    # ----------------------------------------------------------------------------------------------------------
    # ECG BASED BIOMETRICS SCRIPT 3 B) - Compute distances
    #COMPUTE_DISTANCES = True
    if COMPUTE_DISTANCES:
        evaluation.run_classifier(info, test_patterns, train_patterns, classifier_parameters, rejection_thresholds, out_file)   
    # ----------------------------------------------------------------------------------------------------------          
    # RESULTS
    
    RESULTS = True    
    
        
    # TEST = 'normbymeanofmaxs-5-dbscan-median'
    
    # dpath = 'D:/work/BiometricsPyKit/capc_temp/classification/distances/%s/'%TEST
    dpath = 'D:/work/BiometricsPyKit/capc_temp/classification'
    
    os.chdir(dpath)
    
    outfile = 'results/results-%s.png'
    if RESULTS:    
        results = {0: {}}
        for fn in glob.glob(os.path.join(dpath, 'distances', '*.dict')):
            print fn
            fd = gzip.open(fn, 'rb')
            d = cPickle.load(fd)
            fd.close()
            test_user = int(fn.split('-')[-2])
            train_user = int(fn.split('-')[-1].replace('.dict', ''))
            
            # if test_user not in [14, 15, 17, 18, 22, 23, 24, 30, 32, 33]: continue
            
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
        # plot_biometric_results(rejection_thresholds, misc.load_information('results/id_res.dict'), 'FAR-FRR', outfile%'identification')        
        # evaluation.results2report('CVP Enfermagem Wavelets', info, dpath, 'report.pdf')

        
    # ----------------------------------------------------------------------------------------------------------   
    
#    import BiometricSystemWizard
#    from BiometricSystemWizard import *
#    
#    data_path = 'capc_temp/segmentation/engzee'
#    ext = 'output*'
#    out_file = 'capc_temp/segmentation/engzee/distances/%sVS%s'
#    function_to_use = misc.msedistance
#    work_queue = Queue()
#    
#    print "filling queue ...",
#    for fn1 in glob.glob(os.path.join(data_path, ext)):
#        for fn2 in glob.glob(os.path.join(data_path, ext)):
#            work_queue.put({'fn1': fn1, 'fn2': fn2, 'function_to_use': function_to_use, 'outfile': out_file})
#    print "done."
#    # create N processes and associate them with the work_queue and do_work function
#    processes = [Process(target=compute_distances, args=(work_queue,)) for _ in range(NUMBER_PROCESSES)]
#    # lauch processes
#    print "working ...",
#    for p in processes: p.start()
#    # wait for processes to finish
#    for p in processes: p.join()
#    for p in processes: p.terminate()    
    

#
## plot segments and outliers
#data_path = 'capc_temp/segmentation/engzee/output-%d'
#output_folder = 'capc_temp/menagerie/%s/%d-%d-%s.%s'
#pylab.ioff()
#pylab.close('all')
#fig1 = pylab.figure(1)
#ax11 = fig1.add_subplot(311)
#ax21 = fig1.add_subplot(312)
#ax31 = fig1.add_subplot(313)
#for exp in ['Enfermagem']:#, 'Fisioterapia', 'Cardiopneumologia']:    
#    info = {'train_set': [exp, 'T1', 'Sitting'],
#            'test_set': [exp, 'T2', 'Sitting'],
#            'train_time': (0, 10),
#            'test_time': (0, 10), 
#            } 
#    # get subject info
#    subject_info = st.subjectTTSets(info) # name, train record ids, test record ids
#    
#    for si in subject_info:
#        uid = si[0]
#    
#        for rid, label in izip([si[1][0], si[2][0]], ['train_set', 'test_set']):
#            fd = gzip.open(data_path%rid, 'rb')
#            data = scipy.array(cPickle.load(fd)['segments'])
#            fd.close()
#            plot_data(data, ax11, 'all')
#            idxs = map(lambda i: int(i), merge_clusters(outlier_results[rid]))
#            plot_data(data[idxs], ax21, 'selected')
#            outliers = outlier_results[rid]
#            ax31.cla()
#            ax31.plot(data[outliers['-1']].T, 'r')
#            ax31.set_title('outliers')
#            ax31.grid()
#            ax31.autoscale(enable=True, axis='both', tight=True)  
#            fig1.suptitle('%s DBSCAN'%info[label])
#            fig1.savefig(output_folder%('segments', uid, rid, ''.join(i for i in info[label]), 'png'))
#            
#            fig1.clear()
#            ax11 = fig1.add_subplot(311)
#            ax21 = fig1.add_subplot(312)
#            ax31 = fig1.add_subplot(313)
            
            
#            
#
#rejection_thresholds = scipy.arange(0.01, 10.1, 0.1)
#results = {0: {}}
#
#data_path = r'C:\work\python\Cardioprint\BiometricsPyKit\capc_temp\segmentation\engzee\distances\%dVS%d'
#outfile = r'C:\work\python\Cardioprint\BiometricsPyKit\capc_temp\segmentation\engzee\%s.png'
#
#for exp in ['Enfermagem']:#, 'Fisioterapia', 'Cardiopneumologia']:    
#    info = {'train_set': [exp, 'T1', 'Sitting'],
#            'test_set': [exp, 'T2', 'Sitting'],
#            'train_time': (0, 10),
#            'test_time': (0, 10), 
#            }
#    # get subject info
#    subject_info = st.subjectTTSets(info) # name, train record ids, test record ids
#    # Get Data
#    for si1 in subject_info:
#        uid1 = si1[0]
#        rid1 = si1[1][0] # T1
#        for si2 in subject_info: 
#            uid2 = si2[0]
#            rid2 = si2[2][0] # T2       
#            
#            fd = gzip.open(data_path%(rid1, rid2), 'rb')
#            data = scipy.array(cPickle.load(fd)['distances'])
#            fd.close()
#            
#            
#            test_user = uid1
#            train_user = uid2
#            out = {
#                   'distances': data,
#                   'labels': len(data)*[str(uid1)], 
#                   'true_labels': len(data)*[str(uid2)],
#                   'rejection_thresholds': rejection_thresholds}
#            
#            try: results[0][test_user][0][train_user] = out
#            except: results[0][test_user]= {0: {train_user: out}}
#            
#plot_biometric_results(rejection_thresholds, authentication_results(results), 'FAR-FRR', outfile%'authentication')
#plot_biometric_results(rejection_thresholds, identification_results(results), 'FAR-FRR', outfile%'identification')            
            
   
HIST = False
if HIST:     
    # HISTOGRAMS
    threshold = scipy.arange(0.01, 1.1, 0.1)
    output_folder = 'capc_temp/menagerie/%s/%d.%s'
    dis_path = 'capc_temp/classification/distances/0-0-%d-%d.dict'
    
    dis_path = 'capc_temp/classification/distances/normbymeanofmaxs-5-ed-dbscan/0-0-%d-%d.dict'
    
    pylab.ioff()
    
    for exp in ['Enfermagem']:#, 'Fisioterapia', 'Cardiopneumologia']:    
        info = {'train_set': [exp, 'T1', 'Sitting'],
                'test_set': [exp, 'T2', 'Sitting'],
                'train_time': (0, 10),
                'test_time': (0, 10), 
                }
        # get subject info
        subject_info = st.subjectTTSets(info) # name, train record ids, test record ids
        
        pylab.close('all')
        fig2 = pylab.figure(2)
        ax12 = fig2.add_subplot(111)
        #ax22 = fig2.add_subplot(212)    
        
        # Get Data
        for si1 in subject_info:
            
            uid1 = si1[0]
            #rid1 = si1[1][0]    # T1
            #idxs1 = map(lambda i: int(i), merge_clusters(outlier_results[rid1]))
                
            #genuine
            color='g'
            hs = []
            hs_out = []
            for si2 in subject_info: 
                uid2 = si2[0]
                #rid2 = si2[2][0]        # T2
                
                #idxs2 = map(lambda i: int(i), merge_clusters(outlier_results[rid2]))
                if uid1 != uid2:
                    continue      
                try:
                    #fd = gzip.open(dis_path%(rid1, rid2), 'rb')
                    fd = gzip.open(dis_path%(uid1, uid2), 'rb')
                    dis = scipy.array(cPickle.load(fd)['distances'])
                    fd.close()
                except:
                    continue 
                #print rid2,
                # Determine histogram
                for i in dis: hs = scipy.hstack((hs, i))
                #for i in dis[idxs1]: hs_out = scipy.hstack((hs_out, i))  #check outliers
            if hs != []:
                ax12.hist(hs, rejection_thresholds, normed=True, color=color, alpha=0.3)
            ax12.set_title('all')
            ax12.autoscale(enable=True, axis='both', tight=True)    
#            if hs_out != []:
#                ax22.hist(hs_out, rejection_thresholds, normed=True, color=color, alpha=0.3)
#            ax22.set_title('selected')
#            ax22.autoscale(enable=True, axis='both', tight=True)  
            
            #imposter
            color='r'
            hs = []
            hs_out = []
            for si2 in subject_info: 
                uid2 = si2[0]
                rid2 = si2[2][0]    # T2  
                idxs2 = map(lambda i: int(i), misc.merge_clusters(outlier_results[rid2]))
                if uid1 == uid2:
                    continue      
                try:
                    #fd = gzip.open(dis_path%(rid1, rid2), 'rb')
                    fd = gzip.open(dis_path%(uid1, uid2), 'rb')
                    dis = scipy.array(cPickle.load(fd)['distances'])
                    fd.close()
                except:
                    continue 
                print rid2,
                # Determine histogram
                for i in dis: hs = scipy.hstack((hs, i))
                #for i in dis[idxs1]: hs_out = scipy.hstack((hs_out, i))  #check outliers
            if hs != []:
                ax12.hist(hs, rejection_thresholds, normed=True, color=color, alpha=0.3)
            ax12.set_title('all')
            ax12.autoscale(enable=True, axis='both', tight=True)    
#            if hs_out != []:
#                ax22.hist(hs_out, rejection_thresholds, normed=True, color=color, alpha=0.3)
#            ax22.set_title('selected')
#            ax22.autoscale(enable=True, axis='both', tight=True)  
         
            ax12.grid()
            #ax22.grid()
            
            fig2.suptitle('%s vs %s DBSCAN'%(info['train_set'], info['test_set'])) 
            fig2.savefig(output_folder%('hists', uid1, 'png'))
            
            #pylab.show()
            
            fig2.clear()
            ax12 = fig2.add_subplot(111)
            #ax22 = fig2.add_subplot(212)   
        
            ax12.cla()
            #ax22.cla()
 
        
        
    # # determine number of times the system fails for each rejection threshold
    # #    a) user is in database but is rejected by system    (FN)
    # #    b) user is identified as other user                 (FP)
    # # generate report
    # ----------------------------------------------------------------------------------------------------------
    
    
    


#    
#    hs = []
#    hs_out = []
#    for si2 in subject_info: 
#        uid2 = si2[0]
#        rid2 = si2[1][0]  
#        idxs2 = map(lambda i: int(i), merge_clusters(outlier_results[rid2]))
#        color = 'r'
#        if uid1 == uid2:
#            continue      
#            #color = 'g'
#        try:
#            fd = gzip.open(dis_path%(rid1, rid2), 'rb')
#            dis = scipy.array(cPickle.load(fd)['distances'])
#            fd.close()
#        except:
#            continue 
#        print rid2,
#        # Determine histogram
#        #hs = []
#        for i in dis: hs = scipy.hstack((hs, i))
##        ax12.hist(hs, threshold, normed=True, color=color, alpha=0.3)
##        ax12.set_title('all')
##        ax12.autoscale(enable=True, axis='both', tight=True)
#        #hs_out = []
#        for i in dis[idxs]: hs_out = scipy.hstack((hs_out, i))  #check outliers
##        ax22.hist(hs_out, threshold, normed=True, color=color, alpha=0.3)
##        ax22.set_title('selected')
##        ax22.autoscale(enable=True, axis='both', tight=True)
#
#
#
#    ax12.hist(hs, threshold, normed=True, color='r', alpha=0.3)
#    ax12.set_title('all')
#    ax12.autoscale(enable=True, axis='both', tight=True)    
#
#    ax22.hist(hs_out, threshold, normed=True, color='r', alpha=0.3)
#    ax22.set_title('selected')
#    ax22.autoscale(enable=True, axis='both', tight=True)


    


# # HISTOGRAMS
#import os, glob, pylab, scipy, cPickle, gzip
#pylab.ioff()
#fig = pylab.figure(1)
#x = scipy.arange(0,1000, 50)
#for u1 in xrange(12, 35):
#    fig.clear()
#    axg = fig.add_subplot(311)
#    axr = fig.add_subplot(312)
#    axa = fig.add_subplot(313)
#    for u2 in xrange(12, 35):
#        fd=gzip.open('0-0-%d-%d.dict'%(u1,u2), 'rb')
#        d=cPickle.load(fd)
#        fd.close()
#        if u1 == u2: ax, color = axg, 'g'
#        else: ax, color = axr, 'r' 
#        
#        print u1, u2
#        
#        ax.hist(d['distances'][:,0], x, color=color, alpha=0.5)
#        axa.hist(d['distances'][:,0], x, color=color, alpha=0.5)
#        
#    pylab.savefig('hist-%d.png'%u1)
    
    
    
    
    
