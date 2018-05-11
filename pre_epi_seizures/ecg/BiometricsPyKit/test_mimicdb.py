"""
.. module:: 
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

# Biometrics
from misc import misc
from classifiers import classifiers
reload(classifiers)
from evaluation import evaluation
from featureextraction import featureextraction

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
            fd = gzip.open('D:/data/mimic2wdb/segmentation/essf/%s'%recid, 'rb')
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
            #if info['number_mean_waves_%s'%label] == 'all':
            #    data = misc.mean_waves(data, len(data))
            #elif info['number_mean_waves_%s'%label] > 1:
            #    data = misc.mean_waves(data, info['number_mean_waves_%s'%label])
            # determine median waves
            elif info['number_median_waves_%s'%label] == 'all':
                data = misc.median_waves(data, len(data))
            elif info['number_median_waves_%s'%label] > 1:
                data = misc.median_waves(data, info['number_median_waves_%s'%label])
            
            # get n random indexes  
            # ...
            
#            # quantization
#            if info['quantization'] > 0:
#                data = misc.quantize(data, levels=info['quantization'])
#                
#            if info['patterns2strings']:
#                res = []
#                for i, si in enumerate(data):
#                    line = ''.join('%d'%i for i in si)
#                    res.append(line)
#                data = res
                
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
    RESULTS = True
    
    T1T1 = True
    
    raw_input('go>')
    
    # ----------------------------------------------------------------------------------------------------------
    # ECG BASED BIOMETRICS SCRIPT 1 - PREPARE DATA
    # Database
    
    if LOAD:
        print "loading raw ...",
        sampling_rate = 125.
        print "done."
    if FILTER:
        filter_parameters = { }
        print "filtering ...",
        print "done."    
    if SEGMENT:
        segmentation_parameters = {'model': 'ESSF', 'SamplingRate': sampling_rate, 'Params': {'s_win': 0.03, 's_amp': 0.070}}
        print "segmentation ...",
        print "done."
    if OUTLIER:
        outlier_parameters = {'method': 'rcosdmean-mimicdb'}
        print "outlier detection ...",
        print "done."
        outlier_results = {'info': outlier_parameters}
    # ----------------------------------------------------------------------------------------------------------
    
    # ----------------------------------------------------------------------------------------------------------                 
    # ECG BASED BIOMETRICS SCRIPT 2 - PREPARE TO TEST 
    # Initialization
    if PREPARE: 
        info = {
                'database': 'mimicdb',                               # database
                #'filter': filter_parameters,                        # filter
                #'segmentation_method': segmentation_parameters,     # segmentation method used
                #'outlier_detection': outlier_parameters,            # outlier detection method
                'number_train_runs': 1,                            # number of train runs
                'number_test_runs': 1,                             # number of test runs
                'train_set': ['mimicdb'],                              # train set
                'test_set': ['mimicdb'],                               # test set
                'train_time': (0, 3500),                                   # train time window in seconds
                'test_time': (0, 3600),                                    # test time window in seconds                               # 
                'number_median_waves_train': 5,
                'number_median_waves_test': 5,
                'subtractmean': False,
                'normalization': [False, ''],
                'subpattern': [False, -1, -1]
                }
        data_folder = r'D:\data\mimic2wdb\segmentation\essf'
        # Check T1 T2 duration      
        records = {}
        list_files = glob.glob(os.path.join(data_folder, '*-*'))
        
        #good_ones = numpy.loadtxt('D:/data/mimic2wdb/meancurves/1199users-T1.txt', str)
        #good_ones = numpy.loadtxt('D:/data/mimic2wdb/segmentation/essf/234bestt1t2records.txt', str)
        #good_ones = numpy.loadtxt('D:/data/mimic2wdb/tests/t1t1/T1.txt', str)
        
        good_ones = numpy.loadtxt('D:/data/mimic2wdb/segmentation/essf/totalT1.txt', str)
        subs = numpy.loadtxt('D:/data/mimic2wdb/segmentation/essf/1017users.txt', str)
        
        for fn in list_files:
            nm = fn.split('\\')[-1]
            parts = nm.split('-')
            sub = parts[0]
            duration = parts[2]
            
            # abnormal
            ##if sub in ['s00439', 's00719', 's00770', 's00834', 's00894', 's01531', 's01973', 's02317', 's02477', 's02893', 's03372', 's03768', 's04113', 's04420', 's04633', 's04786', 's05712', 's05784', 's05786', 's06204', 's06349', 's06869', 's07655', 's07782', 's07849', 's08368', 's09016', 's09664', 's10638', 's11464', 's11609', 's12739', 's12798', 's13696', 's14167', 's14322', 's14772', 's14784', 's15023', 's15389', 's18910', 's19538', 's20013', 's20846', 's20936', 's24438', 's24922', 's25373', 's25446', 's26688']: continue
            
            #if sub+'-'+parts[1] not in good_ones: continue
            #if sub not in good_ones: continue
            
            if nm not in good_ones: continue
            if sub not in subs: continue
            
            if int(duration) < info['train_time'][1]:
                print "%s out (dur = %s)"%(nm, duration)
                continue
            if not records.has_key(sub): records[sub] = []
            records[sub].append(nm)
        # get subject info
        subjects = []
        for sub in records:
            if T1T1:
                subjects.append([sub, [records[sub][0]], [records[sub][0]]])
            else:
                if len(records[sub]) > 1:
                    subjects.append([sub, [records[sub][0]], [records[sub][1]]])        
            
        data_type = 'segments' # type of data that is going to be verified below according to info['train_time' and 'test_time'] to determine pattern indexes 
        
        #for s in subjects:
        #    print s
        print "%d subjects"%len(subjects)
        
        sts = []

        if OUTLIER:
            outlier_path =  'D:/data/mimic2wdb/outlier/%s/'%outlier_parameters['method']
            outlier_path += 'output-%s'
            for s in subjects:
                for i in [1, 2]:
                    fd = gzip.open(outlier_path%s[i][0], 'rb')
                    outlier_results[s[i][0]] = cPickle.load(fd)
                    fd.close()
                    
                    sts.append(len(outlier_results[s[i][0]]['0']))
        
        print "%0.3f +- %0.3f (m: %0.3f, M: %0.3f)"%(scipy.mean(sts), scipy.std(sts), min(sts), max(sts))
        
        
        # train and test patterns
        print "creating train and test patterns ...",
        train_patterns_idxs, test_patterns_idxs = evaluation.create_patterns_mimicdb(subjects, info, outlier_results)
        print "done."
        # save information about trial:
        info['subjects'] = subjects 
        info['train_patterns'] = train_patterns_idxs
        info['test_patterns'] = test_patterns_idxs
        
        test_name = 't1t1-1017'#raw_input('>')
        misc.save_information(info, 'dict', 'D:/data/mimic2wdb/tests/%s/info.dict'%test_name)
        
    # ----------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------
    # ECG BASED BIOMETRICS SCRIPT 3 A) - TEST (LOAD)
    # Load information
    #test_name = raw_input('>')
    
    source = 'D:/data/mimic2wdb/tests/%s/info.dict'%test_name # 'information source ... file saved above
     
    # load data
    fd = gzip.open(source, 'rb')
    info = cPickle.load(fd)
    fd.close()

    out_file = 'D:/data/mimic2wdb/distances/%d-%d-%s-%s.dict'
    
    data_type = 'segments'
    train_patterns = get_data(info, data_type, 'train')    # get train patterns corresponding to info['train_patterns'] indexes
    test_patterns = get_data(info, data_type, 'test')      # get test patterns corresponding to info['test_patterns'] indexes
    
    # classifier and parameters
    classifier_parameters = {'classifier': 'knn', 'parameters': {'k': 3, 'metric': 'cosine'}}
    # set rejection thresholds
    rejection_thresholds = scipy.arange(1.0, 2.1, 0.01)     # ed norm
    rejection_thresholds = scipy.arange(0.0, 900.25, 5)     # ed
    rejection_thresholds = scipy.arange(0.045, 0.075, 0.005)    # cosine
    
    # ----------------------------------------------------------------------------------------------------------
    
    # ----------------------------------------------------------------------------------------------------------
    # ECG BASED BIOMETRICS SCRIPT 3 B) - Compute distances
    #COMPUTE_DISTANCES = True
    if COMPUTE_DISTANCES:
        evaluation.run_classifier(info, test_patterns, train_patterns, classifier_parameters, rejection_thresholds, out_file)   
    # ----------------------------------------------------------------------------------------------------------          
    # RESULTS
    restrict2users = []
    
    report_title = ''
    dpath = 'D:/data/mimic2wdb/distances/'
    
    os.chdir(dpath)
    outfile = 'results-%s-%s.png'
    
    import time
    st = time.time()
    if RESULTS:    
        results = {0: {}}
        for fn in glob.glob(os.path.join(dpath, '*.dict')):
            fd = gzip.open(fn, 'rb')
            d = cPickle.load(fd)
            fd.close()
            #test_user = int(fn.split('-')[-2])
            test_user = fn.split('-')[-2]
            #train_user = int(fn.split('-')[-1].replace('.dict', ''))
            train_user = fn.split('-')[-1].replace('.dict', '')
            
            if restrict2users != [] and test_user not in restrict2users: continue
            
            out = {
                   'distances': d['distances'],
                   'labels': d['labels'], 
                   'true_labels': d['true_labels'],
                   'rejection_thresholds': rejection_thresholds}
            try: results[0][test_user][0][train_user] = out
            except: results[0][test_user]= {0: {train_user: out}}
        
        print '%0.3f'%(time.time()-st)
        
        fd = gzip.open('0-0-all.all', 'wb')
        cPickle.dump(results, fd)
        fd.close()

#        fd = gzip.open('0-0-all.all', 'rb')
#        results = cPickle.load(fd)
#        fd.close()
#        
#        aut_res = evaluation.authentication_results(results)
#        misc.save_information(aut_res, 'dict', 'aut_res-%s'%classifier_parameters['parameters']['metric'])
#        plot_biometric_results(rejection_thresholds, misc.load_information('aut_res-%s'%classifier_parameters['parameters']['metric']), 'FAR-FRR', outfile%('authentication',classifier_parameters['parameters']['metric']))
#        
#        id_res = evaluation.identification_results(results)
#        misc.save_information(id_res, 'dict', 'id_res-%s'%classifier_parameters['parameters']['metric'])        
#        plot_biometric_results(rejection_thresholds, misc.load_information('id_res-%s'%classifier_parameters['parameters']['metric']), 'FAR-FRR', outfile%('id',classifier_parameters['parameters']['metric']))        
        
        ##evaluation.results2report(report_title, info, dpath, 'report.pdf')