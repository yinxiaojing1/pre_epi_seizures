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
from itertools import izip, cycle
from multiprocessing import Process, Queue, Manager
from Queue import Empty
# Reportlab
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, PageBreak, Table, TableStyle
# Biometrics
from misc import misc
from outlier import outlier
from classifiers import classifiers
from preprocessing import preprocessing
from datamanager import datamanager
# BioSPPy
import ecg.models as ecgmodule

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

# Outlier Detection
outlier_parameters = {
					  'method': 'dbscan', 
					  'min_samples': 10, 
					  'eps': 0.95, 
					  'metric': 'euclidean'}
#        print "outlier detection ...",
#        outlier_results = outlier_detection(data=ecg_segments, data_type='segments', parameters=outlier_parameters)
#        print "done."
##        Add outlier results to database
outlier_results = {'info': outlier_parameters}
#res = {'info': {'outlier': outlier_results, 'segmentation': segmentation_parameters, 'filter': filter_parameters}}
for i in xrange(252):
	fd = gzip.open('falc_temp/outlier/dbscan/output-%d'%i, 'rb')
	outlier_results[i] = cPickle.load(fd)
	fd.close()	
	

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


for exp in ['Enfermagem', 'Fisioterapia', 'Cardiopneumologia']:
    
    pylab.close('all')

    info = {'train_set': [exp, 'T1', 'Sitting'],
            'test_set': [exp, 'T2', 'Sitting'],
            'train_time': (0, 10),
            'test_time': (0, 10), 
            } 
    # get subject info
    subject_info = st.subjectTTSets(info) # name, train record ids, test record ids
    
    threshold = scipy.arange(0.01, 10.1, 0.1)
    
    data_path = 'falc_temp/segmentation/engzee/output-%d'
    output_folder = 'falc_temp/menagerie/%s/%d-%d-%s.%s'
    
    pylab.ioff()        
    fig1 = pylab.figure(1)
    ax11 = fig1.add_subplot(311)
    ax21 = fig1.add_subplot(312)
    ax31 = fig1.add_subplot(313)
    
    # plot segments and outliers
    for si in subject_info:
        uid = si[0]
    
        for rid, label in izip([si[1][0], si[2][0]], ['train_set', 'test_set']):
            fd = gzip.open(data_path%rid, 'rb')
            data = scipy.array(cPickle.load(fd)['segments'])
            fd.close()
            plot_data(data, ax11, 'all')
            idxs = map(lambda i: int(i), merge_clusters(outlier_results[rid]))
            plot_data(data[idxs], ax21, 'selected')
            outliers = outlier_results[rid]
            ax31.cla()
            ax31.plot(data[outliers['-1']].T, 'r')
            ax31.set_title('outliers')
            ax31.grid()
            ax31.autoscale(enable=True, axis='both', tight=True)  
            fig1.suptitle('%s DBSCAN'%info[label])
            fig1.savefig(output_folder%('segments', uid, rid, ''.join(i for i in info[label]), 'png'))    
        