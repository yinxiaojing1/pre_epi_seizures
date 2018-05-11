"""
.. module:: BiometricWizard
   :platform: Unix, Windows
   :synopsis: This module provides various functions to...

.. moduleauthor:: Carlos Carreiras


"""

# Imports
# built-in
import os
import copy
import traceback
from itertools import izip

# 3rd party
import numpy
import matplotlib.pyplot as plt
from bidict import bidict

# BiometricsPyKit
import config
from classifiers import classifiers
from Cloud import parallel
from cluster import cluster
from cluster import templateSelection
from datamanager import datamanager, parted
from dimreduction import dimreduction
from evaluation import evaluation
from featureextraction import featureextraction
from misc import misc
from outlier import outlier
from preprocessing import preprocessing
# from stats import stats
from visualization import visualization
from wavelets import wavelets

# BioSPPy
from database import biomesh
import ecg.models as ecgmodule

# supported classifiers
CLF = ['SVM', 'KNN', 'Agrafioti', 'Dissimilarity', 'DissimilaritySimple', 'DissimilarityMore']


def selector(step):
    # given a string, return the correct function
    
    if step == 'filter':
        return Filter
    elif step == 'segment':
        return Segment
    elif step == 'cluster':
        return Cluster
    elif step == 'templateSelection':
        return TemplateSelection
    elif step == 'outlier':
        return Outlier
    elif step == 'featureExtraction':
        return FeatureExtraction
    elif step == 'featureSelection':
        return FeatureSelection
    else:
        raise ValueError, "Unknown processing step (%s)." % step



def Load(expConfig, store):
    # get starting data
    
    # connect to storage (generalize to datamanager)
    if expConfig['starting_data']['source'] == 'StorageBIT':
        db = biomesh.biomesh(**expConfig['starting_data']['srcParameters'])
    else:
        raise ValueError, "Unknown data source %s." % str(expConfig['starting_data']['source'])
    
    # get appropriate records
    trainRecs = db.records.listSymbolicTags(expConfig['train']['tags'])['idList']
    trainRecs = numpy.array(trainRecs, dtype='int')
    
    testRecs = db.records.listSymbolicTags(expConfig['test']['tags'])['idList']
    testRecs = numpy.array(testRecs, dtype='int')
    
    # subject per record
    subjectDict = {}
    allRecs = numpy.unique(numpy.concatenate((trainRecs, testRecs)))
    for rec in allRecs.tolist():
        subjectDict[rec] = db.records.getById(rec, {'subject': 1})['doc']['subject']
    
    # match to subjects
    trainSubs = numpy.array([subjectDict[rec] for rec in trainRecs])
    testSubs = numpy.array([subjectDict[rec] for rec in testRecs])
    
    # harmonize subjects
    commonSubs = numpy.intersect1d(trainSubs, testSubs)
    
    # convert subject IDs to normalized representation
    subjects = range(len(commonSubs))
    lbl2sub = bidict((k, v) for k, v in izip(subjects, commonSubs))
    
    # filter out the records whose subject is not common to both train and test sets
    f = lambda rec: subjectDict[rec] in commonSubs
    ntrainRecs = filter(f, trainRecs)
    ntestRecs = filter(f, testRecs)
    
    # set logic
    commonRecs = numpy.intersect1d(ntrainRecs, ntestRecs)
    xtrainRecs = numpy.setdiff1d(ntrainRecs, commonRecs).tolist()
    xtestRecs = numpy.setdiff1d(ntestRecs, commonRecs).tolist()
    commonRecs = commonRecs.tolist()
    
    # setup runs
    method = expConfig['run_setup']['method']
    if method in ['simple']:
        trainTasks, testTasks, nbRuns = simpleLoader(expConfig, subjects, lbl2sub, subjectDict, xtrainRecs, xtestRecs, commonRecs)
    elif method in ['LOOCV']:
        trainTasks, testTasks, nbRuns = loocvLoader(expConfig, subjects, lbl2sub, subjectDict, xtrainRecs, xtestRecs, commonRecs)
    elif method in ['LeadSimple']:
        trainTasks, testTasks, nbRuns = simpleLeadLoader(expConfig, subjects, lbl2sub, subjectDict, xtrainRecs, xtestRecs, commonRecs)
    elif method in ['LeadLOOCV']:
        trainTasks, testTasks, nbRuns = leadLoader(expConfig, subjects, lbl2sub, subjectDict, xtrainRecs, xtestRecs, commonRecs)
    else:
        raise ValueError, "Unknown loader method (%s)." % method
    
    # save tasks
    datamanager.skStore(config.getFilePath(('trainTasks.dict', )), trainTasks)
    datamanager.skStore(config.getFilePath(('testTasks.dict', )), testTasks)
    
    # store subjects (bidict)
    datamanager.skStore(config.getFilePath(('subjects.dict',)), lbl2sub)
    
    # close storage
    db.close()
    
    return trainTasks, testTasks, nbRuns


def simpleLoader(expConfig, subjects, lbl2sub, subjectDict, xtrainRecs, xtestRecs, commonRecs):
    # simple loader
    
    nbRuns = expConfig['run_setup']['parameters']['number_runs']
    data_name = expConfig['starting_data']['data_type'][0]
    workQ = parallel.getQueue()
    tasker = parallel.getIntValue(0)
    params = {'path': config.getPath((data_name, )),
              'data_name': data_name,
              'data_type': expConfig['starting_data']['data_type'][1],
              'run_setup': expConfig['run_setup'],
              }
    try:
        params['concatenate_records'] = expConfig['starting_data']['dataParameters']['concatenate_records']
    except KeyError:
        pass
    try:
        params['concatenate_signals'] = expConfig['starting_data']['dataParameters']['concatenate_signals']
    except KeyError:
        pass
    try:
        params['axis'] = expConfig['starting_data']['data_parameters']['axis']
    except KeyError:
        pass
    
    for lbl in subjects:
        # filter records
        f = lambda rec: lbl2sub[lbl] == subjectDict[rec]
        commonSR = filter(f, commonRecs)
        trainSR = filter(f, xtrainRecs)
        testSR = filter(f, xtestRecs)
        
        # update params
        aux = copy.deepcopy(params)
        aux['tasker'] = tasker
        aux['subject'] = lbl
        aux['trainSR'] = trainSR
        aux['testSR'] = testSR
        aux['commonSR'] = commonSR
        
        # add to queue
        workQ.put({'function': simpleTaskLoader,
                   'data': expConfig['starting_data']['srcParameters'],
                   'parameters': aux,
                   'taskid': lbl,
                   })
    
    # run in multiprocess (data mode, with lock)
    store = parallel.getDictManager()
    lock = parallel.getLock()
    parallel.runMultiprocess(workQ, store, log2file=True, lock=lock)
    
    # reorder tasks
    trainTasks = {}
    testTasks = {}
    
    for i in xrange(nbRuns):
        trainTasks[i] = {}
        testTasks[i] = {}
        
        for lbl in subjects:
            # train
            aux = store[lbl]['trainTasks'][i]
            aux.sort()
            trainTasks[i][lbl] = aux
            
            # test
            aux = store[lbl]['testTasks'][i]
            aux.sort()
            testTasks[i][lbl] = aux
    
    return trainTasks, testTasks, nbRuns


def loocvLoader(expConfig, subjects, lbl2sub, subjectDict, xtrainRecs, xtestRecs, commonRecs):
    # Leave One Out Cross Validation loader
    
    data_name = expConfig['starting_data']['data_type'][0]
    workQ = parallel.getQueue()
    tasker = parallel.getIntValue(0)
    params = {'path': config.getPath((data_name, )),
              'data_name': data_name,
              'data_type': expConfig['starting_data']['data_type'][1],
              'run_setup': expConfig['run_setup'],
              }
    try:
        params['concatenate_records'] = expConfig['starting_data']['dataParameters']['concatenate_records']
    except KeyError:
        pass
    try:
        params['concatenate_signals'] = expConfig['starting_data']['dataParameters']['concatenate_signals']
    except KeyError:
        pass
    try:
        params['axis'] = expConfig['starting_data']['data_parameters']['axis']
    except KeyError:
        pass
    
    for lbl in subjects:
        # filter records
        f = lambda rec: lbl2sub[lbl] == subjectDict[rec]
        commonSR = filter(f, commonRecs)
        trainSR = filter(f, xtrainRecs)
        testSR = filter(f, xtestRecs)
        
        # update params
        aux = copy.deepcopy(params)
        aux['tasker'] = tasker
        aux['subject'] = lbl
        aux['trainSR'] = trainSR
        aux['testSR'] = testSR
        aux['commonSR'] = commonSR
        
        # add to queue
        workQ.put({'function': loocvTaskLoader,
                   'data': expConfig['starting_data']['srcParameters'],
                   'parameters': aux,
                   'taskid': lbl,
                   })
    
    # run in multiprocess (data mode, with lock)
    store = parallel.getDictManager()
    lock = parallel.getLock()
    parallel.runMultiprocess(workQ, store, log2file=True, lock=lock)
    
    # reorder tasks
    trainTasks = {}
    testTasks = {}
    
    nbRuns = expConfig['run_setup']['parameters']['number_runs'] * expConfig['run_setup']['parameters']['k']
    for i in xrange(nbRuns):
        trainTasks[i] = {}
        testTasks[i] = {}
        
        for lbl in subjects:
            # train
            aux = store[lbl]['trainTasks'][i]
            aux.sort()
            trainTasks[i][lbl] = aux
            
            # test
            aux = store[lbl]['testTasks'][i]
            aux.sort()
            testTasks[i][lbl] = aux
    
    return trainTasks, testTasks, nbRuns


def windowLoader(expConfig, subjects, lbl2sub, subjectDict, xtrainRecs, xtestRecs, commonRecs):
    # load templates from larger container windows
    
    pass


def leadLoader(expConfig, subjects, lbl2sub, subjectDict, xtrainRecs, xtestRecs, commonRecs):
    # load multiple leads (LOOCV)
    
    data_name = expConfig['starting_data']['data_type'][0]
    workQ = parallel.getQueue()
    tasker = parallel.getIntValue(0)
    params = {'path': config.getPath((data_name, )),
              'data_name': data_name,
              'data_type': expConfig['starting_data']['data_type'][1],
              'run_setup': expConfig['run_setup'],
              }
    try:
        params['concatenate_records'] = expConfig['starting_data']['dataParameters']['concatenate_records']
    except KeyError:
        pass
    try:
        params['concatenate_signals'] = expConfig['starting_data']['dataParameters']['concatenate_signals']
    except KeyError:
        pass
    try:
        params['axis'] = expConfig['starting_data']['data_parameters']['axis']
    except KeyError:
        pass
    
    for lbl in subjects:
        # filter records
        f = lambda rec: lbl2sub[lbl] == subjectDict[rec]
        commonSR = filter(f, commonRecs)
        trainSR = filter(f, xtrainRecs)
        testSR = filter(f, xtestRecs)
        
        # update params
        aux = copy.deepcopy(params)
        aux['tasker'] = tasker
        aux['subject'] = lbl
        aux['trainSR'] = trainSR
        aux['testSR'] = testSR
        aux['commonSR'] = commonSR
        
        # add to queue
        workQ.put({'function': leadTaskLoader,
                   'data': expConfig['starting_data']['srcParameters'],
                   'parameters': aux,
                   'taskid': lbl,
                   })
    
    # run in multiprocess (data mode, with lock)
    store = parallel.getDictManager()
    lock = parallel.getLock()
    parallel.runMultiprocess(workQ, store, log2file=True, lock=lock)
    
    # reorder tasks
    trainTasks = {}
    testTasks = {}
    
    nbRuns = expConfig['run_setup']['parameters']['number_runs'] * expConfig['run_setup']['parameters']['k']
    for i in xrange(nbRuns):
        trainTasks[i] = {}
        testTasks[i] = {}
        
        for lbl in subjects:
            # train
            aux = store[lbl]['trainTasks'][i]
            aux.sort()
            trainTasks[i][lbl] = aux
            
            # test
            aux = store[lbl]['testTasks'][i]
            aux.sort()
            testTasks[i][lbl] = aux
    
    return trainTasks, testTasks, nbRuns


def simpleLeadLoader(expConfig, subjects, lbl2sub, subjectDict, xtrainRecs, xtestRecs, commonRecs):
    # load multiple leads (simple)
    
    data_name = expConfig['starting_data']['data_type'][0]
    workQ = parallel.getQueue()
    tasker = parallel.getIntValue(0)
    params = {'path': config.getPath((data_name, )),
              'data_name': data_name,
              'data_type': expConfig['starting_data']['data_type'][1],
              'run_setup': expConfig['run_setup'],
              }
    try:
        params['concatenate_records'] = expConfig['starting_data']['dataParameters']['concatenate_records']
    except KeyError:
        pass
    try:
        params['concatenate_signals'] = expConfig['starting_data']['dataParameters']['concatenate_signals']
    except KeyError:
        pass
    try:
        params['axis'] = expConfig['starting_data']['data_parameters']['axis']
    except KeyError:
        pass
    
    for lbl in subjects:
        # filter records
        f = lambda rec: lbl2sub[lbl] == subjectDict[rec]
        commonSR = filter(f, commonRecs)
        trainSR = filter(f, xtrainRecs)
        testSR = filter(f, xtestRecs)
        
        # update params
        aux = copy.deepcopy(params)
        aux['tasker'] = tasker
        aux['subject'] = lbl
        aux['trainSR'] = trainSR
        aux['testSR'] = testSR
        aux['commonSR'] = commonSR
        
        # add to queue
        workQ.put({'function': simpleLeadTaskLoader,
                   'data': expConfig['starting_data']['srcParameters'],
                   'parameters': aux,
                   'taskid': lbl,
                   })
    
    # run in multiprocess (data mode, with lock)
    store = parallel.getDictManager()
    lock = parallel.getLock()
    parallel.runMultiprocess(workQ, store, log2file=True, lock=lock)
    
    # reorder tasks
    trainTasks = {}
    testTasks = {}
    
    nbRuns = expConfig['run_setup']['parameters']['number_runs']
    for i in xrange(nbRuns):
        trainTasks[i] = {}
        testTasks[i] = {}
        
        for lbl in subjects:
            # train
            aux = store[lbl]['trainTasks'][i]
            aux.sort()
            trainTasks[i][lbl] = aux
            
            # test
            aux = store[lbl]['testTasks'][i]
            aux.sort()
            testTasks[i][lbl] = aux
    
    return trainTasks, testTasks, nbRuns


def concatenate(data, axis, vec):
    # robust concatenation
    try:
        data = numpy.concatenate(data, axis=axis)
    except ValueError, _:
        pass
    else:
        vec.append(data)
    
    return vec


def simpleTaskLoader(dbParams, lock, tasker, path, subject, trainSR, testSR,
                     commonSR, data_name, data_type, run_setup,
                     concatenate_records=False, concatenate_signals=True, axis=0):
    # Retrieves data of the specified type from database.
    ##### mdata is being completely ignored
    
    # check concatenation compatibility
    if concatenate_records and not concatenate_signals:
        raise ValueError, "Concatenation of records can not be performed without concatenation of signals."
    
    # number of runs
    nbRuns = run_setup['parameters']['number_runs']
    
    # choose partition function
    fractionFcn = parted.selector(run_setup['parameters']['selection_method'])
    
    # connect to db
    db = biomesh.biomesh(**dbParams)
    
    # signal concatenation function
    if concatenate_signals:
        def catS(data, var):
            return concatenate(data, axis, var)
    else:
        def catS(data, var):
            var.extend(data)
            return var
    
    # config folder
    config.folder = path
    
    # output
    output = {'trainTasks': {},
              'testTasks': {},
              }
    
    # db path
    wegits = data_type.split('/')
    wegC = '/' + '/'.join(wegits[1:])
    
    for i in xrange(nbRuns):
        output['trainTasks'][i] = []
        output['testTasks'][i] = []
        
        trainData = []
        testData = []
        
        # exclusively train data
        for rec in trainSR:
            aux = db.records[rec][data_type].list()['local']
            catS([db.records.getSignal(rec, wegC, item[1])['signal'] for item in aux], trainData)
        
        # exclusively test data
        for rec in testSR:
            aux = db.records[rec][data_type].list()['local']
            catS([db.records.getSignal(rec, wegC, item[1])['signal'] for item in aux], testData)
        
        # common data
        commonData = []
        for rec in commonSR:
            aux = db.records[rec][data_type].list()['local']
            catS([db.records.getSignal(rec, wegC, item[1])['signal'] for item in aux], commonData)
        
        # concatenate records
        if concatenate_records:
            # concatenate common data
            commonData = concatenate(commonData, axis, [])
            
            # separate common data
            for item in commonData:
                nb = len(item)
                if nb > 0:
                    use, unuse = fractionFcn(range(nb), i, **run_setup['parameters']['selection_parameters'])
                    trainData.append(item[use])
                    testData.append(item[unuse])
            
            # train data
            trainData = concatenate(trainData, axis, [])
            
            # test data
            testData = concatenate(testData, axis, [])
        else:
            # separate common data
            for item in commonData:         # common data is a matrix. only if not concatenated will this iterate
                nb = len(item)
                if nb > 0:
                    use, unuse = fractionFcn(range(nb), i, **run_setup['parameters']['selection_parameters'])
                    trainData.append(item[use])
                    testData.append(item[unuse])
        
        # tasks
        for item in trainData:
            # get and update task
            lock.acquire()
            try:
                task = tasker.value
                tasker.value = task + 1
            finally:
                lock.release()
            
            # save
            datamanager.skStore(config.getFilePath(('train', 'output-%s' % str(task))), item)
            output['trainTasks'][i].append(task)
        
        for item in testData:
            # get and update task
            lock.acquire()
            try:
                task = tasker.value
                tasker.value = task + 1
            finally:
                lock.release()
            
            # save
            datamanager.skStore(config.getFilePath(('test', 'output-%s' % str(task))), item)
            output['testTasks'][i].append(task)
    
    # close db
    db.close()
    
    return output


def loocvTaskLoader(dbParams, lock, tasker, path, subject, trainSR, testSR,
                    commonSR, data_name, data_type, run_setup,
                    concatenate_records=False, concatenate_signals=True, axis=0):
    # Retrieves data of the specified type from database.
    ##### mdata is being completely ignored
    
    # check concatenation compatibility
    if concatenate_records and not concatenate_signals:
        raise ValueError, "Concatenation of records can not be performed without concatenation of signals."
    
    # connect to db
    db = biomesh.biomesh(**dbParams)
    
    # signal concatenation function
    if concatenate_signals:
        def catS(data, var):
            return concatenate(data, axis, var)
    else:
        def catS(data, var):
            var.extend(data)
            return var
    
    # config folder
    config.folder = path
    
    # db path
    wegits = data_type.split('/')
    wegC = '/' + '/'.join(wegits[1:])
    
    # number of runs
    nbRuns = run_setup['parameters']['number_runs']
    k = run_setup['parameters']['k']
    tRuns = nbRuns * k
    
    # output
    output = {'trainTasks': {},
              'testTasks': {},
              }
    
    for i in xrange(tRuns):
        output['trainTasks'][i] = []
        output['testTasks'][i] = []
    
    # concatenate all records
    ### how can we skip this step?
    records = set()
    records.update(trainSR)
    records.update(testSR)
    records.update(commonSR)
    
    # load and concatenate data
    data = []
    for rec in records:
        aux = db.records[rec][data_type].list()['local']
        catS([db.records.getSignal(rec, wegC, item[1])['signal'] for item in aux], data)
    
    if concatenate_records:
        data = concatenate(data, axis, [])
    
    # separate into training and testing sets
    indx = range(k)
    for item in data:
        t = 0
        for i in xrange(nbRuns):
            # select randomly
            sel, _ = parted.randomSelection(range(len(item)), None, k)
            aux = item[sel]
            
            for j in xrange(k):
                # leave one out
                use, unuse = parted.leaveKOut(indx, j, k=1, random=False)
                
                # get and update task
                lock.acquire()
                try:
                    trainT = tasker.value
                    testT = trainT + 1
                    tasker.value = trainT + 2
                finally:
                    lock.release()
                
                # save
                datamanager.skStore(config.getFilePath(('train', 'output-%s' % str(trainT))), aux[use])
                output['trainTasks'][t].append(trainT)
                
                datamanager.skStore(config.getFilePath(('test', 'output-%s' % str(testT))), aux[unuse])
                output['testTasks'][t].append(testT)
                
                # update t
                t += 1
    
    # close db
    db.close()
    
    return output


def leadTaskLoader(dbParams, lock, tasker, path, subject, trainSR, testSR,
                    commonSR, data_name, data_type, run_setup,
                    concatenate_records=False, concatenate_signals=True, axis=0):
    # Retrieves data of the specified type from database.
    ##### mdata is being completely ignored
    
    # check concatenation compatibility
    if concatenate_records and not concatenate_signals:
        raise ValueError, "Concatenation of records can not be performed without concatenation of signals."
    
    # connect to db
    db = biomesh.biomesh(**dbParams)
    
    # signal concatenation function
    if concatenate_signals:
        def catS(data, var):
            return concatenate(data, axis, var)
    else:
        def catS(data, var):
            var.extend(data)
            return var
    
    # config folder
    config.folder = path
    
    # db path
    wegits = data_type.split('/')
    wegC = '/' + '/'.join(wegits[1:])
    
    # number of runs
    nbRuns = run_setup['parameters']['number_runs']
    k = run_setup['parameters']['k']
    tRuns = nbRuns * k
    
    # leads
    leads = run_setup['parameters']['leads']
    refLead = run_setup['parameters']['reference_lead']
    
    # output
    output = {'trainTasks': {},
              'testTasks': {},
              }
    
    for i in xrange(tRuns):
        output['trainTasks'][i] = []
        output['testTasks'][i] = []
    
    # concatenate all records
    ### how can we skip this step?
    records = set()
    records.update(trainSR)
    records.update(testSR)
    records.update(commonSR)
    
    # load and concatenate data
    data = {}
    for l in leads:
        data[l] = []
        d = data_type % l
        w = wegC % l
        
        for rec in records:
            aux = db.records[rec][d].list()['local']
            try:
                catS([db.records.getSignal(rec, w, item[1])['signal'] for item in aux], data[l])
            except KeyError:
                raise KeyError, "record: %d, lead: %s, path: %s, signal: %s" % (rec, l, w, item[1])
                break
        
        if concatenate_records:
            data[l] = concatenate(data[l], axis, [])
    
    # separate into training and testing sets
    indx = range(k)
    for item in data[refLead]:
        t = 0
        for i in xrange(nbRuns):
            # select randomly
            sel, _ = parted.randomSelection(range(len(item)), None, k)
            
            for j in xrange(k):
                # leave one out
                use, unuse = parted.leaveKOut(indx, j, k=1, random=False)
                
                # get and update task
                lock.acquire()
                try:
                    trainT = tasker.value
                    testT = trainT + 1
                    tasker.value = trainT + 2
                finally:
                    lock.release()
                
                # save
                svTrain = {}
                svTest = {}
                for l in leads:
                    aux = item[sel]
                    svTrain[l] = aux[use]
                    svTest[l] = aux[unuse]
                
                datamanager.skStore(config.getFilePath(('train', 'output-%s' % str(trainT))), svTrain)
                output['trainTasks'][t].append(trainT)
                
                datamanager.skStore(config.getFilePath(('test', 'output-%s' % str(testT))), svTest)
                output['testTasks'][t].append(testT)
                
                # update t
                t += 1
    
    # close db
    db.close()
    
    return output


def simpleLeadTaskLoader(dbParams, lock, tasker, path, subject, trainSR, testSR,
                         commonSR, data_name, data_type, run_setup,
                         concatenate_records=False, concatenate_signals=True, axis=0):
    # Retrieves data of the specified type from database.
    ##### mdata is being completely ignored
    
    # check concatenation compatibility
    if concatenate_records and not concatenate_signals:
        raise ValueError, "Concatenation of records can not be performed without concatenation of signals."
    
    # number of runs
    nbRuns = run_setup['parameters']['number_runs']
    
    # leads
    leads = run_setup['parameters']['leads']
    refLead = run_setup['parameters']['reference_lead']
    
    # choose partition function
    fractionFcn = parted.selector(run_setup['parameters']['selection_method'])
    
    # connect to db
    db = biomesh.biomesh(**dbParams)
    
    # signal concatenation function
    if concatenate_signals:
        def catS(data, var):
            return concatenate(data, axis, var)
    else:
        def catS(data, var):
            var.extend(data)
            return var
    
    # config folder
    config.folder = path
    
    # output
    output = {'trainTasks': {},
              'testTasks': {},
              }
    
    # db path
    wegits = data_type.split('/')
    wegC = '/' + '/'.join(wegits[1:])
    
    # helper
    def helperS(data, records):
        for l in leads:
            data[l] = []
            d = data_type % l
            w = wegC % l
            
            for rec in records:
                aux = db.records[rec][d].list()['local']
                try:
                    catS([db.records.getSignal(rec, w, item[1])['signal'] for item in aux], data[l])
                except KeyError:
                    raise KeyError, "record: %d, lead: %s, path: %s, signal: %s" % (rec, l, w, item[1])
                    break
    
    def helperR(data):
        for l in leads:
            data[l] = concatenate(data[l], axis, [])
    
    for i in xrange(nbRuns):
        output['trainTasks'][i] = []
        output['testTasks'][i] = []
        
        trainData = {}
        testData = {}
        
        # exclusively train data
        helperS(trainData, trainSR)
        
        # exclusively test data
        helperS(testData, testSR)
        
        # common data
        commonData = {}
        helperS(commonData, commonSR)
        
        # concatenate records
        if concatenate_records:
            # concatenate common data
            helperR(commonData)
            
            # separate common data
            for k, item in enumerate(commonData[refLead]):
                nb = len(item)
                if nb > 0:
                    use, unuse = fractionFcn(range(nb), i, **run_setup['parameters']['selection_parameters'])
                    
                    for l in leads:
                        aux = commonData[l][k]
                        if len(aux) == nb:
                            trainData[l].append(aux[use]) 
                            testData[l].append(aux[unuse])
                        else:
                            raise ValueError, "Inconsistent number of items per lead."
            
            # train data
            helperR(trainData)
            # test data
            helperR(testData)
        else:
            # separate common data
            for k, item in enumerate(commonData[refLead]):
                nb = len(item)
                if nb > 0:
                    use, unuse = fractionFcn(range(nb), i, **run_setup['parameters']['selection_parameters'])
                    
                    for l in leads:
                        aux = commonData[l][k]
                        if len(aux) == nb:
                            trainData[l].append(aux[use]) 
                            testData[l].append(aux[unuse])
                        else:
                            raise ValueError, "Inconsistent number of items per lead."
        
        # tasks
        for item, _ in enumerate(trainData[refLead]):
            # get and update task
            lock.acquire()
            try:
                task = tasker.value
                tasker.value = task + 1
            finally:
                lock.release()
            
            # save
            data = {}
            for l in leads:
                data[l] = trainData[refLead][item]
            datamanager.skStore(config.getFilePath(('train', 'output-%s' % str(task))), data)
            output['trainTasks'][i].append(task)
        
        for item, _ in enumerate(testData[refLead]):
            # get and update task
            lock.acquire()
            try:
                task = tasker.value
                tasker.value = task + 1
            finally:
                lock.release()
            
            # save
            data = {}
            for l in leads:
                data[l] = testData[refLead][item]
            datamanager.skStore(config.getFilePath(('test', 'output-%s' % str(task))), data)
            output['testTasks'][i].append(task)
    
    # close db
    db.close()
    
    return output


def Filter(tasks, parameters, case):
    # run a filtering task
    
    # select the function to apply to each task
    ### implement selector somewhere appropriate
    method = parameters['method']
    if method is 'butter':
        fcn = preprocessing.butter
    elif method is 'fir':
        fcn = preprocessing.firfilt
    elif method is 'wavelet':
        fcn = wavelets.RDWT
    else:
        raise ValueError, "Unknown filtering method (%s)." % method
    
    # perform segmentation
    srcPath = config.getPath((parameters['src'], case))
    dstPath = config.getPath(('filtered', case))
    workQ = parallel.getQueue()
    for recid in tasks.iterkeys():
        for i in xrange(len(tasks[recid])):
            workQ.put({
                       'function': fcn,
                       'filePath': srcPath,
                       'parameters': parameters['parameters'],
                       'taskid': tasks[recid][i]
                       })
    
    # run in multiprocessing (file mode)
    parallel.runMultiprocess(workQ, dstPath, mode='file', log2file=True)


def Segment(tasks, parameters, case):
    # run a segmentation task
    
    # select the function to apply to each task
    ### implement selector somewhere appropriate
    method = parameters['method']
    if method is 'engzee':
        fcn = ecgmodule.batch_engzee
    elif method is 'wavelet':
        fcn = wavelets.waveletSegments
    else:
        raise ValueError, "Unknown segmentation method (%s)." % method
    
    # perform segmentation
    srcPath = config.getPath((parameters['src'], case))
    dstPath = config.getPath(('segments', case))
    workQ = parallel.getQueue()
    for recid in tasks.iterkeys():
        for i in xrange(len(tasks[recid])):
            workQ.put({
                       'function': fcn,
                       'filePath': srcPath,
                       'parameters': parameters['parameters'],
                       'taskid': tasks[recid][i]
                       })
    
    # run in multiprocessing (file mode)
    parallel.runMultiprocess(workQ, dstPath, mode='file', log2file=True)


def Cluster(tasks, parameters, case):
    # run a clustering task
    
    # select the function to apply to each task
    fcn = cluster.selector(parameters['method'])
    
    # perform clustering
    srcPath = config.getPath((parameters['src'], case))
    dstPath = config.getPath(('clusters', case))
    workQ = parallel.getQueue()
    for recid in tasks.iterkeys():
        for i in xrange(len(tasks[recid])):
            workQ.put({
                       'function': fcn,
                       'filePath': srcPath,
                       'parameters': parameters['parameters'],
                       'taskid': tasks[recid][i]
                       })
    
    # run in multiprocessing (file mode)
    parallel.runMultiprocess(workQ, dstPath, mode='file', log2file=True)


def TemplateSelection(tasks, parameters, case):
    # run a template selection task
    
    # select the function to apply to each task
    fcn = templateSelection.selector(parameters['method'])
    
    # perform template selection
    clustersPath = config.getPath(('clusters', case, 'output-%d'), verify=False)
    srcPath = config.getPath((parameters['src'], case))
    dstPath = config.getPath(('clusters', 'templates', case))
    workQ = parallel.getQueue()
    for recid in tasks.iterkeys():
        for i in xrange(len(tasks[recid])):
            taskid = tasks[recid][i]
            
            # load clusters
            try:
                clusters = datamanager.skLoad(clustersPath % taskid)
            except IOError:
                clusters = None
            localParams = copy.deepcopy(parameters['parameters'])
            localParams.update({'clusters': clusters})
            
            workQ.put({
                       'function': fcn,
                       'filePath': srcPath,
                       'parameters': localParams,
                       'taskid': taskid,
                       })
    
    # run in multiprocessing (file mode)
    parallel.runMultiprocess(workQ, dstPath, mode='file', log2file=True)


def Outlier(tasks, parameters, case):
    # run an outlier task
    
    # select the function to apply to each task
    fcn = outlier.selector(parameters['method'])
    
    # perform outlier detection
    srcPath = config.getPath((parameters['src'], case))
    dstPath = config.getPath(('outliers', case))
    workQ = parallel.getQueue()
    for recid in tasks.iterkeys():
        for i in xrange(len(tasks[recid])):
            workQ.put({
                       'function': fcn,
                       'filePath': srcPath,
                       'parameters': parameters['parameters'],
                       'taskid': tasks[recid][i]
                       })
    
    # run in multiprocessing (file mode)
    parallel.runMultiprocess(workQ, dstPath, mode='file', log2file=True)
    
    # perform outlier removal
    outliersPath = config.getPath(('outliers', case, 'output-%d'), verify=False)
    dstPath = config.getPath(('outliers', 'templates', case))
    workQ = parallel.getQueue()
    for recid in tasks.iterkeys():
        for i in xrange(len(tasks[recid])):
            # load outliers
            outliers = datamanager.skLoad(outliersPath % tasks[recid][i])
            workQ.put({
                       'function': outlier.removeOutliers,
                       'filePath': srcPath,
                       'parameters': {'outliers': outliers},
                       'taskid': tasks[recid][i]
                       })
    
    # run in multiprocessing (file mode)
    parallel.runMultiprocess(workQ, dstPath, mode='file', log2file=True)


def FeatureExtraction(tasks, parameters, case):
    # run a feature selection task
    
    # select the function to apply to each task
    fcn = featureextraction.selector(parameters['method'])
    
    # fill work queue
    srcPath = config.getPath((parameters['src'], case))
    dstPath = config.getPath(('featureExtraction', case))
    workQ = parallel.getQueue()
    for recid in tasks.iterkeys():
        for i in xrange(len(tasks[recid])):
            workQ.put({
                       'function': fcn,
                       'filePath': srcPath,
                       'parameters': parameters['parameters'],
                       'taskid': tasks[recid][i]
                       })
    
    # run in multiprocessing (file mode)
    parallel.runMultiprocess(workQ, dstPath, mode='file', log2file=True)


def FeatureSelection(tasks, parameters, case):
    # run a feature selection task
    
    # fill work queue
    srcPath = config.getPath((parameters['src'], case))
    dstPath = config.getPath(('features', case))
    workQ = parallel.getQueue()
    for recid in tasks.iterkeys():
        for i in xrange(len(tasks[recid])):
            workQ.put({
                       'function': features,
                       'filePath': srcPath,
                       'parameters': parameters['parameters'],
                       'taskid': tasks[recid][i]
                       })
    
    # run in multiprocessing (file mode)
    parallel.runMultiprocess(workQ, dstPath, mode='file', log2file=True)


def features(data, normalization=None, subpattern=None,
             number_mean_waves=None, number_median_waves=None,
             quantization=None, patterns2strings=None):
    # function to apply the finishing touches
    
    try:
        data = numpy.array(data['templates'])
    except ValueError:
        # data is already numpy?
        if not isinstance(data, numpy.ndarray):
            raise
    
    # normalization
    if normalization:
        fcn = preprocessing.normSelector(normalization)
        data = fcn(data)
    
    # sub-pattern
    if subpattern:
        data = featureextraction.subpattern(data, range(subpattern[0], subpattern[1]))
    
    # mean/median waves
    if number_mean_waves > 1:
        # mean waves
        data = misc.mean_waves(data, number_mean_waves)
    elif number_median_waves > 1:
        # median waves
        data = misc.median_waves(data, number_median_waves)
    
    # quantization
    if quantization > 0:
        data = misc.quantize(data, levels=quantization)
    
    # string patterns
    if patterns2strings:
        res = []
        for i, si in enumerate(data):
            line = ''.join('%d' % i for i in si)
            res.append(line)
        data = res
    
    return data


def blockPrepare(data, normalization=None, subpattern=None, number_mean_waves=None,
                 number_median_waves=None, quantization=None, patterns2strings=None,
                 nbTemplates=None, manList=None, subject=None):
    
    # ensure numpy
    data = numpy.array(data)
    
    # normalization
    if normalization:
        fcn = preprocessing.normSelector(normalization)
        data = fcn(data)
    
    # sub-pattern
    if subpattern:
        data = featureextraction.subpattern(data, range(subpattern[0], subpattern[1]))
    
    # mean/median waves
    if number_mean_waves > 1:
        # mean waves
        data = misc.mean_waves(data, number_mean_waves)
    elif number_median_waves > 1:
        # median waves
        data = misc.median_waves(data, number_median_waves)
    
    # quantization
    if quantization > 0:
        data = misc.quantize(data, levels=quantization)
    
    # string patterns
    if patterns2strings:
        res = []
        for i, si in enumerate(data):
            line = ''.join('%d' % i for i in si)
            res.append(line)
        data = res
    
    # verify number of templates
    if len(data) >= nbTemplates:
        manList.append(subject)
    
    return data


def prepareTemplates(tasks, src=None, subject=None, manList=None,
                     normalization=None, subpattern=None, number_mean_waves=None,
                     number_median_waves=None, quantization=None,
                     patterns2strings=None, nbTemplates=None):
    # function to apply the finishing touches
    
    # load all tasks of subject
    taskData = []
    for item in tasks:
        try:
            aux = datamanager.skLoad(src % item)
        except IOError:
            continue
        else:
            taskData.append(aux)
    
    # check data types
    if all([isinstance(item, dict) for item in taskData]):
        # dict, check for templates keyword
        if all([item.has_key('templates') for item in taskData]):
            data = []
            for item in taskData:
                data.extend(item['templates'])
            data = blockPrepare(data, normalization=normalization, subpattern=subpattern,
                                number_mean_waves=number_mean_waves, number_median_waves=number_median_waves,
                                quantization=quantization, patterns2strings=patterns2strings,
                                nbTemplates=nbTemplates, manList=manList, subject=subject)
        else:
            keys = set()
            for item in taskData:
                keys.update(item.keys())
            data = {k: [] for k in keys}
            for k in keys:
                for item in taskData:
                    aux = item.get(k, [])
                    data[k].extend(aux)
                data[k] = blockPrepare(data[k], normalization=normalization, subpattern=subpattern,
                                       number_mean_waves=number_mean_waves, number_median_waves=number_median_waves,
                                       quantization=quantization, patterns2strings=patterns2strings,
                                       nbTemplates=nbTemplates, manList=manList, subject=subject)
    elif all([isinstance(item, list) for item in taskData]):
        data = []
        for item in taskData:
            data.extend(item)
        data = blockPrepare(data, normalization=normalization, subpattern=subpattern,
                            number_mean_waves=number_mean_waves, number_median_waves=number_median_waves,
                            quantization=quantization, patterns2strings=patterns2strings,
                            nbTemplates=nbTemplates, manList=manList, subject=subject)
    elif all([isinstance(item, tuple) for item in taskData]):
        data = []
        for item in taskData:
            data.extend(item)
        data = blockPrepare(data, normalization=normalization, subpattern=subpattern,
                            number_mean_waves=number_mean_waves, number_median_waves=number_median_waves,
                            quantization=quantization, patterns2strings=patterns2strings,
                            nbTemplates=nbTemplates, manList=manList, subject=subject)
    elif all([isinstance(item, numpy.ndarray) for item in taskData]):
        data = []
        for item in taskData:
            data.extend(item)
        data = blockPrepare(data, normalization=normalization, subpattern=subpattern,
                            number_mean_waves=number_mean_waves, number_median_waves=number_median_waves,
                            quantization=quantization, patterns2strings=patterns2strings,
                            nbTemplates=nbTemplates, manList=manList, subject=subject)
    else:
        raise TypeError, "Unknown data type to prepare."
    
    return data


def Prepare(tasks, expConfig, case, run):
    # load the templates from temp files, apply finishing touches, and organize them by subject
    
    workQueue = parallel.getQueue()
    manList = parallel.getListManager()
    outPath = config.getPath(('templates', case))
    
    # copy params
    localParams = copy.deepcopy(expConfig[case]['prepareParameters'])
    fpath = config.getFilePath((localParams['src'], case, 'output-%d'))
    localParams.update({'src': fpath,
                        'subject': None,
                        'nbTemplates': expConfig[case]['min_nb_templates'],
                        })
    
    for sub in tasks.iterkeys():
        # update subject
        aux = copy.deepcopy(localParams)
        aux.update({'subject': sub,
                    'manList': manList,
                    })
        
        # fill queue
        workQueue.put({'function': prepareTemplates,
                       'data': tasks[sub],
                       'parameters': aux,
                       'taskid': '%d-%d' % (run, sub)})
    
    # run in multiprocess (data mode)
    parallel.runMultiprocess(workQueue, outPath, log2file=True)
    
    # store usable subjects
    subjects = list(set([item for item in manList]))
    datamanager.skStore(config.getFilePath(('templates', case, 'subjects-%d.dict' % run)), subjects)
    
    return None


def Load2Clf(subjects, case, run):
    # load data for the given run of the classifier
    
    data = {}
    src = config.getFilePath(('templates', case, 'output-%d-%d'))
    for sub in subjects:
        data[sub] = datamanager.skLoad(src % (run, sub))
    
    return data


def DimReduction(expConfig, testData, trainData, subjects):
    # run dim reduction
    ######## MISSING: figure out how to save data
    
    # select the function to apply to each task
    fcn = dimreduction.selector(expConfig['dimreduction']['method'])
    train_features = {}
    test_features = {}
    
    for run in xrange(expConfig['train']['number_runs']): 
        
        #flattens the list of train data and creates the train labels 
        trainData_ = []
        trainLabel = []
        for subject in trainData[run]:
            for item in subject:
                trainData_.append(item)
                trainLabel.append(subject)
                
        #trains the algorithm
        FCN = fcn()
        FCN.train(trainData_, trainLabel, expConfig['dimreduction']['parameters']['energy']) #LDA ainda n tem energia implentada, so pode ser 1
        # train_coef = FCN.eigen_values
        # test_coef = FCN.project(testData_) 
    
        #projects and saves data
        train_features[run] = {}
        test_features[run] = {}
        for subject in subjects:
            train_features[run][subject] = FCN.project(trainData [run] [subject])
            test_features[run][subject] = FCN.project(testData [run] [subject])
     
    return test_features, train_features


def Classify(expConfig, run, clf_name, subjects):
    # run a classification task
    
    # check supported classifiers
    clf_method = expConfig['classifier'][clf_name]['method']
    if clf_method not in CLF:
        raise ValueError, "Classifier %s not supported." % clf_method
    
    # get the classifier parameters
    parameters = copy.deepcopy(expConfig['classifier'][clf_name]['parameters'])
    
    if not parameters.has_key('io'):
        parameters['io'] = config.getPath(('classifiers', clf_name, 'io', str(run)))
    
    # select the classifier class to use
    classifier = classifiers.selector(expConfig['classifier'][clf_name]['method'])
    
    # perform the classification
    evalPath = config.getPath(('classifiers', clf_name, 'evaluation'))
    
    # instantiate classifier
    clf = classifier(**parameters)
    
    # change classifier name
    clf._rebrand('classifier-%d' % run)
    
    # load train data to memory
    trainData = Load2Clf(subjects, 'train', run)
    
    # train classifier
    clf.train(trainData, updateThresholds=False)
    
    # save classifier to file
    clf.save(config.getPath(('classifiers', clf_name)))
    # datamanager.gzStore(config.getFilePath(('classifiers', clf_name, 'classifier-%d.dict' % run)), clf)
    
    # load test data to memory
    testData = Load2Clf(subjects, 'test', run)
    
    # evaluate classifier
    res = clf.evaluate(data=testData,
                       rejection_thresholds=expConfig['classifier'][clf_name]['rejection_thresholds'],
                       dstPath=evalPath,
                       log2file=True)
    
    # save results to file
    datamanager.skStore(config.getFilePath(('results', clf_name, 'results-%s.dict' % run)), res)
    
    # save rejection thresholds
    # datamanager.gzStore(config.getFilePath(('classifiers', clf_name, 'rejection_thresholds.dict')), rejection_thresholds)
    
    return res['assessment']


def Results(expConfig, run, clf_name):
    # compute results
    
    # check supported classifiers
    clf_method = expConfig['classifier'][clf_name]['method']
    if clf_method not in CLF:
        raise ValueError, "Classifier %s not supported." % clf_method
    
    # load rejection thresholds from file
    rejection_thresholds = datamanager.skLoad(config.getFilePath(('classifiers', clf_name, 'rejection_thresholds.dict')))
    
    # load results from file
    res = datamanager.skLoad(config.getFilePath(('classifiers', clf_name, 'results-%d.dict' % run)))
    
    # assess results
    res_eval = evaluation.assessClassification(res, rejection_thresholds, log2file=True)
   
    # save
    path = config.getFilePath(('results', clf_name, 'results-%d.dict' % run))
    datamanager.skStore(path, res_eval)
    
    return res_eval


def visualizeSubject(thresholds, rates, filePath):
    # to plot subject FAR-FRR curves in multiprocessing
    
    fig = visualization.singleFARFRRCurve(thresholds, rates)
    fig.savefig(filePath)
    plt.close(fig)
    
    return None


def Visualize(expConfig, run, clf_name):
    # plot results
    
    # check supported classifiers
    clf_method = expConfig['classifier'][clf_name]['method']
    if clf_method not in CLF:
        raise ValueError, "Classifier %s not supported." % clf_method
    
    # load assessment results
    res_eval = datamanager.skLoad(config.getFilePath(('results', clf_name, 'results-%s.dict' % str(run))))
    idx = classifiers.eerIndex(expConfig['classifier'][clf_name]['method'])
    
    try:
        res_eval = res_eval['assessment']
    except KeyError:
        pass
    
    # FAR-FRR, EER (global)
    fig = visualization.EERCurves(res_eval, idx)
    fig.savefig(config.getFilePath(('results', clf_name, 'FAR-FRR-%s.png' % str(run))))
    fig.savefig(config.getFilePath(('results', clf_name, 'FAR-FRR-%s.pdf' % str(run))), dpi=500)
    plt.close(fig)
    
#     # FAR-FRR, EER (subject)
#     workQ = parallel.getQueue()
#     for s in res_eval['subject'].iterkeys():
#         workQ.put({
#                    'function': visualizeSubject,
#                    'data': res_eval['rejection_thresholds'],
#                    'parameters': {
#                                   'rates': res_eval['subject'][s]['authentication']['rates'],
#                                   'filePath': config.getFilePath(('results', clf_name, 'FAR-FRR-%s-%d.png' % (str(run), s))),
#                                   },
#                    'taskid': s,
#                    })
#     # run in multiprocess (data mode)
#     store = parallel.getDictManager()
#     parallel.runMultiprocess(workQ, store, log2file=True)
    
    return None


def popStatsHelper(data, key=None):
    # load and compute length
    
    trainF, testF = data
    
    output = {}
    if key is None:
        output['train'] = len(datamanager.skLoad(trainF))
        output['test'] = len(datamanager.skLoad(testF))
    else:
        output['train'] = len(datamanager.skLoad(trainF)[key])
        output['test'] = len(datamanager.skLoad(testF)[key])
    
    return output


def PopulationStatistics(nbRuns, subjects, key=None):
    # compute population statistics
    
    nbSubs = len(subjects)
    output = {'number_subjects': nbSubs,
              'number_runs': nbRuns,
              }
    
    if key is None:
        params = {}
    else:
        params = {'key': key}
    
    workQ = parallel.getQueue()
    for run in xrange(nbRuns):
        for sub in subjects:
            trainF = config.getFilePath(('templates', 'train', 'output-%d-%d' % (run, sub)))
            testF = config.getFilePath(('templates', 'test', 'output-%d-%d' % (run, sub)))
            aux = (trainF, testF)
            workQ.put({'data': aux,
                       'function': popStatsHelper,
                       'parameters': params,
                       'taskid': '%d-%d' % (run, sub),
                       })
    
    # run in multiprocessing (data mode)
    store = parallel.getDictManager()
    parallel.runMultiprocess(workQ, store, log2file=True)
    
    # gather results
    trainR = []
    trainRs = []
    testR = []
    testRs = []
    for run in xrange(nbRuns):
        trainS = []
        testS = []
        for sub in subjects:
            aux = store['%d-%d' % (run, sub)]
            # get nb templates
            trainS.append(aux['train'])
            testS.append(aux['test'])
            
        trainR.append(numpy.mean(trainS))
        trainRs.append(numpy.std(trainS, ddof=1)**2)
        
        testR.append(numpy.mean(testS))
        testRs.append(numpy.std(testS, ddof=1)**2)
    
    trainRs = numpy.array(trainRs) / float(nbRuns)
    testRs = numpy.array(testRs) / float(nbRuns)
    
    # compute mean
    output['number_train_templates'] = numpy.mean(trainR)
    output['std_train_templates'] = numpy.sqrt(numpy.mean(trainRs))
    
    # compute std (with error propagation)
    output['number_test_templates'] = numpy.mean(testR)
    output['std_test_templates'] = numpy.sqrt(numpy.mean(testRs))
    
    # save
    datamanager.skStore(config.getFilePath(('populationStatistics.dict', )), output)
    
    return None


def MongoReport(expConfig, nbRuns):
    # prepare the results for insertion into MongoDB
    
    output = {}
    for clf_name in expConfig['classifier'].iterkeys():
        output['%s' % clf_name] = {}
        idx = classifiers.eerIndex(expConfig['classifier'][clf_name]['method'])
        
        for run in xrange(nbRuns):
            # load classifier results
            res = datamanager.skLoad(config.getFilePath(('results', clf_name, 'results-%s.dict' % run)))['assessment']
            
            # add to dict
            output['%s' % clf_name]['%s' % run] = {'AEER': res['global']['authentication']['rates']['EER'][idx, 1],
                                                   'IEER': res['global']['identification']['rates']['EER'][idx, 1],
                                                   'EID': res['global']['identification']['rates']['EID'][idx, 1],
                                                   }
        
        # load global results
        res_global = datamanager.skLoad(config.getFilePath(('results', clf_name, 'results-global.dict')))
        
        # add to dict
        output['%s' % clf_name]['global'] = {'AEER': res_global['global']['authentication']['rates']['EER'][idx, 1],
                                             'IEER': res_global['global']['identification']['rates']['EER'][idx, 1],
                                             'EID': res_global['global']['identification']['rates']['EID'][idx, 1],
                                             }
    
    return output


def Main(expConfig, expID):
    try:
        # set folder
        config.folder = os.path.join(config.baseFolder, 'Exp-%s' % str(expID))
        config.deploy()
        
        # what is the answer to life, the universe and everything?
        config.setRandomSeed(42)
        
        # get logger
        logger = misc.getLogger('Exp %s - Main' % str(expID),
                                config.getFilePath(('log', 'main.log')), 'debug')
        logger.info('Starting execution of task.')
        
        # save configuration
        datamanager.skStore(config.getPath(('config.dict',), verify=False), expConfig)
        
        # connect data manager
        try:
            # st = datamanager.Store(expConfig['database'])
            st = None
        except Exception:
            raise
        else:
            # load starting data from store
            logger.info("Loading starting data.")
            trainTasks, testTasks, nbRuns = Load(expConfig, st)
            
            if nbRuns == 0:
                raise ValueError, "Number of runs is zero."
            
            runSubjects = []
            for run in xrange(nbRuns):
                logger.info("Executing run %d of %d." % (run + 1, nbRuns))
            
                # train data processing
                for step in expConfig['train']['processing_sequence']:
                    logger.info("Train - executing %s step." % step)
                    fcn = selector(step)
                    fcn(trainTasks[run], expConfig['train'][step], 'train')
                
                # test data processing
                for step in expConfig['test']['processing_sequence']:
                    logger.info("Test - executing %s step." % step)
                    fcn = selector(step)
                    fcn(testTasks[run], expConfig['test'][step], 'test')
                
                ### dim reduction
                
                # prepare train
                logger.info("Preparing train templates.")
                Prepare(trainTasks[run], expConfig, 'train', run)
                
                # prepare test
                logger.info("Preparing test templates.")
                Prepare(testTasks[run], expConfig, 'test', run)
                
                # harmonize Train-Test subjects
                logger.info("Harmonizing train-test subjects.")
                trainSubs = datamanager.skLoad(config.getFilePath(('templates', 'train',
                                                                   'subjects-%d.dict' % run)))
                testSubs = datamanager.skLoad(config.getFilePath(('templates', 'test',
                                                                   'subjects-%d.dict' % run)))
                subjectsR = numpy.intersect1d(trainSubs, testSubs, assume_unique=True)
                datamanager.skStore(config.getFilePath(('templates', 'subjects-%d.dict' % run)), subjectsR)
                runSubjects.append(subjectsR)
            
            # harmonize subjects across runs
            subjects = numpy.array(runSubjects[0])
            for run in xrange(1, nbRuns):
                subjects = numpy.intersect1d(subjects, runSubjects[run], assume_unique=True)
            
            if len(subjects) == 0:
                raise ValueError, "No common subjects across runs."
            
            # save
            datamanager.skStore(config.getFilePath(('harmonizedSubjects.dict', )), subjects)
            
            # population stats
            logger.info("Computing population statistics.")
            if expConfig['run_setup']['method'] in ['LeadLOOCV']:
                PopulationStatistics(nbRuns, subjects, key=expConfig['run_setup']['parameters']['reference_lead'])
            else:
                PopulationStatistics(nbRuns, subjects)
            
            # classify
            for clf_name in expConfig['classifier'].iterkeys():
                runResults = []
                for run in xrange(nbRuns):
                    logger.info("Running classifier %s, run %d." % (clf_name, run))
                    out = Classify(expConfig, run, clf_name, subjects)
                    
                    # results
                    # logger.info("Assessing results for classifer %s, run %d.", (clf_name, run))
                    # out = Results(expConfig, run, clf_name)
                    runResults.append(out)
                    
                    # visualization
                    logger.info("Plotting figures for classifier %s, run %d." % (clf_name, run))
                    Visualize(expConfig, run, clf_name)
                
                # global results
                logger.info("Assessing gloabl results for classifer %s." % clf_name)
                res_global = evaluation.assessRuns(runResults, subjects)
                
                # save
                datamanager.skStore(config.getFilePath(('results', clf_name, 'results-global.dict')), res_global)
                
                # visualization
                logger.info("Plotting global figures for classifier %s." % clf_name)
                Visualize(expConfig, 'global', clf_name)
            
            # report to MongoDB
            logger.info("Building MongoDB report.")
            output = MongoReport(expConfig, nbRuns)
            
            # close data manager
            # st.close()
            
    except Exception, e:
        res = False
        output = {}
        logger.error("Error executing the task.")
        logger.exception(str(e))
    else:
        res = True
        logger.info("Task successfully executed.")
    
    return res, output



if __name__ == '__main__':
    # configuration
    config.numberProcesses = 10
    config.baseFolder = 'D:\\test'
    # config.baseFolder = 'C:\\Users\\FrancisD23\\Desktop\\Tese\\files_wizard'
    
    
    expConfig = {
                 'starting_data': {
                                   'source': 'StorageBIT',
                                   'srcParameters': {
                                                     'dbName': 'CVP',
                                                     'host': '193.136.222.234',
                                                     # 'host': 'localhost',
                                                     'dstPath': 'D:\\StorageBIT',
                                                     # 'dstPath': 'C:\\Users\\FrancisD23\\Documents\\Repositorios_tese\\StorageBIT',
                                                     'sync': False
                                                     },
                                   
                                   'data_type': ['segments', 'signals/ECG/hand/zee5to20/Segments/engzee/dmean'],
                                   'dataParameters': {
                                                      'concatenate_signals': True,
                                                      'concatenate_records': True,
                                                      'axis': 0,
                                                      },
                                   },
                 
                 'run_setup': {'method': 'simple',
                               'parameters': {'number_runs': 1,
                                              'selection_method': 'random',
                                              'selection_parameters': {'fraction': 0.5,
                                                                       },
                                              },
                               
                               # 'method': 'LOOCV',
                               # 'parameters': {'number_runs': 2,
                               #                'k': 5,
                               #                },
                               
                               
                               },
                 
                 'train': {
                           'tags': 'T1 & Sitting',
                           
                           'processing_sequence': [],
                           
                           'min_nb_templates': 3,   #3,

                        'outlier': {
                                    'src': 'segments',
                                    'method': 'dmean',
                                    'parameters': {
                                                   'metric': 'cosine',
                                                   'alpha': 0.5,
                                                   'R_Position': 200,
                                                   },
                                    },
                           
                           'prepareParameters': {
                                                 'src': 'segments',
                                                 'normalization': False,
                                                 'subpattern': False,
                                                 'number_mean_waves': 0,
                                                 'number_median_waves': 5,
                                                 'quantization': -1,
                                                 'patterns2strings': False,
                                                 },
                           },
                 
                 'test': {
                          'tags': 'T2 & Sitting',
                           
                           'processing_sequence': [],
                           
                           'min_nb_templates': 3,
                          
                           'outlier': {
                                       'src': 'segments',
                                       'method': 'dmean',
                                       'parameters': {
                                                      'metric': 'cosine',
                                                      'alpha': 0.5,
                                                      'R_Position': 200,
                                                      },
                                       },
                           
                           'prepareParameters': {
                                                 'src': 'segments',
                                                 'normalization': False,
                                                 'subpattern': False,
                                                 'number_mean_waves': 0,
                                                 'number_median_waves': 5,
                                                 'quantization': -1,
                                                 'patterns2strings': False,
                                                 },
                          },
                 
                 'dimreduction': {
                                  'method': '',
                                  'parameters': {},
                                  },
                 
                 'classifier': {
#                                'clf-knn': {
#                                            'method': 'KNN',
#                                            'rejection_thresholds': numpy.arange(0., 2001, 1.).tolist(),
#                                            'parameters': {
#                                                           'k': 3,
#                                                           'metric': 'euclidean',
#                                                           },
#                                            },
                                
                                'clf-svm': {
                                            'method': 'SVM',
                                            'rejection_thresholds': numpy.linspace(0.3, 1.0, 15).tolist(),
                                            'parameters': {
                                                           'kernel': 'linear',
                                                           'C': 1.0,
                                                           },
                                            },
                                },
                 }
    
    tmplConfig = {
                  'starting_data': {
                                   'source': 'StorageBIT',
                                   'srcParameters': {
                                                     'dbName': 'HSantaMarta',
                                                     'host': '193.136.222.234',
                                                     'dstPath': 'D:\\StorageBIT',
                                                     'sync': False
                                                     },
                                   
                                   'data_type': ['segments', 'signals/ECG/medZee5to20/%s/Segments/DMEAN_tst'],
                                   'dataParameters': {
                                                      'concatenate_signals': True,
                                                      'concatenate_records': True,
                                                      'axis': 0,
                                                      },
                                   },
                  
                  'run_setup': {'method': 'LeadLOOCV',
                                'parameters': {'number_runs': 1,
                                               'k': 4,
                                               'leads': ['I', 'II', 'III'],
                                               'reference_lead': 'I',
                                               },
                                },
                  
                  'train': {
                            'tags': 'SS_10_0',
                            
                            'processing_sequence': [],
                            
                            'min_nb_templates': 3,
                            
                            'filter': {
                                       'src': '',
                                       'method': 'medZee5to20',
                                       'parameters': {},
                                       },
                            
                            'segment': {
                                        'src': '',
                                        'method': 'hamilton',
                                        'parameters': {},
                                        },
                            
                            'outlier': {
                                        'src': '',
                                        'method': 'dmean_tst',
                                        'parameters': {
                                                       'metric': 'cosine',
                                                       'alpha': 0.5,
                                                       'beta': 1.5,
                                                       'R_Position': 100,
                                                       'absolute': False,
                                                       },
                                        },
                            
                            'prepareParameters': {
                                                  'src': 'segments',
                                                  'normalization': False,
                                                  'subpattern': False,
                                                  'number_mean_waves': 0,
                                                  'number_median_waves': 0,
                                                  'quantization': -1,
                                                  'patterns2strings': False,
                                                  },
                            },
                  
                  'test': {
                           'tags': 'SS_10_0',
                           
                           'processing_sequence': [],
                           
                           'min_nb_templates': 1,
                           
                           'filter': {
                                      'src': '',
                                      'method': 'medZee5to20',
                                      'parameters': {},
                                      },
                           
                           'segment': {
                                       'src': '',
                                       'method': 'hamilton',
                                       'parameters': {},
                                       },
                           
                           'outlier': {
                                       'src': '',
                                       'method': 'dmean_tst',
                                       'parameters': {
                                                      'metric': 'cosine',
                                                      'alpha': 0.5,
                                                      'beta': 1.5,
                                                      'R_Position': 100,
                                                      'absolute': False,
                                                      },
                                       },
                           
                           'prepareParameters': {
                                                 'src': 'segments',
                                                 'normalization': False,
                                                 'subpattern': False,
                                                 'number_mean_waves': 0,
                                                 'number_median_waves': 0,
                                                 'quantization': -1,
                                                 'patterns2strings': False,
                                                 },
                           },
                  
                  'dimreduction': {
                                   'method': '',
                                   'parameters': {},
                                   },
                  
                  'classifier': {'clf-dissim': {'method': 'Dissimilarity',
                                                'rejection_thresholds': numpy.linspace(0, 1.0, 10).tolist(),
                                                'parameters': {
                                                               'k': 3,
                                                               'metric': 'cosine',
                                                               'featmetric': 'cosine',
                                                               'leads': ('I', 'II', 'III'),
                                                               'reflead': 'I',
                                                               },
                                                },
                                 'clf-dissim-simple': {'method': 'DissimilaritySimple',
                                                       'rejection_thresholds': numpy.linspace(0, 200, 50).tolist(),
                                                       'parameters': {
                                                                      'k': 3,
                                                                      'metric': 'euclidean',
                                                                      'featmetric': 'euclidean',
                                                                      'leads': ('I', 'II', 'III'),
                                                                      'reflead': 'I',
                                                                      },
                                                       },
                                 },
                  }
    
    expID = 5001
    # expID = 'Tester-LOO'
    res, output = Main(expConfig, expID)
    if res:
        print "Task completed successfully!"
        print output
    else:
        print "Task terminated with errors."
    
