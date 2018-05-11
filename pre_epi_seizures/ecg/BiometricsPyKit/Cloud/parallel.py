"""
.. module:: server
   :platform: Unix, Windows
   :synopsis: This module implements the BITCloud parallelization services.

.. moduleauthor:: Carlos Carreiras


"""

# Imports
# built-in
import errno
import os
import multiprocessing
import time
import traceback
from multiprocessing import Condition, Manager, managers, Pipe, Pool, Process, Queue
from Queue import Empty

# 3rd party
import numpy as np

#BiometricsPyKit
import config
from datamanager import datamanager
from misc import misc



def getQueue():
    # get a new queue
    # useful because renders unnecessary the import of multiprocessing in every module that uses queues
    return Queue()


def getLock():
    # get a new lock
    return multiprocessing.Lock()


def getManager():
    # get a new manager
    return Manager()


def getDictManager():
    # get a new dict from manager
    
    try:
        obj = config.manager.dict()
    except AttributeError:
        config.manager = Manager()
        obj = config.manager.dict()
    
    return obj


def getListManager():
    # get a new list from manager
    
    try:
        obj = config.manager.list()
    except AttributeError:
        config.manager = Manager()
        obj = config.manager.list()
    
    return obj


def getEventManager():
    # get a new event from manager
    
    try:
        obj = config.manager.Event()
    except AttributeError:
        config.manager = Manager()
        obj = config.manager.Event()
    
    return obj


def getIntValue(value=None):
    # get a new integer value from manager
    
    try:
        obj = config.manager.Value('int', value)
    except AttributeError:
        config.manager = Manager()
        obj = config.manager.Value('int', value)
    
    return obj


def getPipe():
    # get a new pipe
    
    return Pipe()


def getPool(processes):
    # get a new pool
    
    return Pool(processes)


def getCondition():
    # get new condition lock
    
    return Condition()


def copy(data):
    # copy data in multiprocess
    return data


def cleanStore(store):
    # clean up a multiprocessing store
    
    if isinstance(store, managers.DictProxy):
        # manager mode
        for key in store.keys():
            store.pop(key)
    elif isinstance(store, basestring):
        # file mode
        for item in datamanager.listFiles(store, 'output-*'):
            datamanager.deleteFile(item)
    else:
        raise ValueError, "Unsupported storage mode."


def loadStore(store, taskid):
    # load from a multiprocessing store
    
    if isinstance(store, managers.DictProxy):
        # manager mode
        return store[taskid]
    elif isinstance(store, basestring):
        # file mode
        try:
            return datamanager.skLoad(os.path.join(store, 'output-%s' % str(taskid)))
        except IOError, e:
            if e.errno == errno.ENOENT:
                # # errno.ENOENT => no such file or directory
                # raise KeyError to make it compatible with dict manager exception
                raise KeyError("File with key %s not found." % str(taskid))
            else:
                raise
    else:
        raise ValueError, "Unsupported storage mode."


class DoWork(object):
    
    def __init__(self, workQueue, store, timeout, loggerDict, lock=None):
        # self things
        self.workQueue = workQueue
        self.store = store
        self.timeout = timeout
        self.logger = misc.getLogger(**loggerDict)
        self.pid = os.getpid()
        self.lock = lock
        
        # determine mode
        if isinstance(store, managers.DictProxy):
            # manager mode
            self.io = self._manager
        elif isinstance(store, basestring):
            # file mode
            self.io = self._file
        else:
            raise ValueError, "Unsupported storage mode."
    
    def _manager(self, taskid, data):
        self.store[taskid] = data
    
    def _file(self, taskid, data):
        datamanager.skStore(os.path.join(self.store, 'output-%s' % str(taskid)), data)
    
    def go(self):
        self.logger.info("Starting Process %d, waiting for tasks ..." % self.pid)
        while True:
            try:
                # get a task
                q = self.workQueue.get(timeout=self.timeout)
                function = q['function']
                data = q['data']
                parameters = q['parameters']
                if self.lock:
                    parameters.update({'lock': self.lock})
                taskid = q['taskid']
                
                self.logger.info("Processing task %s. with function %s." % (str(taskid), function.__module__ + '.' + function.__name__))
                
                # execute the task
                try:
                    output = function(data, **parameters)
                except Exception, e:
                    self.logger.error("Error running function.")
                    self.logger.exception(e)
                else:
                    # store the results
                    self.io(taskid, output)
                
            except Empty:
                self.logger.info("Empty queue exception.")
                break
            except Exception:
                self.logger.error(traceback.format_exc())
        self.logger.info("Terminating process.")
        return None


class DoWork_File(DoWork):
    
    def go(self):
        self.logger.info("Starting Process %d, waiting for tasks ..." % self.pid)
        while True:
            try:
                # get a task
                q = self.workQueue.get(timeout=self.timeout)
                function = q['function']
                filePath = q['filePath']
                parameters = q['parameters']
                if self.lock:
                    parameters.update({'lock': self.lock})
                taskid = q['taskid']
                
                self.logger.info("Processing task %s. with function %s." % (str(taskid), function.__module__ + '.' + function.__name__))
                
                # load the data from file
                data = datamanager.skLoad(os.path.join(filePath, 'output-%s' % str(taskid)))
                
                # execute the task
                try:
                    output = function(data, **parameters)
                except Exception, e:
                    self.logger.error("Error running function.")
                    self.logger.exception(e)
                    print e
                else:
                    # store the results
                    self.io(taskid, output)
                
            except Empty:
                self.logger.info("Empty queue exception.")
                break
            except Exception:
                self.logger.error(traceback.format_exc())
        self.logger.info("Terminating process.")
        return None


class DoWork_Clf(DoWork):
    
    def go(self):
        self.logger.info("Starting Process %d, waiting for tasks ..." % self.pid)
        while True:
            try:
                # get a task
                q = self.workQueue.get(timeout=self.timeout)
                
                clf = q['classifier']
                data = q['data']
                th = q['threshold']
                subjects = q['subjects']
                taskid = q['taskid']
                parameters = q['parameters']
                if self.lock:
                    parameters.update({'lock': self.lock})
                
                self.logger.info("Processing task %s. with function %s." % (str(taskid), clf.authenticate))
                
                # execute the task
                try:
                    output = {'identification': clf._identify(data, th, **parameters)}
                    auth = []
                    for subject_tst in subjects:
                        auth.append(clf.authenticate(data, subject_tst, th, **parameters))
                    output['authentication'] = np.array(auth)
                except Exception, e:
                    self.logger.error("Error running function.")
                    self.logger.exception(e)
                    print e
                else:
                    # store the results
                    self.io(taskid, output)
                
            except Empty:
                self.logger.info("Empty queue exception.")
                break
            except Exception:
                self.logger.error(traceback.format_exc())
        self.logger.info("Terminating process.")
        return None


class DoWork_Distances(DoWork):
    
    def go(self):
        self.logger.info("Starting Process %d, waiting for tasks ..." % self.pid)
        while True:
            try:
                # get a task
                q = self.workQueue.get(timeout=self.timeout)
                
                dfcn = q['function']
                testData = q['testData']
                trainData = q['trainData']
                taskid = q['taskid']
                
                self.logger.info("Processing task %s. with function %s." % (str(taskid), misc.wavedistance))
                
                # execute the task
                try:
                    clabels = taskid * np.ones((len(testData), len(trainData)), dtype='int')
                    dists = []
                    for obs in testData:
                        dists.append(misc.wavedistance(obs, trainData, dfcn))
                    output = {'distances': np.array(dists), 'labels': clabels}
                except Exception, e:
                    self.logger.error("Error running function.")
                    self.logger.exception(e)
                    print e
                else:
                    # store the results
                    self.io(taskid, output)
                
            except Empty:
                self.logger.info("Empty queue exception.")
                break
            except Exception:
                self.logger.error(traceback.format_exc())
        self.logger.info("Terminating process.")
        return None


def workerProcess(workClass, workQueue, store, timeout, loggerDict, randomSeed, lock=None):
    # instantiate a worker class and run it
    
    # set the random generator
    # reload(config)
    config.setRandomSeed(randomSeed)
    
    # instantiate
    worker = workClass(workQueue, store, timeout, loggerDict, lock=lock)
    
    # run
    worker.go()
    
    return None


def runMultiprocess(workQueue, store, mode='data', numberProcesses=None,
                    lock=None, timeout=None, log2file=False):
    # run in multiprocess
    
    # select the worker function
    if mode == 'data':
        workClass = DoWork
    elif mode == 'file':
        workClass = DoWork_File
    elif mode == 'clf':
        workClass = DoWork_Clf
    elif mode == 'distances':
        workClass = DoWork_Distances
    else:
        raise ValueError, "Unknown worker mode %s." % mode
    
    # determine number of processors
    if numberProcesses is None:
        numberProcesses = config.numberProcesses
    
    # determine queue timeout
    if timeout is None:
        timeout = config.queueTimeOut
    
    # get a logger for each process
    loggers = []
    if log2file:
        for i in xrange(numberProcesses):
            loggers.append({
                            'name': 'Process %d' % i,
                            'logPath': config.getFilePath(('log', 'Process %d.log' % i)),
                            'level': 'debug',
                            })
    else:
        for i in xrange(numberProcesses):
            loggers.append({
                            'name': 'Process %d' % i,
                            'logPath': None,
                            'level': 'debug',
                            })
    
    # random seed
    seed = config.getRandomSeed()
    if seed is None:
        seed = 0
    
    # create N processes and associate each them with the work_queue, do_work function and the logger
    processes = [Process(target=workerProcess, args=(workClass, workQueue, store, timeout, loggers[i], seed+i, lock)) for i in range(numberProcesses)]
    
    # launch processes
    for p in processes: p.start()
    # wait for processes to finish
    for p in processes: p.join()
    for p in processes: p.terminate()


def randomTester(data):
    # test random number generation
    
    out = (config.random.randn(), time.time())
    
    return out

