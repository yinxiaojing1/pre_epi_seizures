"""
.. module:: classifiers
   :platform: Unix, Windows
   :synopsis: This module provides various functions to...

.. moduleauthor:: Filipe Canento, Carlos Carreiras


"""

# Notes:
# include libSVM
# Classifier should have:
#     input n observations x m features
#     output n observations x 1 (class)

# Imports
# built-in
from itertools import izip
import os
import shutil
# import traceback
import warnings

# 3rd party
from bidict import bidict
import numpy
# import scipy
import scipy.spatial.distance as dist
# import pylab
from sklearn import svm as sksvm

# BiometricsPyKit
import config
import rules
from Cloud import parallel
from cluster import clusteringCombination as cc
from datamanager import datamanager, parted
from dimreduction import dimreduction
from evaluation import evaluation
from featureextraction import featureextraction
from misc import misc
from wavelets import wavelets

# from demo import CP77py



def selector(method):
    """
    Selector for the classifier functions and methods.
    
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
    
    if method == 'KNN':
        fcn = KNN
    elif method == 'SVM':
        fcn = SVM
    elif method == 'LeanSVM':
        fcn = LeanSVM
    elif method == 'Fisher':
        fcn = fisher
    elif method == 'Odinaka':
        fcn = Odinaka
    elif method == 'Agrafioti':
        fcn = Agrafioti
    elif method == 'Dissimilarity':
        fcn = Dissimilarity
    elif method == 'DissimilaritySimple':
        fcn = DissimilaritySimple
    elif method == 'DissimilarityMore':
        fcn = DissimilarityMore
    else:
        raise TypeError, "Method %s not implemented." % method
    
    return fcn

def eerIndex(method):
    if method in ['KNN', 'Agrafioti', 'Dissimilarity', 'DissimilaritySimple', 'DissimilarityMore']:
        return 0
    elif method in ['SVM']:
        return -1
    else:
        raise TypeError, "Method %s not implemented." % method


class UnknownSubjectError(Exception):
    # Exception raised when the subject is unknown
    
    def __init__(self, subject):
        # set the subject
        self.subject = subject
    
    def __str__(self):
        return str("Subject %s is not recognized." % str(self.subject))


class UntrainedError(Exception):
    # Exception raised when the classifier is untrained
    
    def __str__(self):
        return str("The classifier is not trained.")


class EmptyCombination(Exception):
    # Exception raised when the the combination method is called on an empty array
    
    def __str__(self):
        return str("Combination of empty array.")


class SubjectDict(bidict):
    # return default values for KeyError
    
    def __getitem__(self, keyorslice):
        # based on the bidict source
        try:
            start, stop, step = keyorslice.start, keyorslice.stop, keyorslice.step
        except AttributeError:
            # keyorslice is a key, e.g. b[key]
            try:
                return self._fwd[keyorslice]
            except KeyError:
                return -1

        # keyorslice is a slice
        if (not ((start is None) ^ (stop is None))) or step is not None:
            raise TypeError('Slice must specify only either start or stop')

        if start is not None: # forward lookup (by key), e.g. b[key:]
            try:
                return self._fwd[start]
            except KeyError:
                return -1

        # inverse lookup (by val), e.g. b[:val]
        assert stop is not None
        try:
            return self._bwd[stop]
        except KeyError:
            return ''


class ClassifierDict(object):
    """
    Class to handle hierarchical subject - label - classifier disambiguation.
    """
    
    def __init__(self):
        # init
        
        self._nbClfs = 0
        self._clf2label = {-1: -1}
        self._label2clf = {-1: set([-1])}
        self._labelWeights = {-1: 1.}
    
    def getClassifiers(self, label):
        # get classifiers for given label
        
        return list(self._label2clf[label])
    
    def getLabel(self, clf):
        # get label for given classifier
        
        return self._clf2label[clf]
    
    def getLabelWeight(self, label):
        # get label weight
        
        return self._labelWeights[label]
    
    def getPairWeight(self, pair):
        # get weight for classifier pair
        
        p1 = self.getLabelWeight(self.getLabel(pair[0]))
        p2 = self.getLabelWeight(self.getLabel(pair[1]))
        
        return p1 * p2
    
    def assign(self, n, label):
        # assign n classifiers to label
        
        try:
            aux = self._label2clf[label]
        except KeyError:
            aux = self._label2clf[label] = set()
        
        for _ in xrange(n):
            self._clf2label[self._nbClfs] = label
            aux.add(self._nbClfs)
            self._nbClfs += 1
        
        self._labelWeights[label] = 1. / float(len(aux))
        
        return self.getClassifiers(label)


class BaseClassifier(object):
    """
    Base Classifier.
    """
    
    NAME = 'BaseClassifier'
    EXT = '.hdf5'
    EER_IDX = 0
    
    def __init__(self, io=None):
        # generic self things
        self._reset()
        
        # choose IO mode
        if io is None:
            # memory-based IO mode
            self.iomode = 'mem'
            self.iofile = {}
        elif isinstance(io, basestring):
            # file-based IO mode, path string as argument
            self.iomode = 'file'
            self.iopath = io
            self.iofile = os.path.join(self.iopath, self.NAME + self.EXT)
            # verify path exists
            if not os.path.exists(io):
                os.makedirs(io)
            # prepare io
            self._prepareIO()
        elif isinstance(io, tuple) or isinstance(io, list):
            # file-based IO mode, tuple of strings as argument
            self.iomode = 'file'
            self.iopath = config.getPath(io)
            self.iofile = os.path.join(self.iopath, self.NAME + self.EXT)
            # prepare io
            self._prepareIO()
        else:
            raise NotImplementedError, "IO mode unknown or not yet implemented."
    
    def _reset(self):
        # reset the classifier
        self.subject2label = SubjectDict()
        self.nbSubjects = 0
        self.is_trained = False
        self.thresholds = {}
        self._autoThresholds = None
    
    def _snapshot(self):
        # snapshot of classifier structures
        subject2label = self.subject2label
        nbSubjects = self.nbSubjects
        
        return subject2label, nbSubjects
    
    def _rebrand(self, name):
        # change io file name
        
        if self.iomode == 'file':
            new = self.iofile.replace(self.NAME, name)
            os.rename(self.iofile, new)
            self.iofile = new
        
        self.NAME = name
    
    def _update_fileIO(self, path):
        # update file IO to new path
        
        if self.iomode == 'file':
            self.iopath = path
            self.iofile = os.path.join(path, self.NAME + self.EXT)
    
    def _prepareIO(self):
        # create dirs, initialize files
        
        datamanager.allocH5(self.iofile)
    
    def io_load(self, label):
        # load data
        
        if self.iomode == 'file':
            return self._fileIO_load(label)
        elif self.iomode == 'mem':
            return self._memIO_load(label)
    
    def io_save(self, label, data):
        # save data
        
        if self.iomode == 'file':
            self._fileIO_save(label, data)
        elif self.iomode == 'mem':
            self._memIO_save(label, data)
    
    def _fileIO_load(self, label):
        # load label from file
        data = datamanager.h5Load(self.iofile, label)
        return data
    
    def _fileIO_save(self, label, data):
        # save data with label to file
        datamanager.h5Store(self.iofile, label, data)
    
    def _memIO_load(self, label):
        # load label from memory
        return self.iofile[label]
    
    def _memIO_save(self, label, data):
        # save data with label to memory
        self.iofile[label] = data
    
    def fileIterator(self):
        # iterator for the classifier files
        
        yield self.NAME + self.EXT
    
    def dirIterator(self):
        # iterator for the directories
        
        return iter([])
    
    def save(self, dstPath):
        # save the classifier to the path
        
        # classifier files
        if self.iomode == 'file':
            tmpPath = os.path.join(self.iopath, 'clf-tmp')
            if not os.path.exists(tmpPath):
                os.makedirs(tmpPath)
            
            # dirs
            for d in self.dirIterator():
                path = os.path.join(tmpPath, d)
                if not os.path.exists(path):
                    os.makedirs(path)
            
            # files
            for f in self.fileIterator():
                src = os.path.join(self.iopath, f)
                dst = os.path.join(tmpPath, f)
                try:
                    shutil.copy(src, dst)
                except IOError:
                    pass
        else:
            tmpPath = os.path.abspath(os.path.expanduser('~/clf-tmp'))
            if not os.path.exists(tmpPath):
                os.makedirs(tmpPath)
        
        # save classifier instance to temp file
        datamanager.skStore(os.path.join(tmpPath, 'clfInstance.p'), self)
        
        # save to zip archive
        dstPath = os.path.join(dstPath, self.NAME)
        datamanager.zipArchiveStore(tmpPath, dstPath)
        
        # remove temp dir
        shutil.rmtree(tmpPath, ignore_errors=True)
        
        return dstPath
    
    @classmethod
    def load(cls, srcPath, dstPath=None):
        # load a classifier instance from a file
        # do not include the extension in the path
        
        if dstPath is None:
            dstPath, _ = os.path.split(srcPath)
        
        # unzip files
        datamanager.zipArchiveLoad(srcPath, dstPath)
        
        # load classifier
        tmpPath = os.path.join(dstPath, 'clfInstance.p')
        clf = datamanager.skLoad(tmpPath)
        
        # classifier files
        clf._update_fileIO(dstPath)
        
        # remove temp file
        os.remove(tmpPath)
        
        if not isinstance(clf, cls):
            raise TypeError, "Mismatch between target class and loaded file."
        
        return clf
    
    def checkSubject(self, subject):
        # check if subject is enrolled
        
        if self.is_trained:
            return subject in self.subject2label
        
        return False
    
    def listSubjects(self):
        # list the enrolled subjects
        
        subjects = [self.subject2label[:i] for i in xrange(self.nbSubjects)]
        
        return subjects
    
    def _prepareData(self, data):
        # prepare date
        ### user
        
        return data
    
    def _updateStrategy(self, oldData, newData):
        # update the training data of a class when new data is available
        
        return newData
    
    def authThreshold(self, subject, ready=False):
        # get the user threshold (authentication)
        
        if not ready:
            aux = subject
            subject = self.subject2label[subject]
            if subject == -1:
                raise UnknownSubjectError(aux)
        
        return self.thresholds[subject]['auth']
    
    def setAuthThreshold(self, subject, threshold, ready=False):
        # set the user threshold (authentication)
        
        if not ready:
            aux = subject
            subject = self.subject2label[subject]
            if subject == -1:
                raise UnknownSubjectError(aux)
        try:
            self.thresholds[subject]['auth'] = threshold
        except KeyError:
            self.thresholds[subject] = {'auth': threshold, 'id': None}
    
    def idThreshold(self, subject, ready=False):
        # get the user threshold (identification)
        
        if not ready:
            aux = subject
            subject = self.subject2label[subject]
            if subject == -1:
                raise UnknownSubjectError(aux)
        
        return self.thresholds[subject]['id']
    
    def setIdThreshold(self, subject, threshold, ready=False):
        # set the user threshold (identification)
        
        if not ready:
            aux = subject
            subject = self.subject2label[subject]
            if subject == -1:
                raise UnknownSubjectError(aux)
        try:
            self.thresholds[subject]['id'] = threshold
        except KeyError:
            self.thresholds[subject] = {'auth': None, 'id': threshold}
    
    def autoRejectionThresholds(self):
        # generate thresholds automatically
        ### user
        
        if self._autoThresholds is not None:
            return self._autoThresholds
        
        ths = numpy.array([])
        
        return ths
    
    def _subThrIterator(self, overwrite):
        # iterate over the subjects in order to update the thresholds
        
        if overwrite:
            for i in xrange(self.nbSubjects):
                yield i
        else:
            for i in xrange(self.nbSubjects):
                try:
                    _ = self.authThreshold(i, ready=True)
                except KeyError:
                    yield i
    
    def updateThresholds(self, overwrite=False, fraction=1.):
        # update the user thresholds based on the enrolled data
        
        ths = self.autoRejectionThresholds()
        
        # gather data to test
        data = {}
        for lbl in self._subThrIterator(overwrite):
            subject = self.subject2label[:lbl]
            
            # select a random fraction of the training data
            aux = self.io_load(lbl)
            indx = range(len(aux))
            use, _ = parted.randomFraction(indx, 0, fraction)
            
            data[subject] = aux[use]
        
        # evaluate classifier
        if len(data.keys()) > 42:
            out = self.evaluate(data, ths)
        else:
            out = self.seqEvaluate(data, ths)
        
        # choose thresholds at EER
        for lbl in self._subThrIterator(overwrite):
            subject = self.subject2label[:lbl]
            
            EER_auth = out['assessment']['subject'][subject]['authentication']['rates']['EER']
            self.setAuthThreshold(lbl, EER_auth[self.EER_IDX, 0], ready=True)
             
            EER_id = out['assessment']['subject'][subject]['identification']['rates']['EER']
            self.setIdThreshold(lbl, EER_id[self.EER_IDX, 0], ready=True)
    
    def updateThresholds_old(self, overwrite=False, N=1):
        # update the user thresholds based on the enrolled data
        
        ths = self.autoRejectionThresholds()
        
        labels = range(self.nbSubjects)
        if self.nbSubjects > 1:
            prob = 1. / (self.nbSubjects - 1)
            p = prob * numpy.ones(self.nbSubjects, dtype='float')
        else:
            p = None
        
        for lbl in self._subThrIterator(overwrite):
            subject = self.subject2label[:lbl]
            # choose random subjects for authentication
            test_lbl = [lbl]
            try:
                p[lbl] = 0
            except TypeError:
                pass
            else:
                test_lbl.extend(config.random.choice(labels, p=p, size=N))
                p[lbl] = prob
            
            # load data
            data = {self.subject2label[:item]: self.io_load(item) for item in test_lbl}
             
            # evaluate classifier
            out = self.evaluate(data, ths)
             
            # choose threshold at EER
            EER_auth = out['assessment']['subject'][subject]['authentication']['rates']['EER']
            self.setAuthThreshold(lbl, EER_auth[self.EER_IDX, 0], ready=True)
             
            EER_id = out['assessment']['subject'][subject]['identification']['rates']['EER']
            self.setIdThreshold(lbl, EER_id[self.EER_IDX, 0], ready=True)
    
    def train(self, data=None, updateThresholds=True):
        # data is {subject: features (array)}
        ### user
        
        # check inputs
        if data is None:
            raise TypeError, "Please provide input data."
        
        # check if classifier was already trained
        if self.is_trained:
            # this is a retrain
            self.re_train(data)
        else:
            # get subjects
            subjects = data.keys()
            self.nbSubjects = len(subjects)
            
            # determine classifier mode
            if self.nbSubjects == 0:
                raise ValueError, "Please provide input data - empty dict."
            else:
                for i in xrange(self.nbSubjects):
                    # build dicts
                    sub = subjects[i]
                    self.subject2label[sub] = i
                    
                    # save data
                    self.io_save(i, data[sub])
                    
                    # prepare data
                    _ = self._prepareData(data[sub])
                    
                    ### user
        
        # train flag
        self.is_trained = True
        
        if updateThresholds:
            # update thresholds
            self.updateThresholds()
    
    def re_train(self, data):
        # data is {subject: features (array)}
        ### user
        
        for sub in data.iterkeys():
            if sub in self.subject2label:
                # existing subject
                if data[sub] is not None:
                    # change subject's data
                    label = self.subject2label[sub]
                    
                    # update templates
                    aux = self._updateStrategy(self.io_load(label), data[sub])
                    
                    # save data
                    self.io_save(label, aux)
                    
                    # prepare data
                    _ = self._prepareData(aux)
                    
                    ### user
                    
                    
                else:
                    # delete subject
                    if self.nbSubjects == 1:
                        # reset classifier to untrained
                        self._reset()
                    else:
                        # reorder the labels/subjects and models dicts
                        subject2label, nbSubjects = self._snapshot()
                        self._reset()
                        
                        label = subject2label[sub]
                        clabels = numpy.setdiff1d(numpy.unique(subject2label.values()),
                                                  [label], assume_unique=True)
                        self.nbSubjects = nbSubjects - 1
                        
                        i = 0
                        for ii in xrange(len(clabels)):
                            # build dicts
                            sub = subject2label[:clabels[ii]]
                            self.subject2label[sub] = i
                            
                            # move data
                            self.io_save(i, self.io_load(clabels[ii]))
                            
                            ### user
                            
                            # update i
                            i += 1
            else:
                # new subject
                # add to dicts
                label = self.nbSubjects
                self.subject2label[sub] = label
                
                # save data
                self.io_save(label, data[sub])
                
                # prepare data
                _ = self._prepareData(data[sub])
                
                ### user
                
                
                # increment number of subjects
                self.nbSubjects += 1
    
    def authenticate(self, data, subject, threshold=None, ready=False, labels=False, **kwargs):
        # data is a list of feature vectors, allegedly belonging to the given subject
        ### user
        
        # check train state
        if not self.is_trained:
            raise UntrainedError
        
        # translate subject ID to class label
        label = self.subject2label[subject]
        if label == -1:
            raise UnknownSubjectError(subject)
        
        if threshold is None:
            # get user-tuned threshold
            threshold = self.authThreshold(label, ready=True)
        
        # prepare data
        if not ready:
            _ = self._prepareData(data)
        else:
            _ = data
        
        # outputs
        decision = []
        prediction = []
        
        ### user
        
        # convert to numpy
        decision = numpy.array(decision)
        
        if labels:
            # translate class label to subject ID
            subPrediction = [self.subject2label[:item] for item in prediction]
            return decision, subPrediction
        else:
            return decision
    
    def _identify(self, data, threshold=None, ready=False):
        # data is list of feature vectors
        ### user
        
        # check train state
        if not self.is_trained:
            raise UntrainedError
        
        # threshold
        if threshold is None:
            _ = lambda label: self.idThreshold(label, ready=True)
        else:
            _ = lambda label: threshold
        
        # prepare data
        if not ready:
            _ = self._prepareData(data)
        else:
            _ = data
        
        # outputs
        labels = []
        
        ### user
        
        return numpy.array(labels)
    
    def identify(self, data, threshold=None, ready=False, **kwargs):
        # data is list of feature vectors
        
        labels = self._identify(data=data, threshold=threshold, ready=ready, **kwargs)
        
        # translate class labels to subject IDs
        subjects = [self.subject2label[:item] for item in labels]
        
        return subjects
    
    def evaluate(self, data, rejection_thresholds=None, dstPath=None, log2file=False):
        """
        Assess the performance of the classifier in both biometric scenarios: authentication and identification.
        
        Workflow:
            For each test subject and for each threshold, create a multiprocessing task that tests authentication and identification;
            Authentication results stored in a 3 dimensional array of booleans, shape = (N thresholds, M subjects, K samples);
            Identification results stored in a 2 dimensional array, shape = (N thresholds, K samples);
            Subject and global statistics are then computed by evaluation.assessClassification.
        
        Kwargs:
            data (dict): Dictionary holding the testing samples for each subject.
            
            rejection_thresholds (array): Thresholds used to compute the ROCs.
            
            dstPath (string): Path for multiprocessing.
            
            log2file (bool): Flag to control the use of logging in multiprocessing.
        
        Kwrvals:
            classification (dict): Results of the classification.
            
            assessment (dict): Biometric statistics.
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        # data is {subject: features (array)}
        
        # check train state
        if not self.is_trained:
            raise UntrainedError
        
        # choose thresholds
        if rejection_thresholds is None:
            rejection_thresholds = self.autoRejectionThresholds()
        nth = len(rejection_thresholds)
        
        # choose store
        if dstPath is None:
            store = parallel.getDictManager()
        else:
            store = dstPath
        
        subjects = data.keys()
        results = {'subjectList': subjects,
                   'subjectDict': self.subject2label,
                   }
        for subject in subjects:
            # prepare data
            aux = self._prepareData(data[subject])
            
            # test
            auth_res = []
            id_res = []
            workQ = parallel.getQueue()
            for i in xrange(nth):
                workQ.put({
                           'classifier': self,
                           'data': aux,
                           'threshold': rejection_thresholds[i],
                           'subjects': subjects,
                           'parameters': {'ready': True},
                           'taskid': i,
                           })
            
            # run in multiprocessing
            parallel.runMultiprocess(workQ, store, mode='clf', log2file=log2file)
            
            # load from files
            for i in xrange(nth):
                res = parallel.loadStore(store, i)
                auth_res.append(res['authentication'])
                id_res.append(res['identification'])
            
            # clean up store
            parallel.cleanStore(store)
            
            auth_res = numpy.array(auth_res)
            id_res = numpy.array(id_res)
            results[subject] = {'authentication': auth_res,
                                'identification': id_res,
                                }
        
        # assess classification results
        assess = evaluation.assessClassification(results, rejection_thresholds,
                                                 dstPath=dstPath, log2file=log2file)
        
        # final output
        output = {'classification': results,
                  'assessment': assess,
                  }
        
        return output
    
    def seqEvaluate(self, data, rejection_thresholds=None, dstPath=None, log2file=False):
        """
        Assess the performance of the classifier in both biometric scenarios: authentication and identification.
        
        Workflow:
            For each test subject and for each threshold, test authentication and identification;
            Authentication results stored in a 3 dimensional array of booleans, shape = (N thresholds, M subjects, K samples);
            Identification results stored in a 2 dimensional array, shape = (N thresholds, K samples);
            Subject and global statistics are then computed by evaluation.assessClassification.
        
        Kwargs:
            data (dict): Dictionary holding the testing samples for each subject.
            
            rejection_thresholds (array): Thresholds used to compute the ROCs.
            
            dstPath (string): Path for multiprocessing.
            
            log2file (bool): Flag to control the use of logging in multiprocessing.
        
        Kwrvals:
            classification (dict): Results of the classification.
            
            assessment (dict): Biometric statistics.
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        # data is {subject: features (array)}
        
        # check train state
        if not self.is_trained:
            raise UntrainedError
        
        # choose thresholds
        if rejection_thresholds is None:
            rejection_thresholds = self.autoRejectionThresholds()
        nth = len(rejection_thresholds)
        
        subjects = data.keys()
        results = {'subjectList': subjects,
                   'subjectDict': self.subject2label,
                   }
        
        for subject in subjects:
            # prepare data
            aux = self._prepareData(data[subject])
            
            # test
            auth_res = []
            id_res = []
            for i in xrange(nth):
                th = rejection_thresholds[i]
                
                auth = []
                for subject_tst in subjects:
                    auth.append(self.authenticate(aux, subject_tst, th, ready=True))
                auth_res.append(numpy.array(auth))
                
                id_res.append(self._identify(aux, th, ready=True))
            
            auth_res = numpy.array(auth_res)
            id_res = numpy.array(id_res)
            results[subject] = {'authentication': auth_res,
                                'identification': id_res,
                                }
        
        # assess classification results
        assess = evaluation.assessClassification(results, rejection_thresholds,
                                                 dstPath=dstPath, log2file=log2file)
        
        # final output
        output = {'classification': results,
                  'assessment': assess,
                  }
        
        return output


class SVM(BaseClassifier):
    """
    SVM classifier.
    """
    
    NAME = 'svmClassifier'
    EXT = '.hdf5'
    EER_IDX = -1
    
    def __init__(self, io=None, **kwargs):
        """
        Wrapper for the scikit-learn SVM classifier, using OneClassSVM, SVC or LinearSVC,
        depending on usage case.
        
        sklearn.svm.SVC kwargs:
            C (float): Penalty parameter C of the error term. Default=1.0
            
            kernel (string): The kernel to be used ('linear', 'poly', 'rbf', 'sigmoid'). Default='rbf'
            
            degree (int): Degree of kernel function; only significant for 'poly' and 'sigmoid'. Default=3
            
            gamma (float): Kernel coefficient for 'rbf' and 'poly'. Default=0.0
            
            coef0 (float): Independent term in kernel function; it is only significant in 'poly' and 'sigmoid'. Default=0.0
            
            probability (bool): Whether to enable probability estimates. Default=False
            
            shrinking (bool): Whether to use the shrinking heuristic. Default=True
            
            tol (float): Tolerance for stopping criterion. Default=1e-3
            
            cache_size (float): Specify the size of the kernel cache (in MB).
            
            class_weight (dict, 'auto'): Weights for each class; the 'auto' mode automatically adjusts the
                                         weights inversely proportional to class frequencies.
            
            verbose (bool): Enable verbose output. Default=False
            
            max_iter (int): Hard limit on iterations within solver, or -1 for no limit. Default=-1
        """
        
        # run parent __init__
        super(SVM, self).__init__(io=io)
        
        try:
            self.kernel = kwargs['kernel']
        except KeyError:
            self.kernel = 'linear'
        else:
            if self.kernel == 'linear':
                _ = kwargs.pop('kernel')
        
        try:
            # let me deal with the class weights
            _ = kwargs.pop('class_weight')
        except KeyError:
            pass
        
        # minimum threshold
        self.minThr = 10 * numpy.finfo('float').eps
        
        self.clf_kwargs = kwargs
    
    def _reset(self):
        # run parent reset
        super(SVM, self)._reset()
        
        self.models = {}
    
    def _snapshot(self):
        # run parent snapshot
        subject2label, nbSubjects = super(SVM, self)._snapshot()
        
        models = self.models
        
        return subject2label, nbSubjects, models
    
    def _prepareData(self, data):
        # prepare data
        X = []
        
        for tpl in data:
            # guarantee numpy
            tpl = numpy.array(tpl)
            # guarantee 1D vector
            tpl = tpl.flatten()
            
            X.append(tpl)
        
        X = numpy.array(X)
        
        return X, len(X)
    
    def _weights(self, n1, n2):
        # compute class weights (inversely proportional to number of samples in class)
        
        w = numpy.array([1./n1, 1./n2])
        w *= 2 / numpy.sum(w)
        weights = {-1: w[0], 1: w[1]}
        
        return weights
    
    def _getSingleClf(self, X):
        # instantiate and train a OneClassSVM classifier
        
        # instantiate and fit
        clf = sksvm.OneClassSVM(kernel='rbf', nu=0.1)
        clf.fit(X)
        
        # add to models
        self.models[(-1, 0)] = clf
    
    def _getLinearClf(self, X1, X2, n1, n2, label1, label2):
        # instantiate and train a LinearSVC classifier
        
        # prepare data to train
        X = numpy.concatenate((X1, X2), axis=0)
        Y = numpy.ones(n1 + n2)
        if label1 < label2:
            Y[:n1] = -1
            pair = (label1, label2)
        else:
            Y[n1:] = -1
            pair = (label2, label1)
        
        # class weights (inversely proportional to number of samples in class)
        weights = self._weights(n1, n2)
        
        # instantiate and fit
        clf = sksvm.LinearSVC(class_weight=weights, **self.clf_kwargs)
        clf.fit(X, Y)
        
        # add to models
        self.models[pair] = clf
    
    def _getKernelClf(self, X1, X2, n1, n2, label1, label2):
        # instantiate and train an SVC classifier
        
        # prepare data to train
        X = numpy.concatenate((X1, X2), axis=0)
        Y = numpy.ones(n1 + n2)
        if label1 < label2:
            Y[:n1] = -1
            pair = (label1, label2)
        else:
            Y[n1:] = -1
            pair = (label2, label1)
        
        # class weights (inversely proportional to number of samples in class)
        weights = self._weights(n1, n2)
        
        # instantiate and fit
        clf = sksvm.SVC(class_weight=weights, **self.clf_kwargs)
        clf.fit(X, Y)
        
        # add to models
        self.models[pair] = clf
    
    def _predict(self, pair, data):
        # get the classifier prediction
        
        # convert pair
        pair = self._convertPair(pair)
        
        aux = self.models[pair].predict(data)
        res = -1 * numpy.ones(aux.shape, dtype='int')
        res[aux < 0] = pair[0]
        res[aux > 0] = pair[1]
        
        return res
    
    def _convertPair(self, pair):
        # sort and convert list pair to tuple
        
        try:
            pair.sort()
        except AttributeError:
            return pair
        
        pair = tuple(pair)
        
        return pair
    
    def _updateStrategy(self, oldData, newData):
        # update the training data of a class when new data is available
        
        # concatenate old with new data
        out = numpy.concatenate([oldData, newData], axis=0)
        
        return out
    
    def _reorderModels(self, pairN, pairO, models):
        # reorder the models
        
        pairN = self._convertPair(pairN)
        pairO = self._convertPair(pairO)
        self.models[pairN] = models[pairO]
    
    def _removeModel(self, pair):
        # remove a model
        
        pair = self._convertPair(pair)
        m = self.models.pop(pair)
        del m
    
    def autoRejectionThresholds(self):
        # generate thresholds automatically
        
        if self._autoThresholds is not None:
            return self._autoThresholds
        
        # rejection thresholds to test
        self._autoThresholds = numpy.linspace(self.minThr, 1., 75)
        
        return self._autoThresholds
    
    def train(self, data=None, updateThresholds=True):
        # data is {subject: features (array)}
        
        # check inputs
        if data is None:
            raise TypeError, "Please provide input data."
        
        # check if classifier was already trained
        if self.is_trained:
            # this is a retrain
            self.re_train(data)
        else:
            # get subjects
            subjects = data.keys()
            self.nbSubjects = len(subjects)
            
            # determine classifier mode
            if self.nbSubjects == 0:
                raise ValueError, "Please provide input data - empty dict."
            elif self.nbSubjects == 1:
                # single class
                sub = subjects[0]
                self.subject2label[sub] = 0
                self.subject2label[""] = -1
                
                # prepare data
                X, _ = self._prepareData(data[sub])
                
                self._getSingleClf(X)
                
                # save data
                self.io_save(0, data[sub])
            else:
                # determine classifier mode
                if self.kernel == 'linear':
                    clff = self._getLinearClf
                else:
                    clff = self._getKernelClf
                
                # prepare data
                prepData = {}
                for i in xrange(self.nbSubjects):
                    # build dicts
                    sub = subjects[i]
                    self.subject2label[sub] = i
                    
                    # save data
                    self.io_save(i, data[sub])
                    
                    # prepare data
                    prepData[i] = self._prepareData(data[subjects[i]])
                
                # train
                for i in xrange(self.nbSubjects):
                    X1, n1 = prepData[i]
                    
                    for j in xrange(i + 1, self.nbSubjects):
                        X2, n2 = prepData[j]
                        
                        # build new model
                        clff(X1, X2, n1, n2, i, j)
        
        # train flag
        self.is_trained = True
        
        if updateThresholds:
            # update thresholds
            self.updateThresholds()
    
    def re_train(self, data):
        # data is {subject: features (array)}
        
        # determine classifier mode
        if self.kernel == 'linear':
            clff = self._getLinearClf
        else:
            clff = self._getKernelClf
        
        for sub in data.iterkeys():
            if sub in self.subject2label:
                # existing subject
                if data[sub] is not None:
                    # change subject's data
                    label = self.subject2label[sub]
                    
                    # update templates
                    aux = self._updateStrategy(self.io_load(label), data[sub])
                    
                    # save data
                    self.io_save(label, aux)
                    
                    # prepare data
                    X1, n1 = self._prepareData(aux)
                    
                    if self.nbSubjects == 1:
                        self._getSingleClf(X1)
                    else:
                        clabels = list(set(self.subject2label.values()) - set([label]))
                        for j in clabels:
                            # prepare data
                            X2, n2 = self._prepareData(self.io_load(j))
                            
                            # build new model
                            clff(X1, X2, n1, n2, label, j)
                else:
                    # delete subject
                    if self.nbSubjects == 1:
                        # reset classifier to untrained
                        self._reset()
                    elif self.nbSubjects == 2:
                        # reduce to OneClassSVM case
                        label = self.subject2label[sub]
                        aux = self.io_load(label)
                        self._reset()
                        
                        # prepare data
                        X, _ = self._prepareData(aux)
                        
                        self.subject2label[sub] = 0
                        self.subject2label[""] = -1
                        self._getSingleClf(X)
                        
                        # save data
                        self.io_save(0, aux)
                        
                    else:
                        # reorder the labels/subjects and models dicts
                        subject2label, nbSubjects, models = self._snapshot()
                        self._reset()
                        
                        label = subject2label[sub]
                        clabels = numpy.setdiff1d(numpy.unique(subject2label.values()),
                                                  [label], assume_unique=True)
                        self.nbSubjects = nbSubjects - 1
                        
                        i = 0
                        for ii in xrange(len(clabels)):
                            # build dicts
                            sub = subject2label[:clabels[ii]]
                            self.subject2label[sub] = i
                            
                            # move data
                            self.io_save(i, self.io_load(clabels[ii]))
                            
                            j = i + 1
                            for jj in xrange(ii + 1, len(clabels)):
                                # reorder models
                                self._reorderModels([i, j], [clabels[ii], clabels[jj]], models)
                                
                                # update j
                                j += 1
                            
                            # update i
                            i += 1
            else:
                # new subject
                # add to dicts
                label = self.nbSubjects
                self.subject2label[sub] = label
                
                if label == 1:
                    # remove the OneClassSVM model and label
                    self.subject2label.pop("")
                    self._removeModel([-1, 0])
                
                # save data
                self.io_save(label, data[sub])
                
                # prepare data
                X1, n1 = self._prepareData(data[sub])
                
                for j in xrange(self.nbSubjects):
                    # prepare data
                    X2, n2 = self._prepareData(self.io_load(j))
                    
                    # build new model
                    clff(X1, X2, n1, n2, label, j)
                
                # increment number of subjects
                self.nbSubjects += 1
    
    def authenticate(self, data, subject, threshold=None, ready=False, labels=False, **kwargs):
        # decision is made by majority
        # data is a list of feature vectors, allegedly belonging to the given subject
        
        # check train state
        if not self.is_trained:
            raise UntrainedError
        
        # translate subject ID to class label
        label = self.subject2label[subject]
        if label == -1:
            raise UnknownSubjectError(subject)
        
        if threshold is None:
            # get user-tuned threshold
            threshold = self.authThreshold(label, ready=True)
        
        # prepare data
        if not ready:
            X, n = self._prepareData(data)
        else:
            X, n = data
        
        # outputs
        decision = []
        prediction = []
        
        if self.nbSubjects == 1:
            prediction = self._predict([-1, 0], X)
            decision = [item == label for item in prediction]
        else:
            # target subjects
            target = list(set(self.subject2label.values()) - set([label]))
            
            res = []
            for t in target:
                res.append(self._predict([label, t], X))
            res = numpy.array(res)
             
            for i in xrange(n):
                # determine majority
                counts = numpy.bincount(res[:, i])
                predMax = counts.argmax()
                rate = float(counts[predMax]) / (self.nbSubjects - 1)
                if rate >= threshold:
                    # rate of agreement is >= threshold
                    decision.append(predMax == label)
                    prediction.append(predMax)
                else:
                    # rate of agreement is < threshold
                    decision.append(False)
                    prediction.append(predMax)
        
        # convert to numpy
        decision = numpy.array(decision)
        
        if labels:
            # translate class label to subject ID
            subPrediction = [self.subject2label[:item] for item in prediction]
            return decision, subPrediction
        else:
            return decision
    
    def _identify(self, data, threshold=None, ready=False, **kwargs):
        # data is list of feature vectors
        
        # check train state
        if not self.is_trained:
            raise UntrainedError
        
        # threshold
        if threshold is None:
            thrFcn = lambda label: self.idThreshold(label, ready=True)
        else:
            thrFcn = lambda label: threshold
        
        # prepare data
        if not ready:
            X, n = self._prepareData(data)
        else:
            X, n = data
        
        if self.nbSubjects == 1:
            labels = self._predict([-1, 0], X)
        else:
            # classify
            res = []
            for p in self.models.iterkeys():
                res.append(self._predict(p, X))
            res = numpy.array(res)
            
            # majority
            labels = []
            for i in xrange(n):
                counts = numpy.bincount(res[:, i])
                predMax = counts.argmax()
                rate = float(counts[predMax]) / (self.nbSubjects - 1)
                if rate >= thrFcn(predMax):
                    # accept
                    labels.append(predMax)
                else:
                    # reject
                    labels.append(-1)
        
        return numpy.array(labels)


class ClusterSVM(SVM):
    """
    Perform clustering on the training data.
    """
    
    NAME = 'clusterSvmClassifier'
    EXT = '.hdf5'
    EER_IDX = -1
    
    def __init__(self, io=None, nensemble=100, partMethod='average', **kwargs):
        # run parent __init__
        super(ClusterSVM, self).__init__(io=io, **kwargs)
        
        # clustering parameters
        self.nensemble = nensemble
        self.partMethod = partMethod
    
    def _reset(self):
        # run parent reset
        super(ClusterSVM, self)._reset()
        
        self.hierLabels = ClassifierDict()
        self.clfWeights = {}
    
    def _snapshot(self):
        # run parent snapshot
        subject2label, nbSubjects, models = super(ClusterSVM, self)._snapshot()
        
        hierLabels = self.hierLabels
        clfWeights = self.clfWeights
        
        return subject2label, nbSubjects, models, hierLabels, clfWeights
    
    def _cluster(self, data):
        # cluster the data
        
        # create ensemble
        ensemble = cc.ensembleCreationParallel(data, nensemble=self.nensemble, fun='KMeans')
        
        # generate coassoc
        coassocs = cc.coassociationCreation(ensemble, len(data))
        
        # extract partition
        partition = cc.consensusExtraction(coassocs=coassocs, k=0, method=self.partMethod)
        
        return partition['clusters']
    
    def _prepapreClusters(self, data):
        # cluster and partition data
        
        clusters = self._cluster(data)
        output = [self._prepareData(data[clusters[key]]) for key in clusters.iterkeys()]
        
        return output, len(output)
    
    def _getSingleClf(self, X):
        # instantiate and train a OneClassSVM classifier
        
        # instantiate and fit
        clf = sksvm.OneClassSVM(kernel='rbf', nu=0.1)
        clf.fit(X)
        
        # add to models
        self.models[(-1, 0)] = clf
        self.clfWeights[(-1, 0)] = 1.
    
    def _getLinearClf(self, X1, X2, n1, n2, label1, label2):
        # instantiate and train a LinearSVC classifier
        
        # prepare data to train
        X = numpy.concatenate((X1, X2), axis=0)
        Y = numpy.ones(n1 + n2)
        if label1 < label2:
            Y[:n1] = -1
            pair = (label1, label2)
        else:
            Y[n1:] = -1
            pair = (label2, label1)
        
        # class weights (inversely proportional to number of samples in class)
        weights = self._weights(n1, n2)
        
        # instantiate and fit
        clf = sksvm.LinearSVC(class_weight=weights, **self.clf_kwargs)
        clf.fit(X, Y)
        
        # add to models
        self.models[pair] = clf
        self.clfWeights[pair] = 1.
    
    def _getKernelClf(self, X1, X2, n1, n2, label1, label2):
        # instantiate and train an SVC classifier
        
        # prepare data to train
        X = numpy.concatenate((X1, X2), axis=0)
        Y = numpy.ones(n1 + n2)
        if label1 < label2:
            Y[:n1] = -1
            pair = (label1, label2)
        else:
            Y[n1:] = -1
            pair = (label2, label1)
        
        # class weights (inversely proportional to number of samples in class)
        weights = self._weights(n1, n2)
        
        # instantiate and fit
        clf = sksvm.SVC(class_weight=weights, **self.clf_kwargs)
        clf.fit(X, Y)
        
        # add to models
        self.models[pair] = clf
        self.clfWeights[pair] = 1.
    
    def _predict(self, pair, data):
        # get the classifier prediction
        
        # convert pair
        pair = self._convertPair(pair)
        
        aux = self.models[pair].predict(data)
        res = -1 * numpy.ones(aux.shape, dtype='int')
        res[aux < 0] = self.hierLabels.getLabel(pair[0])
        res[aux > 0] = self.hierLabels.getLabel(pair[1])
        
        # weight
        w = numpy.ones(aux.shape[0]) * self.hierLabels.getPairWeight(pair) * self.clfWeights[pair]
        
        return res, w
    
    def _iterTargets(self, label=None):
        # iterator over the target pairs for given label
        
        if label is None:
            for item in self.models.iterkeys():
                yield item
        else:
            targetSubjects = set(self.subject2label.values()) - set([label])
            labelClfs = self.hierLabels.getClassifiers(label)
            for tS in targetSubjects:
                targetClfs = self.hierLabels.getClassifiers(tS)
                for lC in labelClfs:
                    for tC in targetClfs:
                        yield [lC, tC]
    
    def train(self, data=None, updateThresholds=True):
        # data is {subject: features (array)}
        
        # check inputs
        if data is None:
            raise TypeError, "Please provide input data."
        
        # check if classifier was already trained
        if self.is_trained:
            # this is a retrain
            self.re_train(data)
        else:
            # get subjects
            subjects = data.keys()
            self.nbSubjects = len(subjects)
            
            # determine classifier mode
            if self.nbSubjects == 0:
                raise ValueError, "Please provide input data - empty dict."
            elif self.nbSubjects == 1:
                # single class
                sub = subjects[0]
                
                # prepare data
                X, _ = self._prepareData(data[sub])
                
                # get classifier
                self._getSingleClf(X)
                
                # label conversion
                self.subject2label[sub] = 0
                self.hierLabels.assign(1, -1)
                self.hierLabels.assign(1, 0)
                
                # save data
                self.io_save(0, data[sub])
            else:
                # determine classifier mode
                if self.kernel == 'linear':
                    clff = self._getLinearClf
                else:
                    clff = self._getKernelClf
                
                # prepare data
                self.prepData = prepData = {}
                self.prepClfs = prepClfs = {}
                for i in xrange(self.nbSubjects):
                    sub = subjects[i]
                    
                    # cluster and prepare data
                    prepData[i], nc = self._prepapreClusters(data[sub])
                    
                    # save data
                    self.io_save(i, data[sub])
                    
                    # label conversion
                    self.subject2label[sub] = i
                    prepClfs[i] = self.hierLabels.assign(nc, i)
                
                # train
                for i in xrange(self.nbSubjects):
                    D1 = prepData[i]
                    C1 = prepClfs[i]
                    
                    for j in xrange(i + 1, self.nbSubjects):
                        D2 = prepData[j]
                        C2 = prepClfs[j]
                        
                        for it, ii in enumerate(C1):
                            X1, n1 = D1[it]
                            
                            for jt, jj in enumerate(C2):
                                X2, n2 = D2[jt]
                                
                                # build new model
                                clff(X1, X2, n1, n2, ii, jj)
        
        # train flag
        self.is_trained = True
        
        if updateThresholds:
            # update thresholds
            self.updateThresholds()
    
    def re_train(self, data):
        # data is {subject: features (array)}
        
        raise NotImplementedError('Re-train not yet implemented.')
        
        # determine classifier mode
        if self.kernel == 'linear':
            clff = self._getLinearClf
        else:
            clff = self._getKernelClf
        
        for sub in data.iterkeys():
            if sub in self.subject2label:
                # existing subject
                if data[sub] is not None:
                    # change subject's data
                    label = self.subject2label[sub]
                    
                    # update templates
                    aux = self._updateStrategy(self.io_load(label), data[sub])
                    
                    # save data
                    self.io_save(label, aux)
                    
                    # prepare data
                    X1, n1 = self._prepareData(aux)
                    
                    if self.nbSubjects == 1:
                        self._getSingleClf(X1)
                    else:
                        clabels = list(set(self.subject2label.values()) - set([label]))
                        for j in clabels:
                            # prepare data
                            X2, n2 = self._prepareData(self.io_load(j))
                            
                            # build new model
                            clff(X1, X2, n1, n2, label, j)
                else:
                    # delete subject
                    if self.nbSubjects == 1:
                        # reset classifier to untrained
                        self._reset()
                    elif self.nbSubjects == 2:
                        # reduce to OneClassSVM case
                        label = self.subject2label[sub]
                        aux = self.io_load(label)
                        self._reset()
                        
                        # prepare data
                        X, _ = self._prepareData(aux)
                        
                        self.subject2label[sub] = 0
                        self.subject2label[""] = -1
                        self._getSingleClf(X)
                        
                        # save data
                        self.io_save(0, aux)
                        
                    else:
                        # reorder the labels/subjects and models dicts
                        subject2label, nbSubjects, models = self._snapshot()
                        self._reset()
                        
                        label = subject2label[sub]
                        clabels = numpy.setdiff1d(numpy.unique(subject2label.values()),
                                                  [label], assume_unique=True)
                        self.nbSubjects = nbSubjects - 1
                        
                        i = 0
                        for ii in xrange(len(clabels)):
                            # build dicts
                            sub = subject2label[:clabels[ii]]
                            self.subject2label[sub] = i
                            
                            # move data
                            self.io_save(i, self.io_load(clabels[ii]))
                            
                            j = i + 1
                            for jj in xrange(ii + 1, len(clabels)):
                                # reorder models
                                self._reorderModels([i, j], [clabels[ii], clabels[jj]], models)
                                
                                # update j
                                j += 1
                            
                            # update i
                            i += 1
            else:
                # new subject
                # add to dicts
                label = self.nbSubjects
                self.subject2label[sub] = label
                
                if label == 1:
                    # remove the OneClassSVM model and label
                    self.subject2label.pop("")
                    self._removeModel([-1, 0])
                
                # save data
                self.io_save(label, data[sub])
                
                # prepare data
                X1, n1 = self._prepareData(data[sub])
                
                for j in xrange(self.nbSubjects):
                    # prepare data
                    X2, n2 = self._prepareData(self.io_load(j))
                    
                    # build new model
                    clff(X1, X2, n1, n2, label, j)
                
                # increment number of subjects
                self.nbSubjects += 1
    
    def authenticate(self, data, subject, threshold=None, ready=False, labels=False, **kwargs):
        # decision is made by majority
        # data is a list of feature vectors, allegedly belonging to the given subject
        
        # check train state
        if not self.is_trained:
            raise UntrainedError
        
        # translate subject ID to class label
        label = self.subject2label[subject]
        if label == -1:
            raise UnknownSubjectError(subject)
        
        if threshold is None:
            # get user-tuned threshold
            threshold = self.authThreshold(label, ready=True)
        
        # prepare data
        if not ready:
            X, n = self._prepareData(data)
        else:
            X, n = data
        
        # outputs
        decision = []
        prediction = []
        
        if self.nbSubjects == 1:
            prediction = self._predict([-1, 0], X)
            decision = [item == label for item in prediction]
        else:
            resL = []
            resW = []
            for p in self._iterTargets(label):
                L, W = self._predict(p, X)
                resL.append(L)
                resW.append(W)
            
            resL = numpy.array(resL)
            resW = numpy.array(resW)
            
            for i in xrange(n):
                # determine majority
                counts = numpy.bincount(resL[:, i], weights=resW[:, i])
                predMax = counts.argmax()
                rate = float(counts[predMax]) / numpy.sum(resW[:, i])
                if rate >= threshold:
                    # rate of agreement is >= threshold
                    decision.append(predMax == label)
                    prediction.append(predMax)
                else:
                    # rate of agreement is < threshold
                    decision.append(False)
                    prediction.append(predMax)
        
        # convert to numpy
        decision = numpy.array(decision)
        
        if labels:
            # translate class label to subject ID
            subPrediction = [self.subject2label[:item] for item in prediction]
            return decision, subPrediction
        else:
            return decision
    
    def _identify(self, data, threshold=None, ready=False, **kwargs):
        # data is list of feature vectors
        
        # check train state
        if not self.is_trained:
            raise UntrainedError
        
        # threshold
        if threshold is None:
            thrFcn = lambda label: self.idThreshold(label, ready=True)
        else:
            thrFcn = lambda label: threshold
        
        # prepare data
        if not ready:
            X, n = self._prepareData(data)
        else:
            X, n = data
        
        if self.nbSubjects == 1:
            labels = self._predict([-1, 0], X)
        else:
            # classify
            resL = []
            resW = []
            for p in self._iterTargets():
                L, W = self._predict(p, X)
                resL.append(L)
                resW.append(W)
            
            resL = numpy.array(resL)
            resW = numpy.array(resW)
            
            # majority
            labels = []
            for i in xrange(n):
                counts = numpy.bincount(resL[:, i], weights=resW[:, i])
                predMax = counts.argmax()
                rate = float(counts[predMax]) / numpy.sum(resW[:, i])
                if rate >= thrFcn(predMax):
                    # accept
                    labels.append(predMax)
                else:
                    # reject
                    labels.append(-1)
        
        return numpy.array(labels)


class LeanSVM(SVM):
    """
    Lean SVM classifier: the individual SVM classifiers are saved as .dict files.
    
    """
    
    NAME = 'leanSvmClassifier'
    EXT = '.hdf5'
    
    def __init__(self, io=None, **kwargs):
        # run parent __init__
        super(LeanSVM, self).__init__(io=io, **kwargs)
        
        # force file io
        if self.iomode != 'file':
            raise TypeError, "LeanSVM only supports the 'file' IO mode."
    
    def _snapshot(self):
        # run parent snapshot
        subject2label, nbSubjects, models = super(LeanSVM, self)._snapshot()
        
        # individual classifiers
        for p, n in self.models.iteritems():
            nn = 'tmp_%s' % n
            models[p] = nn
            
            os.rename(os.path.join(self.cPath, n), os.path.join(self.cPath, nn))
        
        return subject2label, nbSubjects, models
    
    def _rebrand(self, name):
        # change io file name
        
        # run parent
        super(LeanSVM, self)._rebrand(name)
        
        # individual classifiers
        new = os.path.join(self.iopath, '%s_clfs' % self.NAME)
        os.rename(self.cPath, new)
        self.cPath = new
    
    def _prepareIO(self):
        # create dirs, initialize files
        
        # run parent
        super(LeanSVM, self)._prepareIO()
        
        # get path for individual classifiers
        self.cPath = os.path.join(self.iopath, '%s_clfs' % self.NAME)
        if not os.path.exists(self.cPath):
            os.makedirs(self.cPath)
    
    def _getSingleClf(self, X):
        # instantiate and train a OneClassSVM classifier
        
        # instantiate and fit
        clf = sksvm.OneClassSVM(kernel='rbf', nu=0.1)
        clf.fit(X)
        
        # add to models
        n = '-1_0.clf'
        self.models[(-1, 0)] = n
        
        # save
        datamanager.skStore(os.path.join(self.cPath, n), clf)
    
    def _getLinearClf(self, X1, X2, n1, n2, label1, label2):
        # instantiate and train a LinearSVC classifier
        
        # prepare data to train
        X = numpy.concatenate((X1, X2), axis=0)
        Y = numpy.ones(n1 + n2)
        if label1 < label2:
            Y[:n1] = -1
            pair = (label1, label2)
        else:
            Y[n1:] = -1
            pair = (label2, label1)
        
        # class weights (inversely proportional to number of samples in class)
        weights = self._weights(n1, n2)
        
        # instantiate and fit
        clf = sksvm.LinearSVC(class_weight=weights, **self.clf_kwargs)
        clf.fit(X, Y)
        
        # add to models
        n = '%d_%d.clf' % pair
        self.models[pair] = n
        
        # save
        datamanager.skStore(os.path.join(self.cPath, n), clf)
    
    def _getKernelClf(self, X1, X2, n1, n2, label1, label2):
        # instantiate and train an SVC classifier
        
        # prepare data to train
        X = numpy.concatenate((X1, X2), axis=0)
        Y = numpy.ones(n1 + n2)
        if label1 < label2:
            Y[:n1] = -1
            pair = (label1, label2)
        else:
            Y[n1:] = -1
            pair = (label2, label1)
        
        # class weights (inversely proportional to number of samples in class)
        weights = self._weights(n1, n2)
        
        # instantiate and fit
        clf = sksvm.SVC(class_weight=weights, **self.clf_kwargs)
        clf.fit(X, Y)
        
        # add to models
        n = '%d_%d.clf' % pair
        self.models[pair] = n
        
        # save
        datamanager.skStore(os.path.join(self.cPath, n), clf)
    
    def _predict(self, pair, data):
        # get the classifier prediction
        
        # convert pair
        pair = self._convertPair(pair)
        
        # load individual classifier
        clf = datamanager.skLoad(os.path.join(self.cPath, self.models[pair]))
        
        # predict
        aux = clf.predict(data)
        res = -1 * numpy.ones(aux.shape, dtype='int')
        res[aux < 0] = pair[0]
        res[aux > 0] = pair[1]
        
        return res
    
    def _reorderModels(self, pairN, pairO, models):
        # reorder the models
        
        pairN = self._convertPair(pairN)
        pairO = self._convertPair(pairO)
        
        # rename
        n = '%d_%d.clf' % pairN
        os.rename(os.path.join(self.cPath, models[pairO]), os.path.join(self.cPath, n))
        
        self.models[pairN] = n
    
    def _removeModel(self, pair):
        # remove a model
        
        pair = self._convertPair(pair)
        
        # remove individual classifier
        n = self.models.pop(pair)
        try:
            os.remove(os.path.join(self.cPath, n))
        except OSError:
            pass
    
    def fileIterator(self):
        # iterator for the classifier files
        
        # run parent
        for item in super(LeanSVM, self).fileIterator():
            yield item
        
        # individual classifiers
        d = os.path.split(self.cPath)[1]
        for n in self.models.itervalues():
            yield os.path.join(d, n)
    
    def dirIterator(self):
        # iterator for the directories
        
        # run parent
        for item in super(LeanSVM, self).dirIterator():
            yield item
        
        yield os.path.split(self.cPath)[1]


class SifterSVM(BaseClassifier):
    """
    SVM classifier.
    """
    
    NAME = 'SifterSVM'
    EXT = '.hdf5'
    
    def __init__(self, io=None, edges=None, weights=None, **kwargs):
        # run parent __init__
        super(SifterSVM, self).__init__(io=io)
        
        # check inputs
        if edges is None:
            raise TypeError, "Please specify the sieve edges."
        
        # process pool
        # self.pool = parallel.getPool(6)
        
        # algorithm self things
        self.clfs = []
        self.nbLevels = len(edges)
        for i in xrange(self.nbLevels):
            c = SVM(io=io, **kwargs)
            c._rebrand('level-%d' % i)
            # c.pool.close()
            # c.pool = self.pool
            self.clfs.append(c)
        
        if weights is None:
            self.weights = numpy.ones(self.nbLevels, dtype='float')
        else:
            self.weights = weights
        
        assert self.nbLevels == len(self.weights)
    
    def _reset(self, recursive=True):
        # run parent reset
        super(SifterSVM, self)._reset()
        
        # reset each classifier
        if recursive:
            try:
                clfs = self.clfs
            except AttributeError:
                pass
            else:
                for c in clfs:
                    c._reset()
    
    def _prepareIO(self):
        # create dirs, initialize files
        
        pass
    
    def fileIterator(self):
        
        # individual classifiers
        for c in self.clfs:
            for item in c.fileIterator():
                yield item
    
    def dirIterator(self):
        
        # individual classifiers
        for c in self.clfs:
            for item in c.dirIterator():
                yield item
    
    def autoRejectionThresholds(self):
        # generate thresholds automatically
        
        # auto thresholds for each classifier
        cths = []
        for c in self.clfs:
            cths.append(c.autoRejectionThresholds())
        
        # combination threshold
        ths = numpy.linspace(0., 1., self.nbSubjects + 1)
        
        return cths, ths
    
    def combination(self, results):
        # combine the identification or authentication results
        
        # compile results to find all classes
        vec = numpy.concatenate(results)
        unq = numpy.unique(vec)
        
        nb = len(unq)
        if nb == 0:
            # empty combination
            raise EmptyCombination
        elif nb == 1:
            # unanimous result
            return unq[0], 1.
        else:
            # multi-class
            counts = numpy.zeros(nb, dtype='float')
            
            for n in xrange(self.nbLevels):
                # ensure array
                res = numpy.array(results[n])
                ns = float(len(res))
                
                if ns > 0:
                    # get count for each unique class
                    for i in xrange(nb):
                        aux = float(numpy.sum(res == unq[i]))
                        counts[i] += ((aux / ns) * self.weights[n])
            
            # most frequent class
            predMax = counts.argmax()
            counts /= counts.sum()
            
            return unq[predMax], counts[predMax]
    
    def train(self, data=None):
        # data is {subject: features (sifted array)}
        
        # check inputs
        if data is None:
            raise TypeError, "Please provide input data."
        
        # check if classifier was already trained
        if self.is_trained:
            # this is a retrain
            self.re_train(data)
        else:
            # get subjects
            subjects = data.keys()
            self.nbSubjects = len(subjects)
            
            # determine classifier mode
            if self.nbSubjects == 0:
                raise ValueError, "Please provide input data - empty dict."
            else:
                for i in xrange(self.nbSubjects):
                    # build dicts
                    sub = subjects[i]
                    self.subject2label[sub] = i
                    
                    # train each classifier
                    for n in xrange(self.nbLevels):
                        aux = data[sub][n]
                        if len(aux) > 0:
                            self.clfs[n].train({sub: aux})
        
        # train flag
        self.is_trained = True
    
    def re_train(self, data):
        # data is {subject: features (sifted array)}
        
        for sub in data.iterkeys():
            if sub in self.subject2label:
                # existing subject
                if data[sub] is not None:
                    # change subject's data
                    
                    # re-train each classifier
                    for n in xrange(self.nbLevels):
                        aux = data[sub][n]
                        if len(aux) > 0:
                            self.clfs[n].train({sub: aux})
                else:
                    # delete subject
                    if self.nbSubjects == 1:
                        # reset classifier to untrained
                        self._reset()
                    else:
                        # reorder the labels/subjects and models dicts
                        subject2label, nbSubjects = self._snapshot()
                        self._reset(recursive=False)
                        
                        # delete from each classifier
                        for n in xrange(self.nbLevels):
                            c = self.clfs[n]
                            if sub in c.subject2label:
                                c.train({sub: None})
                        
                        label = subject2label[sub]
                        clabels = numpy.setdiff1d(numpy.unique(subject2label.values()),
                                                  [label], assume_unique=True)
                        self.nbSubjects = nbSubjects - 1
                        
                        i = 0
                        for ii in xrange(len(clabels)):
                            # build dicts
                            sub = subject2label[:clabels[ii]]
                            self.subject2label[sub] = i
                            
                            # update i
                            i += 1
            else:
                # new subject
                # add to dicts
                label = self.nbSubjects
                self.subject2label[sub] = label
                
                # train each classifier
                for n in xrange(self.nbLevels):
                    aux = data[sub][n]
                    if len(aux) > 0:
                        self.clfs[n].train({sub: aux})
                
                # increment number of subjects
                self.nbSubjects += 1
    
    def authenticate(self, data, subject, threshold=None, ready=False, labels=False, **kwargs):
        # data is a list of sifted feature vectors, allegedly belonging to the given subject
        
        # check train state
        if not self.is_trained:
            raise UntrainedError
        
        # check existence of subject
        if subject not in self.subject2label:
            raise UnknownSubjectError(subject)
        
        # check thresholds
        if threshold is None:
            cths, ths = self.autoRejectionThresholds()
        else:
            try:
                cths = threshold[0]
                ths = threshold[1]
            except TypeError:
                # threshold is not tuple
                cths, _ = self.autoRejectionThresholds()
                ths = [threshold]
        
        # get decision from each classifier
        decision = []
        prediction = []
        for n in xrange(self.nbLevels):
            aux = data[n]
            c = self.clfs[n]
            if (len(aux) > 0) and (subject in c.subject2label):
                d, p = c.authenticate(aux, subject, threshold=cths[n][-1], labels=True)
                decision.append(d)
                prediction.append([self.subject2label[item:] for item in p])
            else:
                decision.append([])
                prediction.append([])
        
        # combine results
        try:
            decision, confidence = self.combination(decision)
        except EmptyCombination:
            decision = False
        else:
            if confidence < ths[-1]:
                # confidence below threshold
                decision = False
        decision = [decision]
        
        try:
            prediction, _ = self.combination(prediction)
        except EmptyCombination:
            prediction = numpy.array([''])
        else:
            prediction = numpy.array([self.subject2label[:prediction]])
        
        # convert to numpy
        decision = numpy.array(decision)
        
        if labels:
            return decision, prediction
        else:
            return decision
    
    def identify(self, data, ready=False, **kwargs):
        # data is a list of sifted feature vectors
        
        # check train state
        if not self.is_trained:
            raise UntrainedError
        
        # get decision from each classifier
        prediction = []
        for n in xrange(self.nbLevels):
            aux = data[n]
            c = self.clfs[n]
            if len(aux) > 0:
                p = c.identify(aux)
                prediction.append([self.subject2label[item:] for item in p])
            else:
                prediction.append([])
        
        # combine results
        try:
            prediction, _ = self.combination(prediction)
        except EmptyCombination:
            prediction = numpy.array([''])
        else:
            prediction = numpy.array([self.subject2label[:prediction]])
        
        return prediction
    
    def evaluate(self, data, rejection_thresholds='auto', dstPath=None, log2file=False):
        # data is {subject: features (sifted array)}
        
        # check train state
        if not self.is_trained:
            raise UntrainedError
        
        # choose thresholds
        if rejection_thresholds == 'auto':
            cths, ths = self.autoRejectionThresholds()
        else:
            cths, ths = rejection_thresholds
        
        # choose store
        if dstPath is None:
            store = parallel.getDictManager()
            storeF = [parallel.getDictManager() for _ in self.clfs]
            # storeF = lambda name: parallel.getDictManager()
        elif isinstance(dstPath, basestring):
            store = os.path.join(dstPath, self.NAME)
            if not os.path.exists(store):
                os.makedirs(store)
            storeF = []
            for c in self.clfs:
                aux = os.path.join(dstPath, c.NAME)
                if not os.path.exists(aux):
                    os.makedirs(aux)
                storeF.append(aux)
            # storeF = lambda name: os.path.join(dstPath, name)
        else:
            store = dstPath
            storeF = [dstPath for _ in self.clfs]
            # storeF = lambda name: dstPath
        
        subjects = numpy.array(data.keys())
        
        # evaluate each of the classifiers
        clfres = []
        EER_th = numpy.zeros((self.nbLevels, 1), dtype='float')
        for n in xrange(self.nbLevels):
            # get data from each subject
            aux = {}
            for sub in subjects:
                if len(data[sub][n]) > 0:
                    aux[sub] = data[sub][n]
            
            # evaluate classifier
            c = self.clfs[n]
            # name = c.NAME
            
            try:
                cevl = c.evaluate(aux, cths[n], dstPath=storeF[n], log2file=log2file)
            except UntrainedError:
                cevl = {}
            else:
                # use thresholds at EER
                EER_th[n, 0] = cevl['assessment']['global']['authentication']['rates']['EER'][0, 0]
            
            clfres.append(cevl)
        
        # return clfres
        
        # evaluate combination
        results = {}
        nths = len(ths)
        for sub in subjects:
            # get identification labels
            id_res = []
            for n in xrange(self.nbLevels):
                try:
                    idv = clfres[n]['classification'][sub]['identification']
                except KeyError:
                    idv = []
                id_res.append(idv)
            # combine
            try:
                id_res, _ = self.combination(id_res)
            except EmptyCombination:
                id_res = ''
            id_res = [id_res]
            
            # authentication test
            auth_res = []
            workQ = parallel.getQueue()
            for i in xrange(nths):
                workQ.put({
                           'classifier': self,
                           'data': data[sub],
                           'threshold': (EER_th, [ths[i]]),
                           'subjects': subjects,
                           'parameters': {'ready': True},
                           'taskid': i,
                           })
            
            # run in multiprocessing
            parallel.runMultiprocess(workQ, store, mode='clf', log2file=log2file)
            
            # load from files
            for i in xrange(nths):
                auth_res.append(parallel.loadStore(store, i))
            
            # clean up store
            parallel.cleanStore(store)
            
            auth_res = numpy.array(auth_res)
            results[sub] = {'identification': id_res,
                            'authentication': {'cube': auth_res,
                                               'subjectLabels': subjects,
                                               },
                            }
        
        
        # assess classification results
        assess = evaluation.assessClassification(results, ths,
                                                 dstPath=dstPath, log2file=log2file)
        
        # final output
        output = {'classification': results,
                  'assessment': assess,
                  'singles': clfres,
                  }
        
        return output


class Odinaka(BaseClassifier):
    """
    Odinaka classifier.
    """
    
    NAME = 'odinakaClassifier'
    EXT = '.hdf5'
    
    def __init__(self, io=None, n=None, overlap=0, nfft=None, size=None, window='hamming', k=0.):
        # run parent __init__
        super(Odinaka, self).__init__(io=io)
        
        # parameters
        if n is None:
            raise TypeError, "Please specify thw window length."
        
        if overlap >= n:
            raise ValueError, "Overlap is bigger than or equal to window length."
        
        if nfft is None:
            nfft = n
        
        if size is None:
            size = nfft / 2
        
        if size > (nfft / 2):
            raise ValueError, "Size is bigger than NFFT/2."
        
        self.n = n
        self.overlap = overlap
        self.nfft = nfft
        self.size = size
        self.window = featureextraction.getWindow(window, n)
        self.k = k
        
        # algorithm self things
        self.userModels = {}
        self.globalModels = {}
    
    def _reset(self):
        # run parent reset
        super(Odinaka, self)._reset()
        
        self.userModels = {}
        self.globalModels = {}
    
    def _snapshot(self):
        # run parent snapshot
        subject2label, nbSubjects = super(Odinaka, self)._snapshot()
        
        userModels = self.userModels
        globalModels = self.globalModels
        
        return subject2label, nbSubjects, userModels, globalModels
    
    def _prepareData(self, data):
        # prepare data
        
        X = []
        for tpl in data:
            # compute the short time FT
            aux = featureextraction.shortTimeFT(tpl, self.n, self.overlap, self.nfft, self.window)
            # transform
            aux = numpy.log(numpy.abs(aux[:, :self.size])**2)
            X.append(aux)
        
        X = numpy.array(X)
        
        return X
    
    def _getSubjectParams(self, X, label):
        # compute mean and variance of STFT
        
        m = numpy.mean(X, axis=0)
        m = m.flatten()
        v = numpy.var(X, ddof=1, axis=0)
        v = v.flatten()
        c = 1. / numpy.sqrt(2 * numpy.pi * v)
        
        self.userModels[label] = {'m': m, 'v': v, 'd': 0., 'N': X.shape[0], 'const': c}
    
    def _getGlobalParams(self):
        # compute global mean and variance
        
        # average the user means and variances
        m = 0.
        v = 0.
        for label in xrange(self.nbSubjects):
            m += self.userModels[label]['m']
            v += self.userModels[label]['v']
        
        nb = float(self.nbSubjects)
        m /= nb
        v /= nb
        
        self.globalModels['m'] = m
        self.globalModels['v'] = v
        # self.globalModels['const'] = c = 1. / numpy.sqrt(2 * numpy.pi * v)
        self.globalModels['const'] = 1. / numpy.sqrt(2 * numpy.pi * v)
        
        # compute distance
        for label in xrange(self.nbSubjects):
            mi = self.userModels[label]['m']
            vi = self.userModels[label]['v']
            # ci = self.userModels[label]['const']
            # di = self.userModels[label]['d']
            aux = (mi - m)**2
            
            # distance
            d = (vi + aux) / (2 * v) + (v + aux) / (2 * vi) - 1
            self.userModels[label]['d'] = d
    
    def _predict(self, X, labels=None):
        # get the classifier prediction
        
        max_score = -numpy.inf
        guess = -1
        
        if labels is None:
            labels = xrange(self.nbSubjects)
        
        # mt = numpy.mean(X, axis=0)
        mt = X.flatten()
        
        m = self.globalModels['m']
        v = self.globalModels['v']
        c = self.globalModels['const']
        
        B = 1 + c * numpy.exp((-(mt - m)**2) / (2 * v))
        
        for label in labels:
            mi = self.userModels[label]['m']
            vi = self.userModels[label]['v']
            ci = self.userModels[label]['const']
            di = self.userModels[label]['d']
            
            A = 1 + ci * numpy.exp((-(mt - mi)**2) / (2 * vi))
            
            score = numpy.sum(numpy.log(A / B) * (di > self.k))
            
            if score > max_score and score < numpy.inf:
                max_score = score
                guess = label
        
        return guess, max_score
    
    def _updateStrategy(self, oldData, newData):
        # update the training data of a class when new data is available
        
        # concatenate old with new data
        out = numpy.concatenate([oldData, newData], axis=0)
        
        return out
    
    def autoRejectionThresholds(self):
        # generate thresholds automatically
        
        scores = []
        for label in xrange(self.nbSubjects):
            labels = [label]
            data = self.io_load(label)
            for i in xrange(len(data)):
                aux = [data[i]]
                X = self._prepareData(aux)
                guess, score = self._predict(X, labels=labels)
                
                if guess == label:
                    scores.append(score)
                    break
        
        scores = numpy.array(scores)
        
        std = scores.std(ddof=1)
        mi = scores.min() - std
        ma = scores.max() + std
        
        ths = numpy.linspace(mi, ma, self.nbSubjects + 1)
        
        return ths
    
    def train(self, data=None):
        # data is {subject: features (array)}
        
        # check inputs
        if data is None:
            raise TypeError, "Please provide input data."
        
        # check if classifier was already trained
        if self.is_trained:
            # this is a retrain
            self.re_train(data)
        else:
            # get subjects
            subjects = data.keys()
            self.nbSubjects = len(subjects)
            
            # determine classifier mode
            if self.nbSubjects == 0:
                raise ValueError, "Please provide input data - empty dict."
            else:
                for i in xrange(self.nbSubjects):
                    # build dicts
                    sub = subjects[i]
                    self.subject2label[sub] = i
                    
                    # save data
                    self.io_save(i, data[sub])
                    
                    # prepare data
                    X = self._prepareData(data[subjects[i]])
                    
                    # get user model
                    self._getSubjectParams(X, i)
        
        # get global model
        self._getGlobalParams()
        
        # train flag
        self.is_trained = True
    
    def re_train(self, data):
        # data is {subject: features (array)}
        
        for sub in data.iterkeys():
            if sub in self.subject2label:
                # existing subject
                if data[sub] is not None:
                    # change subject's data
                    label = self.subject2label[sub]
                    
                    # update templates
                    aux = self._updateStrategy(self.io_load(label), data[sub])
                    
                    # save data
                    self.io_save(label, aux)
                    
                    # prepare data
                    X = self._prepareData(aux)
                    
                    # get user model
                    self._getSubjectParams(X, label)
                else:
                    # delete subject
                    if self.nbSubjects == 1:
                        # reset classifier to untrained
                        self._reset()
                    else:
                        # reorder the labels/subjects and models dicts
                        subject2label, nbSubjects, userModels, _ = self._snapshot()
                        self._reset()
                        
                        label = subject2label[sub]
                        self.nbSubjects = nbSubjects - 1
                        
                        for i in xrange(0, label):
                            self.subject2label[:i] = subject2label[:i]
                            self.userModels[i] = userModels[i]
                        
                        for i in xrange(label, self.nbSubjects):
                            self.subject2label[:i] = subject2label[:i + 1]
                            self.userModels[i] = userModels[i + 1]
            else:
                # new subject
                # add to dicts
                label = self.nbSubjects
                self.subject2label[sub] = label
                
                # save data
                self.io_save(label, data[sub])
                
                # prepare data
                X = self._prepareData(data[sub])
                
                # get user model
                self._getSubjectParams(X, label)
                
                # increment number of subjects
                self.nbSubjects += 1
    
    def authenticate(self, data, subject, threshold=None, ready=False, labels=False, **kwargs):
        # data is a list of feature vectors, allegedly belonging to the given subject
        
        # check train state
        if not self.is_trained:
            raise UntrainedError
        
        # translate subject ID to class label
        label = self.subject2label[subject]
        if label == -1:
            raise UnknownSubjectError(subject)
        
        # check threshold
        if threshold is None:
            threshold = -numpy.inf
        
        # prepare data
        if not ready:
            X = self._prepareData(data)
        else:
            X = data
        
        # outputs
        decision = []
        prediction = []
        
        testLabels = [label]
        for i in xrange(X.shape[0]):
            # predict
            guess, score = self._predict(X[i, :, :], labels=testLabels)
            
            # test against threshold
            if score >= threshold:
                # criterion met
                decision.append(guess == label)
                prediction.append(guess)
            else:
                # criterion NOT met
                decision.append(False)
                prediction.append(guess)
        
        # convert to numpy
        decision = numpy.array(decision)
        
        if labels:
            # translate class label to subject ID
            subPrediction = [self.subject2label[:item] for item in prediction]
            return decision, subPrediction
        else:
            return decision
    
    def identify(self, data, ready=False, **kwargs):
        # data is list of feature vectors
        
        # check train state
        if not self.is_trained:
            raise UntrainedError
        
        # prepare data
        if not ready:
            X = self._prepareData(data)
        else:
            X = data
        
        labels = []
        for i in xrange(X.shape[0]):
            # predict
            guess, _ = self._predict(X[i, :, :])
            
            labels.append(guess)
        
        # translate class labels to subject IDs
        subjects = numpy.array([self.subject2label[:item] for item in labels])
        
        return subjects


class KNN(BaseClassifier):
    """
    k-NN classifier.
    """
    
    NAME = 'KNN'
    EXT = '.hdf5'
    EER_IDX = 0
    
    def __init__(self, io=None, k=3, metric='euclidean', **kwargs):
        # run parent __init__
        super(KNN, self).__init__(io=io)
        
        # algorithm self things
        self.k = k
        self.metric = metric
        
        # minimum threshold
        self.minThr = 10 * numpy.finfo('float').eps
        
        # choose metric
        if metric == 'euclidean':
            self.distFcn = misc.msedistance
        elif metric == 'cosine':
            self.distFcn = misc.cosdistance
        elif metric == 'wavelet':
            self.distFcn = wavelets.waveDist
        elif metric == 'cp77':
            raise NotImplementedError, "Metric %s not yet implemented." % metric
        else:
            raise NotImplementedError, "Metric %s unknown." % metric
    
    def _prepareData(self, data):
        # prepare data - compute the distances
        
        dists, trainLabels = self._compute(data, label='all')
        
        return {'dists': dists, 'trainLabels': trainLabels}
     
    def _compute(self, data, label='all'):
        # compute the distance of the observations to each of the class labels
        
        # class labels
        if label == 'all':
            clabels = self.subject2label.values()
        else:
            clabels = [label]
        
        # compute distances
        if len(clabels) > (4 * config.numberProcesses):
            # run in multiprocessing
            workQ = parallel.getQueue()
            for lbl in clabels:
                workQ.put({
                           'function': self.distFcn,
                           'testData': data,
                           'trainData': self.io_load(lbl),
                           'taskid': lbl,
                           })
            # run in multiprocess (distances mode)
            store = parallel.getDictManager()
            parallel.runMultiprocess(workQ, store, mode='distances')
            
            # amass results
            dists = numpy.concatenate([store[lbl]['distances'] for lbl in clabels], axis=1)
            trainLabels = numpy.concatenate([store[lbl]['labels'] for lbl in clabels], axis=1)
        else:
            # run sequentially
            dists = []
            trainLabels = []
            ltd = len(data)
            for lbl in clabels:
                trainData = self.io_load(lbl)
                # labels
                trainLabels.append(lbl * numpy.ones((ltd, len(trainData)), dtype='int'))
                # distances
                dists.append([misc.wavedistance(obs, trainData, self.distFcn) for obs in data])
            
            # amass results
            dists = numpy.concatenate(dists, axis=1)
            trainLabels = numpy.concatenate(trainLabels, axis=1)
        
        # sort
        dists, trainLabels = self._sort(dists, trainLabels)
        
        return dists, trainLabels
    
    def _sort(self, dists, trainLabels):
        # sort
        
        ind = dists.argsort(axis=1)
        # sneaky trick from http://stackoverflow.com/questions/6155649
        static_inds = numpy.arange(dists.shape[0]).reshape((dists.shape[0], 1))
        dists = dists[static_inds, ind]
        trainLabels = trainLabels[static_inds, ind]
        
        return dists, trainLabels
    
    def _updateStrategy(self, oldData, newData):
        # update the training data of a class when new data is available
        
        # concatenate old with new data
        out = numpy.concatenate([oldData, newData], axis=0)
        
        return out
    
    def autoRejectionThresholds(self):
        # generate thresholds automatically
        
        if self._autoThresholds is not None:
            return self._autoThresholds
        
        if self.metric == 'cosine':
            self._autoThresholds = numpy.linspace(self.minThr, 1., 100)
            return self._autoThresholds
        
        maxD = []
        for _ in xrange(3):
            # repeat
            for label in self.subject2label.values():
                # randomly select one sample
                aux = self.io_load(label)
                ind = config.random.randint(0, aux.shape[0], 1)
                obs = aux[ind]
                
                # compute distances
                dists, _ = self._compute(obs)
                maxD.append(numpy.max(numpy.max(dists)))
        
        # max distance
        maxD = 1.5 * numpy.max(maxD)
        
        # rejection thresholds to test
        self._autoThresholds = numpy.linspace(self.minThr, maxD, 100)
        
        return self._autoThresholds
    
    def train(self, data=None, updateThresholds=True):
        # data is {subject: features (array)}
        
        # check inputs
        if data is None:
            raise TypeError, "Please provide input data."
        
        # check if classifier was already trained
        if self.is_trained:
            # this is a retrain
            self.re_train(data)
        else:
            # get subjects
            subjects = data.keys()
            self.nbSubjects = len(subjects)
            
            # determine classifier mode
            if self.nbSubjects == 0:
                raise ValueError, "Please provide input data - empty dict."
            else:
                for i in xrange(self.nbSubjects):
                    # build dicts
                    sub = subjects[i]
                    self.subject2label[sub] = i
                    
                    # save data
                    self.io_save(i, data[sub])
        
        # train flag
        self.is_trained = True
        
        if updateThresholds:
            # update thresholds
            self.updateThresholds()
    
    def re_train(self, data):
        # data is {subject: features (array)}
        ### user
        
        for sub in data.iterkeys():
            if sub in self.subject2label:
                # existing subject
                if data[sub] is not None:
                    # change subject's data
                    label = self.subject2label[sub]
                    
                    # update templates
                    aux = self._updateStrategy(self.io_load(label), data[sub])
                    
                    # save data
                    self.io_save(label, aux)
                else:
                    # delete subject
                    if self.nbSubjects == 1:
                        # reset classifier to untrained
                        self._reset()
                    else:
                        # reorder the labels/subjects and models dicts
                        subject2label, nbSubjects = self._snapshot()
                        self._reset()
                        
                        label = subject2label[sub]
                        clabels = numpy.setdiff1d(numpy.unique(subject2label.values()),
                                                  [label], assume_unique=True)
                        self.nbSubjects = nbSubjects - 1
                        
                        i = 0
                        for ii in xrange(len(clabels)):
                            # build dicts
                            sub = subject2label[:clabels[ii]]
                            self.subject2label[sub] = i
                            
                            # move data
                            self.io_save(i, self.io_load(clabels[ii]))
                            
                            # update i
                            i += 1
            else:
                # new subject
                # add to dicts
                label = self.nbSubjects
                self.subject2label[sub] = label
                
                # save data
                self.io_save(label, data[sub])
                
                # increment number of subjects
                self.nbSubjects += 1
    
    def authenticate(self, data, subject, threshold=None, ready=False, labels=False, **kwargs):
        # decision is made by majority
        
        # check train state
        if not self.is_trained:
            raise UntrainedError
        
        # translate subject ID to class label
        label = self.subject2label[subject]
        if label == -1:
            raise UnknownSubjectError(subject)
        
        if threshold is None:
            # get user-tuned threshold
            threshold = self.authThreshold(label, ready=True)
        
        # compute distances
        if ready:
            # data has the precomputed distances
            dists = data['dists']
            trainLabels = data['trainLabels']
            
            # select based on subject label
            aux = []
            for i in xrange(len(dists)):
                aux.append(dists[i, trainLabels[i, :] == label])
            dists = numpy.array(aux)
        else:
            # compute distances
            dists, trainLabels = self._compute(data, label)
        
        # nearest neighbors
        dists = dists[:, :self.k]
        
        # outputs
        decision = []
        prediction = []
        for i in xrange(len(dists)):
            # compare distances to threshold
            count = numpy.sum(dists[i, :] <= threshold)
            
            # decide
            if count > (self.k / 2):
                # accept
                decision.append(True)
            else:
                # reject
                decision.append(False)
            
            prediction.append(label)
        
        # convert to numpy
        decision = numpy.array(decision)
        
        if labels:
            # translate class label to subject ID
            subPrediction = [self.subject2label[:item] for item in prediction]
            return decision, subPrediction
        else:
            return decision
    
    def _identify(self, data, threshold=None, ready=False, **kwargs):
        # data is list of feature vectors
        
        # check train state
        if not self.is_trained:
            raise UntrainedError
        
        # threshold
        if threshold is None:
            thrFcn = lambda label: self.idThreshold(label, ready=True)
        else:
            thrFcn = lambda label: threshold
        
        if ready:
            # data has the precomputed distances
            dists = data['dists']
            trainLabels = data['trainLabels']
        else:
            # compute distances
            dists, trainLabels = self._compute(data)
        
        # nearest neighbors
        dists = dists[:, :self.k]
        trainLabels = trainLabels[:, :self.k]
        
        # outputs
        labels = []
        for i in xrange(len(trainLabels)):
            lbl = rules.pluralityRule(trainLabels[i, :])[0]
            
            # compare distances to threshold
            count = numpy.sum(dists[i, :] <= thrFcn(lbl))
            
            # decide
            if count > (self.k / 2):
                # accept
                labels.append(lbl)
            else:
                # reject
                labels.append(-1)
        
        return numpy.array(labels)


class DissimilarityMore(KNN):
    """
    Class to extract multilead distances subjectwise
    
    Performs kNN for classification based on those distances
    """
    
    NAME = 'dissimilarityClassifier'
    EXT = '.hdf5'
    EER_IDX = 0
    
    def __init__(self, io=None, k=3, metric='euclidean', featmetric='euclidean', leads=('I', 'II', 'III'), reflead='I',
                 median=False, meansubs=None, percent=0.15, tn=5, **kwargs):
        # run parent __init__
        super(DissimilarityMore, self).__init__(io=io, k=k)
        
        # algorithm self things
        self.k = k
        self.leads = leads
        self.leadlen = len(leads)
        self.reflead = reflead
        self.median = median
        self.percent = percent
        self.meansubs = meansubs
        self.tn = tn
        
        # minimum threshold
        self.minThr = 10 * numpy.finfo('float').eps
        
        # choose metric
        if metric == 'euclidean':
            self.metric = 'euclidean'
        elif metric == 'cosine':
            self.metric = 'cosine'
        else:
            raise NotImplementedError("Metric %s unknown." % metric)
        
        # choose metric
        if featmetric == 'euclidean':
            self.featmetric = 'euclidean'
        elif featmetric == 'cosine':
            self.featmetric = 'cosine'
        else:
            raise NotImplementedError("Metric %s unknown." % featmetric)
    
    def _fileIO_load(self, label):
        # load label from file
        
        fmt = '%s-%s'
        data = {}
        for l in self.leads:
            data[l] = datamanager.h5Load(self.iofile, fmt % (label, l))
        
        return data
    
    def _fileIO_save(self, label, data):
        # save data with label to file
        
        fmt = '%s-%s'
        for l in self.leads:
            datamanager.h5Store(self.iofile, fmt % (label, l), data[l])
    
    def _reset(self):
        # reset the classifier
        self.subject2label = SubjectDict()
        self.nbSubjects = 0
        self.is_trained = False
        self.thresholds = {}
        self._autoThresholds = None
        self.leadavglist = []
    
    def _snapshot(self):
        # snapshot of classifier structures
        subject2label = self.subject2label
        nbSubjects = self.nbSubjects
        leadavglist = self.leadavglist
        
        return subject2label, nbSubjects, leadavglist
    
    def dict2list(self, dictionary):
        """
        Gets a dictionary as input and returns a list
        
        Needed so as to join all average ECGs leadwise and feed them to scipy.spatial.distance.cdist()
        """
        
        datalist = []
        for lead in self.leads:
            datalist.append(dictionary[lead])
            
        return datalist
    
    def extract_feats(self, data, label=None):
        """
        data - dictionary whose entries are discriminated by leads; entries have a matrix of ECG segments
        """
        
        feats = dist.cdist(data[self.reflead], self.leadavglist, self.featmetric)
        
        return feats
    
    def _avgmaker(self, subjects, data):
        leadavglist = []
        
        # GET THE NUMBER OF USED REFERENCES
        
        # random choice of subjects -- should be done outside classifier
        subints = range(self.nbSubjects)
        if self.meansubs is None:
            self.representatives = int(self.percent * self.nbSubjects)
            if not self.representatives:
                self.representatives = 1
            indx = config.random.choice(subints, size=self.representatives, replace=False)
            self.meansubs = [subjects[i] for i in indx]
        
        # distance to templates (organization is made as in C leads for template i, C leads for template i+1 (...))
        for sub in self.meansubs:
            leadavg = dict()
            for lead in self.leads:
                for template in xrange(self.tn):
                    if len(data[sub][lead][template]) > 1:
                        leadavg[lead] = data[sub][lead][template]
#                     if len(data[sub][lead + '-tpl'][template]) > 1:
#                         leadavg[lead] = data[sub][lead + '-tpl'][template]
                    else:
                        pass
                    leadavglist.append(leadavg[lead])
                    
        self.leadavglist = numpy.array(leadavglist)
    
    def train(self, data=None, updateThresholds=True):
        # data is {subject: {leads: beats, templates: {leads: beats}}
        # print len(data), len(data[0]), len(data[0][0]), len(data[0][1])
        
        # check inputs
        if data is None:
            raise TypeError("Please provide input data.")
        
        # check if classifier was already trained
        if self.is_trained:
            # this is a retrain
            # print 'RETRAIN'
            self.re_train(data)
        else:
            # get subjects
            subjects = data.keys()
            # print '\t WTF', subjects
            self.nbSubjects = len(subjects)
            dissim_cannon = []
            cannon_label = []
            # create average ECG segments
            self._avgmaker(subjects, data)
            
            # determine classifier mode
            if self.nbSubjects == 0:
                raise ValueError("Please provide input data - empty dict.")
            else:
                for i in xrange(self.nbSubjects):
                    # build dicts
                    sub = subjects[i]
                    self.subject2label[sub] = i
                    # print '\t*****', i, sub, self.subject2label, '*****'
                    
                    # extract dissimilarities
                    feats = self.extract_feats(data[sub])
                    
                    # build feature cannons
                    dissim_cannon.append(feats)
                    cannon_label.append(numpy.ones(len(data[sub][self.reflead]), dtype='int64')*i)
                    
                    # save data
                    self.io_save(i, data[sub])
                    
            # concatenate cannon and label
            self.feat_cannon = numpy.concatenate(dissim_cannon, axis=0)
            self.cannon_label = numpy.concatenate(cannon_label, axis=0)
            # print '--->', numpy.shape(self.feat_cannon), numpy.shape(self.cannon_label)
            # print self.feat_cannon
            # print self.cannon_label
            
        self.is_trained = True
        
        # update thresholds
        if updateThresholds:
            self.updateThresholds_old()
    
    def _updateStrategy_dissim(self, oldData, newData, sub):
        # update the training data of a class when new data is available
        
        out = dict()
        sub = int(sub)
        # print 'compare the add', sub, self.meansubs
        
        if sub in self.meansubs:
            # print 'SAME'
            label = numpy.argmax(self.meansubs == sub)
            leadavg = dict()
            inc = 0
            for lead in self.leads:
                # print '\tstrategy', label, inc, lead
                # concatenate old with new data
                out[lead] = numpy.concatenate([newData[lead], oldData[lead]])
                # update the average ECG segments
                if len(out[lead]) > 1:
                    if self.median:
                        leadavg[lead] = numpy.median(out[lead], 0)
                    else:
                        leadavg[lead] = numpy.mean(out[lead], 0)
                else:
                    leadavg[lead] = out[lead]
                # print 'before', self.leadavglist[label*self.leadlen+inc, 0:10], numpy.shape(self.leadavglist)
                self.leadavglist[label*self.leadlen+inc] = leadavg[lead]
                # print 'after', self.leadavglist[label*self.leadlen+inc, 0:10], numpy.shape(self.leadavglist)
                inc += 1
                
            # update the feature cannon in relation to new ECG averages
            for i in xrange(self.nbSubjects):
                currdata = self.io_load(i)
                print len(currdata[self.reflead]), len(self.feat_cannon), numpy.shape(self.feat_cannon[self.cannon_label == i, label*self.leadlen:label*self.leadlen+self.leadlen])
                newdists = dist.cdist(currdata[self.reflead], self.leadavglist[label*self.leadlen:label*self.leadlen+self.leadlen], self.featmetric)
                self.feat_cannon[self.cannon_label == i, label*self.leadlen:label*self.leadlen+self.leadlen] = newdists
        else:
            # print 'NOT SAME'
            for lead in self.leads:
                # print '\tstrategy', lead
                # concatenate old with new data
                out[lead] = numpy.concatenate([newData[lead], oldData[lead]])
                
        return out
    
    def _addsubject(self, data, label):
        # FOR NOW IT IS ONLY POSSIBLE TO ADD NON-REFERENCE SUBJECTS ALRIGHT?
        
        leadavg = dict()
        for lead in self.leads:
            # print '\tstrategy', lead
            # update the average ECG segments
            if len(data[lead]) > 1:
                if self.median:
                    leadavg[lead] = numpy.median(data[lead], 0)
                else:
                    leadavg[lead] = numpy.mean(data[lead], 0)
            else:
                leadavg[lead] = data[lead]
            self.leadavglist = numpy.concatenate((self.leadavglist, numpy.array([leadavg[lead]])), 0)
        # print 'length of all', numpy.shape(self.leadavglist)
        
        # update the feature cannon in relation to new ECG averages
        newdists = []
        for i in xrange(self.nbSubjects):
            currdata = self.io_load(i)
            # print len(currdata[self.reflead]), numpy.shape(self.feat_cannon)
            newdists_buff = dist.cdist(currdata[self.reflead], self.leadavglist[label*self.leadlen:label*self.leadlen+self.leadlen], self.featmetric)
            newdists.append(newdists_buff)
            # print '\tnewdists', numpy.shape(newdists_buff)
        newdists = numpy.concatenate(newdists, 0)
        # print 'newfinal', numpy.shape(newdists)
        self.feat_cannon = numpy.concatenate((self.feat_cannon, newdists), 1)
    
    def re_train(self, data):
        raise NotImplementedError, "Method to re-train is not implemented."
    
    def _compute(self, data, label='all', feat=False):
        """
        Computes all distances between points in data and in the training set

        Inputs:

            data - the data dictionary (each entry is a data matrix defined by the lead), it is either composed by
            dissimilarity between segments or raw ECG segments to be transformed here according to...

            feat - flag which dictates whether the features are extracted or not

            label - the training data labels with which to compare input data

        Outputs

            dictionary with distance matrix (number of data window rows & number of train data columns) and
            corresponding labels

        """
        # print 'HELLO'
        datalen = len(data[self.reflead])
        if label == 'all':
            clabels = sorted(self.subject2label.values())
        else:
            clabels = [label]
        if not feat:
            feats = self.extract_feats(data)
        else:
            feats = numpy.copy(data)
            
        # print 'shape:', datalen, self.leadlen, numpy.shape(feats)
        
        # class labels
        # print '\tCLABELS', clabels
        # compute distances ---FRANCIS--- WTF?
        # if len(clabels) > (4 * config.numberProcesses):
        #     # run in multiprocessing
        #     workQ = parallel.getQueue()
        #     for lbl in clabels:
        #         workQ.put({
        #                    'testData': data,
        #                    'trainData': self.io_load(lbl),      # isto vao ser os dados transformados
        #                    'taskid': lbl,
        #                    })
        #     # run in multiprocess (distances mode)
        #     store = parallel.getDictManager()
        #     parallel.runMultiprocess(workQ, store, mode='distances')
        #
        #     # amass results
        #     dists = numpy.concatenate([store[lbl]['distances'] for lbl in clabels], axis=1)
        #     trainLabels = numpy.concatenate([store[lbl]['labels'] for lbl in clabels], axis=1)
        # else:
            # run sequentially
        dists = []
        trainLabels = []
        for lbl in clabels:
            trainData = self.feat_cannon[self.cannon_label == lbl, :]
            # print '---', lbl, numpy.shape(self.feat_cannon), numpy.shape(trainData), '---'
            # labels
            trainLabels.append(lbl * numpy.ones((datalen, len(trainData)), dtype='int'))
            # distances
            dists.append(dist.cdist(feats, trainData, self.metric))
            
        # amass results
        dists = numpy.concatenate(dists, axis=1)
        trainLabels = numpy.concatenate(trainLabels, axis=1)
        # print '\n\tdistsPRE\n', dists, numpy.shape(dists)
        # print '\n\ttrainLabelsPRE\n', trainLabels, numpy.shape(trainLabels)
        
        # sort
        dists, trainLabels = self._sort(dists, trainLabels)
        # print '\n\tdistsPOST\n', dists[:, 0:6], numpy.shape(dists)
        # print '\n\ttrainLabelsPOST\n', trainLabels[:, 0:6], numpy.shape(trainLabels)
        return dists, trainLabels
    
    def autoRejectionThresholds(self):
        # generate thresholds automatically
        
        if self._autoThresholds is not None:
            return self._autoThresholds
        
        if self.metric == 'cosine':
            self._autoThresholds = numpy.linspace(self.minThr, 1., 100)
            return self._autoThresholds
        
        maxD = []
        for _ in xrange(3):
            # repeat
            for label in sorted(self.subject2label.values()):
                # randomly select one sample
                aux = self.io_load(label)
                ind = config.random.randint(0, len(aux[self.reflead+'-raw']), 1)
                obs = dict()
                for lead in self.leads:
                    obs[lead] = aux[lead][ind, :]
                    
                # compute distances
                dists, _ = self._compute(obs)
                # print 'CURR_MAX:', numpy.max(dists)
                maxD.append(numpy.max(dists))
                
        # max distance
        maxD = 1.5 * numpy.max(maxD)
        
        # rejection thresholds to test
        self._autoThresholds = numpy.linspace(self.minThr, maxD, 100)
        
        return self._autoThresholds


class Dissimilarity(KNN):
    """
    Class to extract multilead distances subjectwise
    
    Performs kNN for classification based on those distances
    """
    
    NAME = 'dissimilarityClassifier'
    EXT = '.hdf5'
    EER_IDX = 0
    
    def __init__(self, io=None, k=3, metric='euclidean', featmetric='euclidean', leads=('I', 'II', 'III'), reflead='I',
                 median=False, **kwargs):
        # run parent __init__
        super(Dissimilarity, self).__init__(io=io, k=k)
        
        # algorithm self things
        self.k = k
        self.leads = leads
        self.leadlen = len(leads)
        self.reflead = reflead
        self.median = median
        self.percent = 0.1
        
        # minimum threshold
        self.minThr = 10 * numpy.finfo('float').eps
        
        # choose metric
        if metric == 'euclidean':
            self.metric = 'euclidean'
        elif metric == 'cosine':
            self.metric = 'cosine'
        else:
            raise NotImplementedError("Metric %s unknown." % metric)
        
        # choose metric
        if featmetric == 'euclidean':
            self.featmetric = 'euclidean'
        elif featmetric == 'cosine':
            self.featmetric = 'cosine'
        else:
            raise NotImplementedError("Metric %s unknown." % featmetric)
    
    def _fileIO_load(self, label):
        # load label from file
        
        fmt = '%s-%s'
        data = {}
        for l in self.leads:
            data[l] = datamanager.h5Load(self.iofile, fmt % (label, l))
        
        return data
    
    def _fileIO_save(self, label, data):
        # save data with label to file
        
        fmt = '%s-%s'
        for l in self.leads:
            datamanager.h5Store(self.iofile, fmt % (label, l), data[l])
    
    def _reset(self):
        # reset the classifier
        self.subject2label = SubjectDict()
        self.nbSubjects = 0
        self.is_trained = False
        self.thresholds = {}
        self._autoThresholds = None
        self.avglist = []
        self.leadavglist = []
    
    def _snapshot(self):
        # snapshot of classifier structures
        subject2label = self.subject2label
        nbSubjects = self.nbSubjects
        avglist = self.avglist
        leadavglist = self.leadavglist
        
        return subject2label, nbSubjects, avglist, leadavglist
    
    def dict2list(self, dictionary):
        """
        Gets a dictionary as input and returns a list
        
        Needed so as to join all average ECGs leadwise and feed them to scipy.spatial.distance.cdist()
        """
        
        datalist = []
        for lead in self.leads:
            datalist.append(dictionary[lead])
            
        return datalist
    
    def extract_feats(self, data, label=None):
        """
        data - dictionary whose entries are discriminated by leads; entries have a matrix of ECG segments
        """

        feats = dist.cdist(data[self.reflead], self.leadavglist, self.featmetric)
        
        return feats
    
    def _avgmaker(self, subjects, data):

        avglist = []
        leadavglist = []
        
        for i in subjects:
            leadavg = dict()
            for lead in self.leads:
                # print '\tlead', lead, len(data[i][lead]), i
                if len(data[i][lead]) > 1:
                    if self.median:
                        leadavg[lead] = numpy.median(data[i][lead], 0)
                    else:
                        leadavg[lead] = numpy.mean(data[i][lead], 0)
                else:
                    leadavg[lead] = data[i][lead]
                # fig = pylab.figure()
                # ax = fig.add_subplot(111)
                # ax.plot(leadavg[lead])
                # pylab.suptitle('Lead'+lead+'sub'+i)
                # raw_input()
                # pylab.close()
                leadavglist.append(leadavg[lead])
            avglist.append(leadavg)
            
        self.avglist = avglist
        self.leadavglist = numpy.array(leadavglist)
        # print 'length of all', numpy.shape(self.leadavglist)
    
    def train(self, data=None, updateThresholds=True):
        # data is {subject: features (array)}
        # print len(data), len(data[0]), len(data[0][0]), len(data[0][1])
        
        # check inputs
        if data is None:
            raise TypeError("Please provide input data.")
        
        # check if classifier was already trained
        if self.is_trained:
            # this is a retrain
            # print 'RETRAIN'
            self.re_train(data)
        else:
            # get subjects
            subjects = data.keys()
            # print '\t WTF', subjects
            self.nbSubjects = len(subjects)
            dissim_cannon = []
            cannon_label = []
            # create average ECG segments
            self._avgmaker(subjects, data)
            
            # determine classifier mode
            if self.nbSubjects == 0:
                raise ValueError("Please provide input data - empty dict.")
            else:
                for i in xrange(self.nbSubjects):
                    # build dicts
                    sub = subjects[i]
                    self.subject2label[sub] = i
                    # print '\t*****', i, sub, self.subject2label, '*****'
                    
                    # extract dissimilarities
                    feats = self.extract_feats(data[sub])
                    
                    # build feature cannons
                    dissim_cannon.append(feats)
                    cannon_label.append(numpy.ones(len(data[sub][self.reflead]), dtype='int64')*i)
                    
                    # save data
                    self.io_save(i, data[sub])
                    
            # concatenate cannon and label
            self.feat_cannon = numpy.concatenate(dissim_cannon, axis=0)
            self.cannon_label = numpy.concatenate(cannon_label, axis=0)
            # print '--->', numpy.shape(self.feat_cannon), numpy.shape(self.cannon_label)
            # print self.feat_cannon
            # print self.cannon_label
            
        self.is_trained = True
        
        # update thresholds
        if updateThresholds:
            self.updateThresholds_old()
    
    def _updateStrategy_dissim(self, oldData, newData, label):
        # update the training data of a class when new data is available
        
        out = dict()
        leadavg = dict()
        inc = 0
        for lead in self.leads:
            # print '\tstrategy', label, inc, lead
            # concatenate old with new data
            out[lead] = numpy.concatenate([newData[lead], oldData[lead]])
            # update the average ECG segments
            if len(out[lead]) > 1:
                if self.median:
                    leadavg[lead] = numpy.median(out[lead], 0)
                else:
                    leadavg[lead] = numpy.mean(out[lead], 0)
            else:
                leadavg[lead] = out[lead]
            # print 'before', self.leadavglist[label*self.leadlen+inc, 0:10], numpy.shape(self.leadavglist)
            self.leadavglist[label*self.leadlen+inc] = leadavg[lead]
            # print 'after', self.leadavglist[label*self.leadlen+inc, 0:10], numpy.shape(self.leadavglist)
            inc += 1
            
        self.avglist[label] = leadavg
        
        # update the feature cannon in relation to new ECG averages
        for i in xrange(self.nbSubjects):
            currdata = self.io_load(i)
            # print len(currdata[self.reflead]), len(self.feat_cannon), numpy.shape(self.feat_cannon[self.cannon_label == i, label*self.leadlen:label*self.leadlen+self.leadlen])
            newdists = dist.cdist(currdata[self.reflead], self.leadavglist[label*self.leadlen:label*self.leadlen+self.leadlen], self.featmetric)
            self.feat_cannon[self.cannon_label == i, label*self.leadlen:label*self.leadlen+self.leadlen] = newdists
            
        return out
    
    def _addsubject(self, data, label):
        leadavg = dict()
        for lead in self.leads:
            # print '\tstrategy', lead
            # update the average ECG segments
            if len(data[lead]) > 1:
                if self.median:
                    leadavg[lead] = numpy.median(data[lead], 0)
                else:
                    leadavg[lead] = numpy.mean(data[lead], 0)
            else:
                leadavg[lead] = data[lead]
            self.leadavglist = numpy.concatenate((self.leadavglist, numpy.array([leadavg[lead]])), 0)
        self.avglist.append(leadavg)
        # print 'length of all', numpy.shape(self.leadavglist)
        
        # update the feature cannon in relation to new ECG averages
        newdists = []
        for i in xrange(self.nbSubjects):
            currdata = self.io_load(i)
            # print len(currdata[self.reflead]), numpy.shape(self.feat_cannon)
            newdists_buff = dist.cdist(currdata[self.reflead], self.leadavglist[label*self.leadlen:label*self.leadlen+self.leadlen], self.featmetric)
            newdists.append(newdists_buff)
            # print '\tnewdists', numpy.shape(newdists_buff)
        newdists = numpy.concatenate(newdists, 0)
        # print 'newfinal', numpy.shape(newdists)
        self.feat_cannon = numpy.concatenate((self.feat_cannon, newdists), 1)
    
    def re_train(self, data):
        raise NotImplementedError, "Method to re-train is not implemented."
    
    def _compute(self, data, label='all', feat=False):
        """
        Computes all distances between points in data and in the training set
        
        Inputs:
            
            data - the data dictionary (each entry is a data matrix defined by the lead), it is either composed by
            dissimilarity between segments or raw ECG segments to be transformed here according to...
            
            feat - flag which dictates whether the features are extracted or not
            
            label - the training data labels with which to compare input data
        
        Outputs
    
            dictionary with distance matrix (number of data window rows & number of train data columns) and
            corresponding labels
        
        """
        
        datalen = len(data[self.reflead])
        if label == 'all':
            clabels = sorted(self.subject2label.values())
        else:
            clabels = [label]
        if not feat:
            feats = self.extract_feats(data)
        else:
            feats = numpy.copy(data)
        
        # print 'shape:', datalen, self.leadlen, numpy.shape(feats)
        
        # class labels
        # print '\tCLABELS', clabels
        # compute distances ---FRANCIS--- WTF?
        # if len(clabels) > (4 * config.numberProcesses):
        #     # run in multiprocessing
        #     workQ = parallel.getQueue()
        #     for lbl in clabels:
        #         workQ.put({
        #                    'testData': data,
        #                    'trainData': self.io_load(lbl),      # isto vao ser os dados transformados
        #                    'taskid': lbl,
        #                    })
        #     # run in multiprocess (distances mode)
        #     store = parallel.getDictManager()
        #     parallel.runMultiprocess(workQ, store, mode='distances')
        #
        #     # amass results
        #     dists = numpy.concatenate([store[lbl]['distances'] for lbl in clabels], axis=1)
        #     trainLabels = numpy.concatenate([store[lbl]['labels'] for lbl in clabels], axis=1)
        # else:
            # run sequentially
        dists = []
        trainLabels = []
        for lbl in clabels:
            trainData = self.feat_cannon[self.cannon_label == lbl, :]
            # print '---', lbl, numpy.shape(self.feat_cannon), numpy.shape(trainData), '---'
            # labels
            trainLabels.append(lbl * numpy.ones((datalen, len(trainData)), dtype='int'))
            # distances
            dists.append(dist.cdist(feats, trainData, self.metric))
        
        # amass results
        dists = numpy.concatenate(dists, axis=1)
        trainLabels = numpy.concatenate(trainLabels, axis=1)
        # print '\n\tdistsPRE\n', dists, numpy.shape(dists)
        # print '\n\ttrainLabelsPRE\n', trainLabels, numpy.shape(trainLabels)
        
        # sort
        dists, trainLabels = self._sort(dists, trainLabels)
        # print '\n\tdistsPOST\n', dists[:, 0:6], numpy.shape(dists)
        # print '\n\ttrainLabelsPOST\n', trainLabels[:, 0:6], numpy.shape(trainLabels)
        return dists, trainLabels
    
    def autoRejectionThresholds(self):
        # generate thresholds automatically
        
        if self._autoThresholds is not None:
            return self._autoThresholds
        
        if self.metric == 'cosine':
            self._autoThresholds = numpy.linspace(self.minThr, 1., 100)
            return self._autoThresholds
        
        maxD = []
        for _ in xrange(3):
            # repeat
            for label in sorted(self.subject2label.values()):
                # randomly select one sample
                aux = self.io_load(label)
                ind = config.random.randint(0, len(aux[self.reflead]), 1)
                obs = dict()
                for lead in self.leads:
                    obs[lead] = aux[lead][ind, :]
                    
                # compute distances
                dists, _ = self._compute(obs)
                # print 'CURR_MAX:', numpy.max(dists)
                maxD.append(numpy.max(dists))
        
        # max distance
        maxD = 1.5 * numpy.max(maxD)
        
        # rejection thresholds to test
        self._autoThresholds = numpy.linspace(self.minThr, maxD, 100)
        
        return self._autoThresholds


class DissimilaritySimple(Dissimilarity):
    """
    Class to extract multilead distances subjectwise
    
    Performs kNN for classification based on those distances
    """
    
    NAME = 'dissimilaritySimpleClassifier'
    EXT = '.hdf5'
    EER_IDX = 0
    
    def __init__(self, io=None, k=3, metric='euclidean', featmetric='euclidean', leads=('I', 'II', 'III'), reflead='I', median=False,
                 tn=5, **kwargs):
        # run parent __init__
        super(DissimilaritySimple, self).__init__(io=io, k=k, metric=metric, featmetric=featmetric, leads=leads,
                                                  median=median, reflead=reflead, **kwargs)
        
        self.tn = tn
        self.avglist = []
    
    def extract_feats(self, data, label=None):
        """
        data - dictionary whose entries are discriminated by leads; entries have a matrix of ECG segments
        """
        if label is None:
            raise TypeError("Please give label")
        feats = dist.cdist(data[self.reflead], self.leadavglist[label*self.leadlen*self.tn:(label+1)*self.leadlen*self.tn], self.featmetric)
        # print '\tfeats', numpy.shape(feats)
        
        return feats
    
    def _avgmaker(self, subjects, data):
        
        leadavglist = []
        
        # distance to templates (organization is made as in C leads for template i, C leads for template i+1 (...))
        for i in subjects:
            leadavg = dict()
            for lead in self.leads:
                for template in xrange(self.tn):
                    # print '\tsubject:', i, 'lead', lead, 'template', template, 'len', len(data[i][lead][template])
                    if len(data[i][lead + '-tpl'][template]) > 1:
                        leadavg[lead] = data[i][lead + '-tpl'][template]
                    else:
                        # print 'BUG IN LEADAVG'
                        pass
                    leadavglist.append(leadavg[lead])
        
        self.leadavglist = numpy.array(leadavglist)
        # print 'length of all', numpy.shape(self.leadavglist)
    
    def train(self, data=None, updateThresholds=True):
        # data is {subject: features (array)}
        
        # check inputs
        if data is None:
            raise TypeError("Please provide input data.")
        
        # check if classifier was already trained
        if self.is_trained:
            # this is a retrain
            # print 'RETRAIN'
            self.re_train(data)
        else:
            # get subjects
            subjects = data.keys()
            # print '\t WTF', subjects
            self.nbSubjects = len(subjects)
            dissim_cannon = []
            cannon_label = []
            # create average ECG segments
            self._avgmaker(subjects, data)
            
            # determine classifier mode
            if self.nbSubjects == 0:
                raise ValueError("Please provide input data - empty dict.")
            else:
                for i in xrange(self.nbSubjects):
                    # build dicts
                    sub = subjects[i]
                    self.subject2label[sub] = i
                    # print '\t*****', i, sub, self.subject2label, '*****'
                    
                    # extract dissimilarities
                    feats = self.extract_feats(data[sub], i)
                    
                    # build feature cannons
                    dissim_cannon.append(feats)
                    cannon_label.append(numpy.ones(len(data[sub][self.reflead]), dtype='int64')*i)
                    
                    # save data
                    self.io_save(i, data[sub])
            
            # concatenate cannon and label
            self.feat_cannon = numpy.concatenate(dissim_cannon, axis=0)
            self.cannon_label = numpy.concatenate(cannon_label, axis=0)
            # print '--->', numpy.shape(self.feat_cannon), numpy.shape(self.cannon_label)
            # print self.feat_cannon
            # print self.cannon_label
        
        self.is_trained = True
        
        # update thresholds
        if updateThresholds:
            self.updateThresholds_old()
    
    def _updateStrategy_dissim(self, oldData, newData, label):
        # update the training data of a class when new data is available
        
        out = dict()
        leadavg = dict()
        inc = 0
        for lead in self.leads:
            # print '\tstrategy', label, inc, lead
            # concatenate old with new data
            out[lead] = numpy.concatenate([newData[lead], oldData[lead]])
            # update the average ECG segments
            if len(out[lead]) > 1:
                if self.median:
                    leadavg[lead] = numpy.median(out[lead], 0)
                else:
                    leadavg[lead] = numpy.mean(out[lead], 0)
            else:
                leadavg[lead] = out[lead]
            # print 'before', self.leadavglist[label*self.leadlen+inc, 0:10], numpy.shape(self.leadavglist)
            self.leadavglist[label*self.leadlen+inc] = leadavg[lead]
            # print 'after', self.leadavglist[label*self.leadlen+inc, 0:10], numpy.shape(self.leadavglist)
            inc += 1
        
        # update the feature cannon in relation to new ECG averages
        currdata = self.io_load(label)
        # print len(currdata[self.reflead]), numpy.shape(self.feat_cannon), numpy.shape(self.feat_cannon[self.cannon_label == label, :])
        newdists = dist.cdist(currdata[self.reflead], self.leadavglist[label*self.leadlen:label*self.leadlen+self.leadlen], self.featmetric)
        self.feat_cannon[self.cannon_label == label, :] = newdists
        
        return out
    
    def _addsubject(self, data, label):
        leadavg = dict()
        for lead in self.leads:
            # print '\tstrategy', lead
            # update the average ECG segments
            if len(data[lead]) > 1:
                if self.median:
                    leadavg[lead] = numpy.median(data[lead], 0)
                else:
                    leadavg[lead] = numpy.mean(data[lead], 0)
            else:
                leadavg[lead] = data[lead]
            self.leadavglist = numpy.concatenate((self.leadavglist, numpy.array([leadavg[lead]])), 0)
        # print 'length of all', numpy.shape(self.leadavglist)
    
    def re_train(self, data):
        raise NotImplementedError, "Method to re-train is not implemented."
    
    def _compute(self, data, label='all', feat=False):
        """
        Computes all distances between points in data and in the training set
        
        Inputs:
        
            data - the data dictionary (each entry is a data matrix defined by the lead), it is either composed by
            dissimilarity between segments or raw ECG segments to be transformed here according to...
            
            feat - flag which dictates whether the features are extracted or not
            
            label - the training data labels with which to compare input data
            
        Outputs
            
            dictionary with distance matrix (number of data window rows & number of train data columns) and
            corresponding labels
        
        """
        
        datalen = len(data[self.reflead])
        if label == 'all':
            clabels = sorted(self.subject2label.values())
        else:
            clabels = [label]
            
        # print 'shape start:', datalen, self.leadlen
        
        # class labels
        # print '\tCLABELS', clabels
        # compute distances ---FRANCIS--- WTF?
        # if len(clabels) > (4 * config.numberProcesses):
        #     # run in multiprocessing
        #     workQ = parallel.getQueue()
        #     for lbl in clabels:
        #         workQ.put({
        #                    'testData': data,
        #                    'trainData': self.io_load(lbl),      # isto vao ser os dados transformados
        #                    'taskid': lbl,
        #                    })
        #     # run in multiprocess (distances mode)
        #     store = parallel.getDictManager()
        #     parallel.runMultiprocess(workQ, store, mode='distances')
        #
        #     # amass results
        #     dists = numpy.concatenate([store[lbl]['distances'] for lbl in clabels], axis=1)
        #     trainLabels = numpy.concatenate([store[lbl]['labels'] for lbl in clabels], axis=1)
        # else:
            # run sequentially
        dists = []
        trainLabels = []
        for lbl in clabels:
            feats = self.extract_feats(data, lbl)
            # print '\tfeats', numpy.shape(feats)
            trainData = self.feat_cannon[self.cannon_label == lbl, :]
            # print '---', lbl, numpy.shape(self.feat_cannon), numpy.shape(trainData), '---'
            # labels
            trainLabels.append(lbl * numpy.ones((datalen, len(trainData)), dtype='int'))
            # distances
            dists.append(dist.cdist(feats, trainData, self.metric))
            
        # amass results
        dists = numpy.concatenate(dists, axis=1)
        trainLabels = numpy.concatenate(trainLabels, axis=1)
        # print '\n\tdistsPRE\n', dists, numpy.shape(dists)
        # print '\n\ttrainLabelsPRE\n', trainLabels, numpy.shape(trainLabels)
        
        # sort
        dists, trainLabels = self._sort(dists, trainLabels)
        # print '\n\tdistsPOST\n', dists[:, 0:6], numpy.shape(dists)
        # print '\n\ttrainLabelsPOST\n', trainLabels[:, 0:6], numpy.shape(trainLabels)
        return dists, trainLabels


class Dissimilarity_old(KNN):
    """
    Class to extract multilead distances subjectwise

    Performs kNN for classification based on those distances
    """

    NAME = 'dissimilarityClassifier'
    EXT = '.hdf5'
    EER_IDX = 0

    def __init__(self, io=None, k=3, metric='euclidean', featmetric='euclidean', leads=('I', 'II', 'III'), reflead='I',
                 **kwargs):
        # run parent __init__
        super(Dissimilarity_old, self).__init__(io=io, k=k)

        # algorithm self things
        self.k = k
        self.leads = leads
        self.leadlen = len(leads)
        self.reflead = reflead

        # minimum threshold
        self.minThr = 10 * numpy.finfo('float').eps

        # choose metric
        if metric == 'euclidean':
            self.metric = 'euclidean'
        elif metric == 'cosine':
            self.metric = 'cosine'
        else:
            raise NotImplementedError("Metric %s unknown." % metric)

        # choose metric
        if featmetric == 'euclidean':
            self.featmetric = 'euclidean'
        elif featmetric == 'cosine':
            self.featmetric = 'cosine'
        else:
            raise NotImplementedError("Metric %s unknown." % featmetric)
    
    def _fileIO_load(self, label):
        # load label from file
        
        fmt = '%s-%s'
        data = {}
        for l in self.leads:
            data[l] = datamanager.h5Load(self.iofile, fmt % (label, l))
        
        return data
    
    def _fileIO_save(self, label, data):
        # save data with label to file
        
        fmt = '%s-%s'
        for l in self.leads:
            datamanager.h5Store(self.iofile, fmt % (label, l), data[l])
    
    def _reset(self):
        # reset the classifier
        self.subject2label = SubjectDict()
        self.nbSubjects = 0
        self.is_trained = False
        self.thresholds = {}
        self._autoThresholds = None
        self.avglist = []
        self.leadavglist = []

    def _snapshot(self):
        # snapshot of classifier structures
        subject2label = self.subject2label
        nbSubjects = self.nbSubjects
        avglist = self.avglist
        leadavglist = self.leadavglist

        return subject2label, nbSubjects, avglist, leadavglist

    def dict2list(self, dictionary):

        """
            Gets a dictionary as input and returns a list

            Needed so as to join all average ECGs leadwise and feed them to scipy.spatial.distance.cdist()
        """

        datalist = []
        for lead in self.leads:
            datalist.append(dictionary[lead])

        return datalist

    def extract_feats(self, data, label=None):

        """
            data - dictionary whose entries are discriminated by leads; entries have a matrix of ECG segments
        """

        feats = dist.cdist(data[self.reflead], self.leadavglist, self.featmetric)

        return feats

    def _avgmaker(self, subjects, data):

        avglist = []
        leadavglist = []

        for i in subjects:
            leadavg = dict()
            for lead in self.leads:
                # print '\tlead', lead, len(data[i][lead]), i
                if len(data[i][lead]) > 1:
                    leadavg[lead] = numpy.mean(data[i][lead], 0)
                else:
                    leadavg[lead] = data[i][lead]
                leadavglist.append(leadavg[lead])
            avglist.append(leadavg)

        self.avglist = avglist
        self.leadavglist = numpy.array(leadavglist)
        # print 'length of all', numpy.shape(self.leadavglist)

    def train(self, data=None, updateThresholds=True):
        # data is {subject: features (array)}
        # print len(data), len(data[0]), len(data[0][0]), len(data[0][1])

        # check inputs
        if data is None:
            raise TypeError("Please provide input data.")

        # check if classifier was already trained
        if self.is_trained:
            # this is a retrain
            # print 'RETRAIN'
            self.re_train(data)
        else:
            # get subjects
            subjects = data.keys()
            # print '\t WTF', subjects
            self.nbSubjects = len(subjects)
            dissim_cannon = []
            cannon_label = []
            # create average ECG segments
            self._avgmaker(subjects, data)

            # determine classifier mode
            if self.nbSubjects == 0:
                raise ValueError("Please provide input data - empty dict.")
            else:
                for i in xrange(self.nbSubjects):
                    # build dicts
                    sub = subjects[i]
                    self.subject2label[sub] = i
                    # print '\t*****', i, sub, self.subject2label, '*****'

                    # extract dissimilarities
                    feats = self.extract_feats(data[sub])

                    # build feature cannons
                    dissim_cannon.append(feats)
                    cannon_label.append(numpy.ones(len(data[sub][self.reflead]), dtype='int64')*i)

                    # save data
                    self.io_save(i, data[sub])

            # concatenate cannon and label
            self.feat_cannon = numpy.concatenate(dissim_cannon, axis=0)
            self.cannon_label = numpy.concatenate(cannon_label, axis=0)
            # print '--->', numpy.shape(self.feat_cannon), numpy.shape(self.cannon_label)
            # print self.feat_cannon
            # print self.cannon_label

        self.is_trained = True

        # update thresholds
        if updateThresholds:
            self.updateThresholds()

    def _updateStrategy_dissim(self, oldData, newData, label):
        # update the training data of a class when new data is available

        out = dict()
        leadavg = dict()
        inc = 0
        for lead in self.leads:
            # print '\tstrategy', label, inc, lead
            # concatenate old with new data
            out[lead] = numpy.concatenate([newData[lead], oldData[lead]])
            # update the average ECG segments
            if len(out[lead]) > 1:
                leadavg[lead] = numpy.mean(out[lead], 0)
            else:
                leadavg[lead] = out[lead]
            # print 'before', self.leadavglist[label*self.leadlen+inc, 0:10], numpy.shape(self.leadavglist)
            self.leadavglist[label*self.leadlen+inc] = leadavg[lead]
            # print 'after', self.leadavglist[label*self.leadlen+inc, 0:10], numpy.shape(self.leadavglist)
            inc += 1

        self.avglist[label] = leadavg

        # update the feature cannon in relation to new ECG averages
        for i in xrange(self.nbSubjects):
            currdata = self.io_load(i)
            # print len(currdata[self.reflead]), len(self.feat_cannon), numpy.shape(self.feat_cannon[self.cannon_label == i, label*self.leadlen:label*self.leadlen+self.leadlen])
            newdists = dist.cdist(currdata[self.reflead], self.leadavglist[label*self.leadlen:label*self.leadlen+self.leadlen], self.featmetric)
            self.feat_cannon[self.cannon_label == i, label*self.leadlen:label*self.leadlen+self.leadlen] = newdists

        return out

    def _addsubject(self, data, label):
        leadavg = dict()
        for lead in self.leads:
            # print '\tstrategy', lead
            # update the average ECG segments
            if len(data[lead]) > 1:
                leadavg[lead] = numpy.mean(data[lead], 0)
            else:
                leadavg[lead] = data[lead]
            self.leadavglist = numpy.concatenate((self.leadavglist, numpy.array([leadavg[lead]])), 0)
        self.avglist.append(leadavg)
        # print 'length of all', numpy.shape(self.leadavglist)

        # update the feature cannon in relation to new ECG averages
        newdists = []
        for i in xrange(self.nbSubjects):
            currdata = self.io_load(i)
            # print len(currdata[self.reflead]), numpy.shape(self.feat_cannon)
            newdists_buff = dist.cdist(currdata[self.reflead], self.leadavglist[label*self.leadlen:label*self.leadlen+self.leadlen], self.featmetric)
            newdists.append(newdists_buff)
            # print '\tnewdists', numpy.shape(newdists_buff)
        newdists = numpy.concatenate(newdists, 0)
        # print 'newfinal', numpy.shape(newdists)
        self.feat_cannon = numpy.concatenate((self.feat_cannon, newdists), 1)

    def re_train(self, data):
        # data is {subject: features (array)}
        ### user

        for sub in sorted( data.iterkeys() ):
            # print '---', sub, '---'
            if sub in self.subject2label:
                # existing subject
                if data[sub] is not None:
                    # change subject's data
                    label = self.subject2label[sub]
                    # print 'label', label

                    # update templates
                    existing = self.io_load(label)
                    aux = self._updateStrategy_dissim(existing, data[sub], label)
                    # print 'concatdata', numpy.shape(existing[self.reflead]), numpy.shape(data[sub][self.reflead]), numpy.shape(aux[self.reflead])

                    # extract features for the new segments
                    feats = self.extract_feats(data[sub])

                    # update cannon and its label
                    insertind = numpy.argmax(self.cannon_label >= label)
                    # print 'before', numpy.shape(self.feat_cannon), numpy.shape(feats), insertind
                    self.feat_cannon = numpy.insert(self.feat_cannon, insertind, feats, 0)
                    # print 'insert', numpy.shape(self.feat_cannon)
                    self.cannon_label = numpy.insert(self.cannon_label, insertind,
                                                     label*numpy.ones(len(data[sub][self.reflead]), dtype='int64'), 0)
                    # print '----------> UPDATE', sub
                    raw_input('')
                    # save data
                    self.io_save(label, aux)
                else:
                    # delete subject
                    if self.nbSubjects == 1:
                        # reset classifier to untrained
                        self._reset()
                    else:
                        # reorder the labels/subjects and models dicts
                        subject2label, nbSubjects, avglist, leadavglist = self._snapshot()
                        self._reset()

                        label = subject2label[sub]
                        # print 'sub+label', sub, label

                        # update cannon and its label
                        removeinds = numpy.where(self.cannon_label == label)[0]
                        self.feat_cannon = numpy.delete(self.feat_cannon, removeinds, axis=0)
                        self.cannon_label = numpy.delete(self.cannon_label, removeinds, axis=0)

                        # remove distances to deleted subjects
                        # print 'removal', numpy.arange(label*self.leadlen, label*self.leadlen+self.leadlen), \
                        #     numpy.shape(self.feat_cannon), numpy.shape(self.feat_cannon.T)
                        # print 'compare prev', self.feat_cannon[0:5, :]
                        cannon_buff = numpy.delete(self.feat_cannon.T,
                                                   numpy.arange(label*self.leadlen, label*self.leadlen+self.leadlen), 0)
                        self.feat_cannon = cannon_buff.T
                        # print 'compare', numpy.shape(self.feat_cannon), self.feat_cannon[0:5, :]

                        clabels = numpy.setdiff1d(numpy.unique(subject2label.values()),
                                                  [label], assume_unique=True)
                        self.nbSubjects = nbSubjects - 1

                        # print 'clabels', clabels
                        i = 0
                        for ii in xrange(len(clabels)):
                            # build dicts
                            sub = subject2label[:clabels[ii]]
                            self.subject2label[sub] = i
                            clabel_inds = numpy.where(self.cannon_label == clabels[ii])[0]
                            self.cannon_label[clabel_inds] = i
                            # print '\tlabels', clabels[ii], i, self.cannon_label[clabel_inds]
                            self.avglist.append(avglist[clabels[ii]])
                            self.leadavglist.append(leadavglist[clabels[ii]*self.leadlen:clabels[ii]*self.leadlen+self.leadlen])

                            # print '\tavg', avglist[clabels[ii]][self.reflead] == self.avglist[i][self.reflead]

                            # move data
                            self.io_save(i, self.io_load(clabels[ii]))

                            # update i
                            i += 1
                        self.leadavglist = numpy.concatenate(self.leadavglist, 0)
                        # print leadavglist[:, 0:10], '\n'
                        # print self.leadavglist[:, 0:10]
                        # print '----------> DELETE', sub, numpy.shape(self.feat_cannon), numpy.shape(self.leadavglist)
                        raw_input('')
            else:
                # new subject
                # add to dicts
                label = self.nbSubjects
                self.subject2label[sub] = label

                # add subject to avg dictionary and list
                # print 'compare start', self.feat_cannon[0:5, :]
                self._addsubject(data[sub], label)

                # extract feats
                feats = self.extract_feats(data[sub])

                # update cannon and its label
                # print 'compare middle', self.feat_cannon[0:5, :]
                # print 'adding', numpy.shape(self.feat_cannon), numpy.shape(feats)
                self.feat_cannon = numpy.vstack((self.feat_cannon, feats))
                self.cannon_label = numpy.hstack((self.cannon_label, numpy.ones(len(data[sub][self.reflead]),
                                                                                dtype='int64')*label))
                # print 'compare last', self.feat_cannon[0:5, :]
                # save data
                self.io_save(label, data[sub])

                # increment number of subjects
                self.nbSubjects += 1

                # print '----------> ADD', sub, numpy.shape(self.feat_cannon)
                raw_input('')

    def _compute(self, data, label='all', feat=False):
        """
            Computes all distances between points in data and in the training set

        Inputs:

            data - the data dictionary (each entry is a data matrix defined by the lead), it is either composed by
            dissimilarity between segments or raw ECG segments to be transformed here according to...

            feat - flag which dictates whether the features are extracted or not

            label - the training data labels with which to compare input data

        Outputs

            dictionary with distance matrix (number of data window rows & number of train data columns) and
            corresponding labels

        """

        datalen = len(data[self.reflead])
        if label == 'all':
            clabels = sorted(self.subject2label.values())
        else:
            clabels = [label]
        if not feat:
            feats = self.extract_feats(data)
        else:
            feats = numpy.copy(data)

        # print 'shape:', datalen, self.leadlen, numpy.shape(feats)

        # class labels
        # print '\tCLABELS', clabels
        # compute distances ---FRANCIS--- WTF?
        # if len(clabels) > (4 * config.numberProcesses):
        #     # run in multiprocessing
        #     workQ = parallel.getQueue()
        #     for lbl in clabels:
        #         workQ.put({
        #                    'testData': data,
        #                    'trainData': self.io_load(lbl),      # isto vao ser os dados transformados
        #                    'taskid': lbl,
        #                    })
        #     # run in multiprocess (distances mode)
        #     store = parallel.getDictManager()
        #     parallel.runMultiprocess(workQ, store, mode='distances')
        #
        #     # amass results
        #     dists = numpy.concatenate([store[lbl]['distances'] for lbl in clabels], axis=1)
        #     trainLabels = numpy.concatenate([store[lbl]['labels'] for lbl in clabels], axis=1)
        # else:
            # run sequentially
        dists = []
        trainLabels = []
        for lbl in clabels:
            trainData = self.feat_cannon[self.cannon_label == lbl, :]
            # print '---', lbl, numpy.shape(self.feat_cannon), numpy.shape(trainData), '---'
            # labels
            trainLabels.append(lbl * numpy.ones((datalen, len(trainData)), dtype='int'))
            # distances
            dists.append(dist.cdist(feats, trainData, self.metric))

        # amass results
        dists = numpy.concatenate(dists, axis=1)
        trainLabels = numpy.concatenate(trainLabels, axis=1)
        # print '\n\tdistsPRE\n', dists, numpy.shape(dists)
        # print '\n\ttrainLabelsPRE\n', trainLabels, numpy.shape(trainLabels)


        # sort
        dists, trainLabels = self._sort(dists, trainLabels)
        # print '\n\tdistsPOST\n', dists[:, 0:6], numpy.shape(dists)
        # print '\n\ttrainLabelsPOST\n', trainLabels[:, 0:6], numpy.shape(trainLabels)
        return dists, trainLabels

    def autoRejectionThresholds(self):
        # generate thresholds automatically

        if self._autoThresholds is not None:
            return self._autoThresholds

        if self.metric == 'cosine':
            self._autoThresholds = numpy.linspace(self.minThr, 1., 100)
            return self._autoThresholds

        maxD = []
        for _ in xrange(3):
            # repeat
            for label in sorted(self.subject2label.values()):
                # randomly select one sample
                aux = self.io_load(label)
                ind = config.random.randint(0, len(aux[self.reflead]), 1)
                obs = dict()
                for lead in self.leads:
                    obs[lead] = aux[lead][ind, :]

                # compute distances
                dists, _ = self._compute(obs)
                # print 'CURR_MAX:', numpy.max(dists)
                maxD.append(numpy.max(dists))

        # max distance
        maxD = 1.5 * numpy.max(maxD)

        # rejection thresholds to test
        self._autoThresholds = numpy.linspace(self.minThr, maxD, 100)

        return self._autoThresholds


class DissimilaritySimple_old(Dissimilarity_old):
    """
    Class to extract multilead distances subjectwise

    Performs kNN for classification based on those distances
    """

    NAME = 'dissimilaritySimpleClassifier'
    EXT = '.hdf5'
    EER_IDX = 0

    def __init__(self, io=None, k=3, metric='euclidean', featmetric='euclidean', leads=('I', 'II', 'III'), reflead='I',
                 **kwargs):
        # run parent __init__
        super(DissimilaritySimple_old, self).__init__(io=io, k=k, metric=metric, featmetric=featmetric, leads=leads,
                                            reflead=reflead)

    def extract_feats(self, data, label=None):

        """
            data - dictionary whose entries are discriminated by leads; entries have a matrix of ECG segments
        """
        if label is None:
            raise TypeError("Please give label")
        feats = dist.cdist(data[self.reflead], self.leadavglist[label*self.leadlen:label*self.leadlen+self.leadlen],
                           self.featmetric)
        # print '\tfeats', numpy.shape(feats)

        return feats

    def train(self, data=None, updateThresholds=True):
        # data is {subject: features (array)}

        # check inputs
        if data is None:
            raise TypeError("Please provide input data.")

        # check if classifier was already trained
        if self.is_trained:
            # this is a retrain
            # print 'RETRAIN'
            self.re_train(data)
        else:
            # get subjects
            subjects = data.keys()
            # print '\t WTF', subjects
            self.nbSubjects = len(subjects)
            dissim_cannon = []
            cannon_label = []
            # create average ECG segments
            self._avgmaker(subjects, data)

            # determine classifier mode
            if self.nbSubjects == 0:
                raise ValueError("Please provide input data - empty dict.")
            else:
                for i in xrange(self.nbSubjects):
                    # build dicts
                    sub = subjects[i]
                    self.subject2label[sub] = i
                    # print '\t*****', i, sub, self.subject2label, '*****'

                    # extract dissimilarities
                    feats = self.extract_feats(data[sub], i)

                    # build feature cannons
                    dissim_cannon.append(feats)
                    cannon_label.append(numpy.ones(len(data[sub][self.reflead]), dtype='int64')*i)

                    # save data
                    self.io_save(i, data[sub])

            # concatenate cannon and label
            self.feat_cannon = numpy.concatenate(dissim_cannon, axis=0)
            self.cannon_label = numpy.concatenate(cannon_label, axis=0)
            # print '--->', numpy.shape(self.feat_cannon), numpy.shape(self.cannon_label)
            # print self.feat_cannon
            # print self.cannon_label

        self.is_trained = True

        # update thresholds
        if updateThresholds:
            self.updateThresholds()

    def _updateStrategy_dissim(self, oldData, newData, label):
        # update the training data of a class when new data is available

        out = dict()
        leadavg = dict()
        inc = 0
        for lead in self.leads:
            # print '\tstrategy', label, inc, lead
            # concatenate old with new data
            out[lead] = numpy.concatenate([newData[lead], oldData[lead]])
            # update the average ECG segments
            if len(out[lead]) > 1:
                leadavg[lead] = numpy.mean(out[lead], 0)
            else:
                leadavg[lead] = out[lead]
            # print 'before', self.leadavglist[label*self.leadlen+inc, 0:10], numpy.shape(self.leadavglist)
            self.leadavglist[label*self.leadlen+inc] = leadavg[lead]
            # print 'after', self.leadavglist[label*self.leadlen+inc, 0:10], numpy.shape(self.leadavglist)
            inc += 1

        self.avglist[label] = leadavg

        # update the feature cannon in relation to new ECG averages
        currdata = self.io_load(label)
        # print len(currdata[self.reflead]), numpy.shape(self.feat_cannon), numpy.shape(self.feat_cannon[self.cannon_label == label, :])
        newdists = dist.cdist(currdata[self.reflead], self.leadavglist[label*self.leadlen:label*self.leadlen+self.leadlen], self.featmetric)
        self.feat_cannon[self.cannon_label == label, :] = newdists

        return out

    def _addsubject(self, data, label):
        leadavg = dict()
        for lead in self.leads:
            # print '\tstrategy', lead
            # update the average ECG segments
            if len(data[lead]) > 1:
                leadavg[lead] = numpy.mean(data[lead], 0)
            else:
                leadavg[lead] = data[lead]
            self.leadavglist = numpy.concatenate((self.leadavglist, numpy.array([leadavg[lead]])), 0)
        self.avglist.append(leadavg)
        # print 'length of all', numpy.shape(self.leadavglist)

    def re_train(self, data):
        # data is {subject: features (array)}
        ### user

        for sub in sorted( data.iterkeys() ):
            # print '---', sub, '---'
            if sub in self.subject2label:
                # existing subject
                if data[sub] is not None:
                    # change subject's data
                    label = self.subject2label[sub]
                    # print 'label', label

                    # update templates
                    existing = self.io_load(label)
                    aux = self._updateStrategy_dissim(existing, data[sub], label)
                    # print 'concatdata', numpy.shape(existing[self.reflead]), numpy.shape(data[sub][self.reflead]), numpy.shape(aux[self.reflead])

                    # extract features for the new segments
                    feats = self.extract_feats(data[sub], label)

                    # update cannon and its label
                    insertind = numpy.argmax(self.cannon_label >= label)
                    # print 'before', numpy.shape(self.feat_cannon), numpy.shape(feats), insertind
                    self.feat_cannon = numpy.insert(self.feat_cannon, insertind, feats, 0)
                    # print 'insert', numpy.shape(self.feat_cannon)
                    self.cannon_label = numpy.insert(self.cannon_label, insertind,
                                                     label*numpy.ones(len(data[sub][self.reflead]), dtype='int64'), 0)
                    # print '----------> UPDATE', sub
                    # raw_input('')
                    # save data
                    self.io_save(label, aux)
                else:
                    # delete subject
                    if self.nbSubjects == 1:
                        # reset classifier to untrained
                        self._reset()
                    else:
                        # reorder the labels/subjects and models dicts
                        subject2label, nbSubjects, avglist, leadavglist = self._snapshot()
                        self._reset()

                        label = subject2label[sub]
                        # print 'sub+label', sub, label

                        # update cannon and its label
                        # print 'removal', numpy.shape(self.feat_cannon), self.feat_cannon
                        removeinds = numpy.where(self.cannon_label == label)[0]
                        self.feat_cannon = numpy.delete(self.feat_cannon, removeinds, axis=0)
                        self.cannon_label = numpy.delete(self.cannon_label, removeinds, axis=0)
                        # print 'compare', numpy.shape(self.feat_cannon), self.feat_cannon

                        clabels = numpy.setdiff1d(numpy.unique(subject2label.values()),
                                                  [label], assume_unique=True)
                        self.nbSubjects = nbSubjects - 1

                        # print 'clabels', clabels
                        i = 0
                        for ii in xrange(len(clabels)):
                            # build dicts
                            sub = subject2label[:clabels[ii]]
                            self.subject2label[sub] = i
                            clabel_inds = numpy.where(self.cannon_label == clabels[ii])[0]
                            self.cannon_label[clabel_inds] = i
                            # print '\tlabels', clabels[ii], i, self.cannon_label[clabel_inds]
                            self.avglist.append(avglist[clabels[ii]])
                            self.leadavglist.append(leadavglist[clabels[ii]*self.leadlen:clabels[ii]*self.leadlen+self.leadlen])

                            # print '\tavg', avglist[clabels[ii]][self.reflead] == self.avglist[i][self.reflead]

                            # move data
                            self.io_save(i, self.io_load(clabels[ii]))

                            # update i
                            i += 1
                        self.leadavglist = numpy.concatenate(self.leadavglist, 0)
                        # print leadavglist[:, 0:10], '\n'
                        # print self.leadavglist[:, 0:10]
                        # print '----------> DELETE', sub, numpy.shape(self.feat_cannon), numpy.shape(self.leadavglist)
                        # raw_input('')
            else:
                # new subject
                # add to dicts
                label = self.nbSubjects
                self.subject2label[sub] = label

                # add subject to avg dictionary and list
                # print 'compare start', self.feat_cannon[0:5, :]
                self._addsubject(data[sub], label)

                # extract feats
                feats = self.extract_feats(data[sub], label)

                # update cannon and its label
                # print 'adding', numpy.shape(self.feat_cannon), numpy.shape(feats), self.feat_cannon
                self.feat_cannon = numpy.vstack((self.feat_cannon, feats))
                self.cannon_label = numpy.hstack((self.cannon_label, numpy.ones(len(data[sub][self.reflead]),
                                                                                dtype='int64')*label))
                # print 'compare last', self.feat_cannon
                # save data
                self.io_save(label, data[sub])

                # increment number of subjects
                self.nbSubjects += 1

                # print '----------> ADD', sub, numpy.shape(self.feat_cannon)
                # raw_input('')

    def _compute(self, data, label='all', feat=False):
        """
            Computes all distances between points in data and in the training set

        Inputs:

            data - the data dictionary (each entry is a data matrix defined by the lead), it is either composed by
            dissimilarity between segments or raw ECG segments to be transformed here according to...

            feat - flag which dictates whether the features are extracted or not

            label - the training data labels with which to compare input data

        Outputs

            dictionary with distance matrix (number of data window rows & number of train data columns) and
            corresponding labels

        """

        datalen = len(data[self.reflead])
        if label == 'all':
            clabels = sorted(self.subject2label.values())
        else:
            clabels = [label]

        # print 'shape start:', datalen, self.leadlen

        # class labels
        # print '\tCLABELS', clabels
        # compute distances ---FRANCIS--- WTF?
        # if len(clabels) > (4 * config.numberProcesses):
        #     # run in multiprocessing
        #     workQ = parallel.getQueue()
        #     for lbl in clabels:
        #         workQ.put({
        #                    'testData': data,
        #                    'trainData': self.io_load(lbl),      # isto vao ser os dados transformados
        #                    'taskid': lbl,
        #                    })
        #     # run in multiprocess (distances mode)
        #     store = parallel.getDictManager()
        #     parallel.runMultiprocess(workQ, store, mode='distances')
        #
        #     # amass results
        #     dists = numpy.concatenate([store[lbl]['distances'] for lbl in clabels], axis=1)
        #     trainLabels = numpy.concatenate([store[lbl]['labels'] for lbl in clabels], axis=1)
        # else:
            # run sequentially
        dists = []
        trainLabels = []
        for lbl in clabels:
            feats = self.extract_feats(data, lbl)
            # print '\tfeats', numpy.shape(feats)
            trainData = self.feat_cannon[self.cannon_label == lbl, :]
            # print '---', lbl, numpy.shape(self.feat_cannon), numpy.shape(trainData), '---'
            # labels
            trainLabels.append(lbl * numpy.ones((datalen, len(trainData)), dtype='int'))
            # distances
            dists.append(dist.cdist(feats, trainData, self.metric))

        # amass results
        dists = numpy.concatenate(dists, axis=1)
        trainLabels = numpy.concatenate(trainLabels, axis=1)
        # print '\n\tdistsPRE\n', dists, numpy.shape(dists)
        # print '\n\ttrainLabelsPRE\n', trainLabels, numpy.shape(trainLabels)


        # sort
        dists, trainLabels = self._sort(dists, trainLabels)
        # print '\n\tdistsPOST\n', dists[:, 0:6], numpy.shape(dists)
        # print '\n\ttrainLabelsPOST\n', trainLabels[:, 0:6], numpy.shape(trainLabels)
        return dists, trainLabels


class Agrafioti(KNN):
    """
    Class to train and classify AC coefficients from ECG Signals

    Performs LDA if coefficient number is bigger than subject number
    """

    NAME = 'agrafiotiClassifier'
    EXT = '.hdf5'
    EER_IDX = 0

    def __init__(self, io=None, k=3, metric='euclidean', h=4., M=100., w=None, **kwargs):
        # run parent __init__
        super(Agrafioti, self).__init__(io=io, k=k)

        # algorithm self things
        self.k = k
        self.h = h
        self.ac_step = int(M/h)
        self.transform = []
        self.transfdata = []
        if w is None:
            self.w = numpy.linspace(1., 0.6, int(h))
        else:
            self.w = numpy.linspace(1., w, int(h))

        # minimum threshold
        self.minThr = 10 * numpy.finfo('float').eps

        # choose metric
        if metric == 'euclidean':
            self.metric = 'euclidean'
        elif metric == 'cosine':
            self.metric = 'cosine'
        else:
            raise NotImplementedError, "Metric %s unknown." % metric

    def _updateStrategy(self, oldData, newData):
        # update the training data of a class when new data is available

        # concatenate old with new data
        out = numpy.concatenate([newData, oldData], axis=0)

        return out

    def train(self, data=None, updateThresholds=True):
        # data is {subject: features (array)}
        # print len(data), len(data[0]), len(data[0][0]), len(data[0][1])

        # check inputs
        if data is None:
            raise TypeError, "Please provide input data."

        # check if classifier was already trained
        if self.is_trained:
            # this is a retrain
            self.re_train(data)
        else:
            rxx_cannon = []
            cannon_label = []
            # get subjects
            subjects = sorted( data.keys() )
            # print '\t WTF', subjects
            self.nbSubjects = len(subjects)

            # determine classifier mode
            if self.nbSubjects == 0:
                raise ValueError, "Please provide input data - empty dict."
            else:
                for i in xrange(self.nbSubjects):
                    # build dicts
                    sub = subjects[i]
                    self.subject2label[sub] = i

                    # print '\t*****', i, sub, self.subject2label, '*****'

                    # build rxx cannon
                    rxx_cannon.append(data[sub])
                    cannon_label.append(numpy.ones(len(data[sub]), dtype='int64')*i)

                    # save data
                    self.io_save(i, data[sub])

            # concatenate cannon and label
            self.rxx_cannon = numpy.concatenate(rxx_cannon, axis=0)
            self.cannon_label = numpy.concatenate(cannon_label, axis=0)

        # perform LDA transform
        if self.ac_step >= self.nbSubjects:
            for j in xrange(int(self.h)):
                transf_mat = dimreduction.lda_train(self.rxx_cannon[:, j*self.ac_step:(j+1)*self.ac_step], self.cannon_label)
                self.transform.append(transf_mat)
                transf_data = dimreduction.lda_project(self.rxx_cannon[:, j*self.ac_step:(j+1)*self.ac_step], transf_mat)
                self.transfdata.append(transf_data)
        else:
            for j in xrange(int(self.h)):
                # print '---', j, '---'
                # print '\tcannon', numpy.shape(self.rxx_cannon), (j+1)*self.ac_step-j*self.ac_step
                transf_data = self.rxx_cannon[:, j*self.ac_step:(j+1)*self.ac_step]
                # print '\ttransf', numpy.shape(transf_data)
                self.transfdata.append(transf_data)
                # print '\tselftransf', len(self.transfdata)
        self.transfdata = numpy.array(self.transfdata)
        # print 'shapes:', numpy.shape(self.transfdata), numpy.shape(self.transform)
        # print 'label:', self.cannon_label, numpy.shape(self.cannon_label)
        # print 'demo transf:', self.transfdata[:, :4, :]
        # raw_input('-')
        # # train flag
        self.is_trained = True

        # update thresholds
        if updateThresholds:
            self.updateThresholds()

    def re_train(self, data):
        # data is {subject: features (array)}
        ### user

        for sub in sorted( data.iterkeys() ):
            # print '---', sub, '---'
            if sub in self.subject2label:
                # existing subject
                if data[sub] is not None:
                    # change subject's data
                    label = self.subject2label[sub]
                    # print 'label', label

                    # update templates
                    aux = self._updateStrategy(self.io_load(label), data[sub])

                    # update cannon and its label
                    insertind = numpy.argmax(self.cannon_label >= label)
                    self.rxx_cannon = numpy.insert(self.rxx_cannon, insertind, data[sub], 0)
                    self.cannon_label = numpy.insert(self.cannon_label, insertind, label*numpy.ones(len(data[sub]), dtype='int64'), 0)
                    # print 'UPDATE', sub
                    # save data
                    self.io_save(label, aux)
                else:
                    # delete subject
                    if self.nbSubjects == 1:
                        # reset classifier to untrained
                        self._reset()
                    else:
                        # reorder the labels/subjects and models dicts
                        subject2label, nbSubjects = self._snapshot()
                        self._reset()

                        label = subject2label[sub]

                        # update cannon and its label
                        removeinds = numpy.where(self.cannon_label == label)[0]
                        self.rxx_cannon = numpy.delete(self.rxx_cannon, removeinds, axis=0)
                        self.cannon_label = numpy.delete(self.cannon_label, removeinds, axis=0)

                        clabels = numpy.setdiff1d(numpy.unique(subject2label.values()),
                                                  [label], assume_unique=True)
                        self.nbSubjects = nbSubjects - 1

                        i = 0
                        for ii in xrange(len(clabels)):
                            # build dicts
                            sub = subject2label[:clabels[ii]]
                            self.subject2label[sub] = i
                            clabel_inds = numpy.where(self.cannon_label == clabels[ii])[0]
                            self.cannon_label[clabel_inds] = i

                            # move data
                            self.io_save(i, self.io_load(clabels[ii]))

                            # update i
                            i += 1
                        # print 'DELETE', sub
            else:
                # new subject
                # add to dicts
                label = self.nbSubjects
                self.subject2label[sub] = label

                # update cannon and its label
                self.rxx_cannon = numpy.vstack((self.rxx_cannon, data[sub]))
                self.cannon_label = numpy.hstack((self.cannon_label, numpy.ones(len(data[sub]), dtype='int64')*label))

                # save data
                self.io_save(label, data[sub])

                # increment number of subjects
                self.nbSubjects += 1

                # print 'ADD', sub

            self.transfdata = []

    def _getdist(self, testwaves, trainwaves):
        """
            Get distances (a la agrafioti) from test windows to requested training windows

            Inputs:

                testwaves - the testwaves which will be compared (they are input in LDA space)

                trainwaves - the trainwaves to which the testwaves will be compared (also input in LDA space)

                fdistance - the metric used to measure distances (usually euclidean)

            Outputs:

                distmat - matrix with the windowwise distances (row according to test window, column according to train)

        """
        d = numpy.zeros( (numpy.shape(testwaves)[1], numpy.shape(trainwaves)[1]) )
        hdists = numpy.zeros( (numpy.shape(testwaves)[1], numpy.shape(trainwaves)[1]) )
        for j in xrange(int(self.h)):
            scale = 1.*numpy.shape(testwaves[j])[1]
            modulator = numpy.abs(testwaves[j][:, None] - trainwaves[j][None, :]) # check trick at https://stackoverflow.com/questions/21938189/scipy-cdist-or-pdist-on-arrays-of-complex-numbers
            for i in xrange(len(modulator)):
                hdists[i] = numpy.sqrt(numpy.sum(modulator[i]**2, axis=1))/scale
            buff = self.w[j]*hdists/self.h
            d += buff
        return d

    def _compute(self, data, label='all', transf=False):
        """

            Computes all distances between points in data and in the training set

        Inputs:

            data - the data matrix, it is either transformed or Rxx windows to be transformed here according to...

            transf - flag which dictates whether the input data is transformed or not

            label - the training data labels with which to compare input data

        Outputs

            dictionary with distance matrix (number of data window rows & number of train data columns) and
            corresponding labels

        """

        # perform LDA transform over test data
        transf_test = []
        # print 'shape0:', numpy.shape(data), self.ac_step, self.nbSubjects
        # print 'PRE', data
        if not transf:
            if self.ac_step >= self.nbSubjects:
                for j in xrange(int(self.h)):
                    if len(numpy.shape(data)) > 1:
                        transf_data = dimreduction.lda_project(data[:, j*self.ac_step:(j+1)*self.ac_step], self.transform[j])
                    else:
                        transf_data = dimreduction.lda_project(data[j*self.ac_step:(j+1)*self.ac_step], self.transform[j])
                    transf_test.append(transf_data)
            else:
                for j in xrange(int(self.h)):
                    if len(numpy.shape(data)) > 1:
                        transf_data = data[:, j*self.ac_step:(j+1)*self.ac_step]
                    else:
                        transf_data = data[j*self.ac_step:(j+1)*self.ac_step]
                    transf_test.append(transf_data)
            data = numpy.array(transf_test)
        # print 'shape:', numpy.shape(data), numpy.shape(self.transfdata)
        # print 'demo:\n', data
        # print 'demo train:\n', self.transfdata[:, :4, :]

        # class labels
        if label == 'all':
            clabels = sorted(self.subject2label.values())
        else:
            clabels = [label]
        # print '\tCLABELS', clabels
        # compute distances ---FRANCIS--- WTF?
        # if len(clabels) > (4 * config.numberProcesses):
        #     # run in multiprocessing
        #     workQ = parallel.getQueue()
        #     for lbl in clabels:
        #         workQ.put({
        #                    'testData': data,
        #                    'trainData': self.io_load(lbl),      # isto vao ser os dados transformados
        #                    'taskid': lbl,
        #                    })
        #     # run in multiprocess (distances mode)
        #     store = parallel.getDictManager()
        #     parallel.runMultiprocess(workQ, store, mode='distances')
        #
        #     # amass results
        #     dists = numpy.concatenate([store[lbl]['distances'] for lbl in clabels], axis=1)
        #     trainLabels = numpy.concatenate([store[lbl]['labels'] for lbl in clabels], axis=1)
        # else:
            # run sequentially
        dists = []
        trainLabels = []
        # ltd = len(data)
        ltd = numpy.shape(data)[1]
        for lbl in clabels:
            # print '---', lbl, numpy.shape(self.transfdata), '---'
            # trainData = self.io_load(lbl)
            trainData = self.transfdata[:, numpy.array(self.cannon_label == lbl), :]
            # labels
            trainLabels.append(lbl * numpy.ones((ltd, numpy.shape(trainData)[1]), dtype='int'))
            # distances
            # dists.append([self._getdist(obs, trainData, self.distFcn) for obs in data])
            dists.append(self._getdist(data, trainData))

        # amass results
        dists = numpy.concatenate(dists, axis=1)
        trainLabels = numpy.concatenate(trainLabels, axis=1)
        # print '\n\tdistsPRE\n', dists
        # print '\n\ttrainLabelsPRE\n', trainLabels


        # sort
        dists, trainLabels = self._sort(dists, trainLabels)
        # print '\n\tdistsPOST\n', dists
        # print '\n\ttrainLabelsPOST\n', trainLabels

        return dists, trainLabels

    def autoRejectionThresholds(self):
        # generate thresholds automatically

        if self._autoThresholds is not None:
            return self._autoThresholds

        if self.metric == 'cosine':
            self._autoThresholds = numpy.linspace(self.minThr, 1., 100)
            return self._autoThresholds

        maxD = []
        for _ in xrange(3):
            # repeat
            for label in sorted(self.subject2label.values()):
                # randomly select one sample
                aux = self.transfdata[:, numpy.array(self.cannon_label == label), :]
                ind = config.random.randint(0, aux.shape[1], 1)
                obs = aux[:, ind, :]

                # compute distances
                dists, _ = self._compute(obs, transf=True)
                # print 'CURR_MAX:', numpy.max(dists)
                maxD.append(numpy.max(dists))

        # max distance
        maxD = 1.5 * numpy.max(maxD)

        # rejection thresholds to test
        self._autoThresholds = numpy.linspace(self.minThr, maxD, 100)

        return self._autoThresholds


def combinationDecorator(cls):
    # decorate the class in order to combine blobs of features into a single result
    # changes the data representation to {sub: [features, ]}
    
    # preserve original class methods (these will be overwritten)
    cls._identify_PMK = cls._identify
    cls._prepareData_PMK = cls._prepareData
    cls.authenticate_PMK = cls.authenticate
    cls.train_PMK = cls.train
    cls.updateThresholds_PMK = cls.updateThresholds
    
    # special flag for _prepareData
    cls._PMK = False
    
    # override methods
    cls._prepareData = _prepareData_Comb
    cls._identify = _identify_Comb
    cls.authenticate = authenticate_Comb
    cls.train = train_Comb
    cls.updateThresholds = updateThresholds_Comb
    
    return cls


def _identify_Comb(self, data, threshold=None, ready=False, **kwargs):
    # override _identify method for combination
    
    # set PMK
    self._PMK = True
    try:
        return numpy.array([rules.combination(self._identify_PMK(item, threshold=threshold, ready=ready, **kwargs))[0] for item in data])
    finally:
        self._PMK = False


def _prepareData_Comb(self, data, **kwargs):
    # override _prepareData method for combination
    
    if self._PMK:
        # old method
        return self._prepareData_PMK(data, **kwargs)
    else:
        # new method
        return [self._prepareData_PMK(item, **kwargs) for item in data]


def authenticate_Comb(self, data, subject, threshold=None, ready=False, labels=False, **kwargs):
    # override authenticate method for combination
    
    # set PMK
    self._PMK = True
    try:
        if labels:
            decision, prediction = [], []
            for item in data:
                d, p = self.authenticate_PMK(item, subject, threshold=threshold, ready=ready, labels=True, **kwargs)
                decision.append(rules.combination(d)[0])
                prediction.append(rules.combination(p)[0])
            decision = numpy.array(decision)
            return decision, prediction
        else:
            return numpy.array([rules.combination(self.authenticate_PMK(item, subject, threshold=threshold, ready=ready, labels=False, **kwargs))[0] for item in data])
    finally:
        self._PMK = False


def train_Comb(self, data=None, updateThresholds=True, **kwargs):
    # override train method for combination
    
    # set PMK
    self._PMK = True
    try:
        cdata = {}
        for sub in data.iterkeys():
            aux = []
            for item in data[sub]:
                if len(item) > 0:
                    aux.extend(item)
            cdata[sub] = numpy.array(aux)
        
        self.train_PMK(data=cdata, updateThresholds=False, **kwargs)
    finally:
        self._PMK = False
    
    if updateThresholds:
        self.updateThresholds()


def updateThresholds_Comb(self, overwrite=False, N=1, **kwargs):
    # override the updateThresholds method
    
    ths = self.autoRejectionThresholds()
    
    labels = range(self.nbSubjects)
    if self.nbSubjects > 1:
        prob = 1. / (self.nbSubjects - 1)
        p = prob * numpy.ones(self.nbSubjects, dtype='float')
    else:
        p = None
    
    for lbl in self._subThrIterator(overwrite):
        subject = self.subject2label[:lbl]
        # choose random subjects for authentication
        test_lbl = [lbl]
        try:
            p[lbl] = 0
        except TypeError:
            pass
        else:
            test_lbl.extend(config.random.choice(labels, p=p, size=N))
            p[lbl] = prob
        
        # load data
        data = {self.subject2label[:item]: [self.io_load(item), ] for item in test_lbl}
        
        # evaluate classifier
        out = self.evaluate(data, ths)
        
        # choose threshold at EER
        EER_auth = out['assessment']['subject'][subject]['authentication']['rates']['EER']
        self.setAuthThreshold(lbl, EER_auth[self.EER_IDX, 0], ready=True)
        
        EER_id = out['assessment']['subject'][subject]['identification']['rates']['EER']
        self.setIdThreshold(lbl, EER_id[self.EER_IDX, 0], ready=True)


# SVM_Comb = combinationDecorator(SVM)


# OLD STUFF
class fisher(object):
    # Init
    def __init__(self, cStep=0.1, **kwargs):
        # generic self things
        self.is_trained = False
        self.label2subject = {}
        self.subject2label = {}
        self.nbSubjects = 0
        
        # parameters and configurations
        self.cStep = cStep
        
        # algorithm-specific
        self.authSide = []
        self.authPairs = None
        self.models = []
        
    
    def _computeBoundary(self, data1, data2):
        # binary Fisher classifier
        
        # compute w
        cov1 = numpy.cov(data1.T)
        cov2 = numpy.cov(data2.T)
        mean1 = numpy.mean(data1, axis=0)
        mean2 = numpy.mean(data2, axis=0)
#        w = numpy.dot(numpy.linalg.inv(cov1 + cov2) , (mean2 - mean1))
        w = numpy.linalg.solve(cov1 + cov2, mean2 - mean1)
        w /= numpy.sqrt(numpy.dot(w, w))
        
        # compute c
        pm1 = numpy.dot(w, mean1)
        pm2 = numpy.dot(w, mean2)
        c_min = numpy.min([pm1, pm2])
        c_max = numpy.max([pm1, pm2])
        c_try = numpy.arange(-c_max, -c_min , self.cStep)
        error_c = []
        
        for t in c_try:
            verify = []
            # class 1
            for x in data1:
                p = numpy.dot(w.T, x)
                if p < -t:
                    # classified as class 1
                    verify.append(True)
                else:
                    # classified as class 2
                    verify.append(False)
            
            # class 2
            for x in data2:
                p = numpy.dot(w.T, x)
                if p < -t:
                    # classified as class 1
                    verify.append(False)
                else:
                    # classified as class 2
                    verify.append(True)
            
            # error stat
            error_c.append(verify.count(False) / float(len(verify)))
            
        # find minimum
        index = numpy.where(error_c == numpy.min(error_c))[0]
        #if there is more than one minimum error value, it chooses the middle one
        nb = len(index)
        if nb > 1:
            index = index[nb / 2]
        else :
            index = index[0]
        
        return {'w': w, 'c': c_try[index], 'error': error_c[index]}
    
    def _predict(self, data, classTuple, w, c, **kwargs):
        # classify data (a single sample)
        
        if numpy.dot(w.T, data) < -c:
            # class 1
            res = classTuple[0] 
        else:
            # class 2
            res = classTuple[1]
        
        return res
    
    def train(self, data=None):
        # data is {subject: features (array)}
        
        # check if classifier was already trained
        if self.is_trained:
            warnings.warn("Classifier already trained; nothing will be done.")
            return
        
        # check inputs
        if data is None:
            raise TypeError, "Please provide input data."
        
        # get subjects
        subjects = data.keys()
        self.nbSubjects = len(subjects)
        self.authPairs = numpy.zeros((self.nbSubjects, self.nbSubjects - 1), dtype='int')
        
        if self.nbSubjects == 1:
            raise NotImplementedError, "Single class Fischer classifier not yet implemented."
        
        start = 0
        for i in xrange(self.nbSubjects):
            # build dicts
            sub = subjects[i]
            self.label2subject[i] = sub
            self.subject2label[sub] = i
            
            for j in xrange(i+1, self.nbSubjects):
                # build model
                self.models.append(self._computeBoundary(data[subjects[i]], data[subjects[j]]))
                self.authSide.append((i, j))
            
            stop = start + self.nbSubjects - i - 1
            self.authPairs[i, i:] = numpy.arange(start, stop)
            if i < self.nbSubjects - 1:
                self.authPairs[i+1:, i] = self.authPairs[i, i:]
            start = stop
        
        # convert to numpy
        self.models = numpy.array(self.models)
        self.authSide = numpy.array(self.authSide)
        
        self.is_trained = True
    
    def authenticate(self, data, subject, threshold=1., labels=False, **kwargs):
        # data is a list of feature vectors, allegedly belonging to the given subject
        
        # check train state
        if not self.is_trained:
            raise ValueError, "Please train the classifier first."
        
        # translate subject ID to class label
        label = self.subject2label[subject]
        
        # get models and class tuples
        models = self.models[self.authPairs[label, :]]
        classTuples = self.authSide[self.authPairs[label, :]]
        
        # predict classes for each classifier
        decision = []
        prediction = []
        for obs in data:
            pred = [self._predict(obs, ct, **m) for m, ct in izip(models, classTuples)]
            
            # determine "majority"
            counts = numpy.bincount(pred)
            predMax = counts.argmax()
            rate = float(counts[predMax]) / (self.nbSubjects - 1)
            if rate >= threshold:
                # rate of agreement is >= threshold
                decision.append(predMax == label)
                prediction.append(predMax)
            else:
                # rate of agreement is < threshold
                decision.append(False)
                prediction.append(predMax)
        
        # convert to numpy
        decision = numpy.array(decision)
        
        if labels:
            # translate class label to subject ID
            subPrediction = [self.label2subject[item] for item in prediction]
            return decision, subPrediction
        else:
            return decision
    
    def identify(self, data, **kwargs):
        # data is list of feature vectors
        
        # check train state
        if not self.is_trained:
            raise ValueError, "Please train the classifier first."
        
        labels = []
        for obs in data:
            pred = [self._predict(obs, ct, **m) for m, ct in izip(self.models, self.authSide)]
            # determine maximum
            counts = numpy.bincount(pred)
            labels.append(counts.argmax())
        
        # translate class labels to subject IDs
        subjects = numpy.array([self.label2subject[item] for item in labels])
        
        return subjects
    
    def evaluate(self, data, rejection_thresholds='auto', dstPath=None):
        # data is {subject: features (array)}
        
        # check train state
        if not self.is_trained:
            raise ValueError, "Please train the classifier first."
        
        # choose thresholds
        if rejection_thresholds == 'auto':
            rejection_thresholds = numpy.arange(1, self.nbSubjects, dtype='float')
            rejection_thresholds /= (self.nbSubjects - 1)
        
        # choose store
        if dstPath is None:
            store = parallel.getDictManager()
        else:
            store = dstPath
        
        results = {}
        subjects = numpy.array(data.keys())
        for subject in subjects:
            # identification test
            id_res = self.identify(data[subject], ready=True)
            
            # authentication test
            auth_res = []
            workQ = parallel.getQueue()
            for i in xrange(len(rejection_thresholds)):
                workQ.put({
                           'classifier': self,
                           'data': data[subject],
                           'threshold': rejection_thresholds[i],
                           'subjects': subjects,
                           'parameters': {},
                           'taskid': i,
                           })
            
            # run in multiprocessing
            parallel.runMultiprocess(workQ, store, mode='clf', log2file=True)
            
            # load from files
            for i in xrange(len(rejection_thresholds)):
                auth_res.append(parallel.loadStore(store, i))
            
            # clean up store
            parallel.cleanStore(store)
            
            auth_res = numpy.array(auth_res)
            results[subject] = {'identification': id_res,
                                'authentication': {'cube': auth_res,
                                                   'subjectLabels': subjects,
                                                   },
                                }
            
        return results, rejection_thresholds



if __name__=='__main__':    
    pass
    
