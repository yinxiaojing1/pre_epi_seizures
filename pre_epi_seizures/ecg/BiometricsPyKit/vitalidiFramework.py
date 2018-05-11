"""
.. module:: vitalidiFramework
   :platform: Unix, Windows
   :synopsis: This module interfaces between the Vitalidi demonstrator Python back-end and BiometricsPyKit.

.. moduleauthor:: Ana Priscila Alves, Carlos Carreiras


"""

# Imports
# built-in
import copy
import datetime
import glob
import json
from multiprocessing import Process
import shutil
import traceback
import os

# 3rd party
import numpy

# BioSPPy
from database import h5db

# BiometricsPyKit
import config
from classifiers import classifiers
from datamanager import datamanager
from evaluation import evaluation
from misc import misc
from outlier import outlier
from preprocessing import preprocessing
from visualization import visualization as vis


# settings template
settings = {'MAC': '',
            
            'classification': {'clf_0': {'method': '',
                                         'parameters': {},
                                         },
                               # ...
                               'clf_N': {'method': '',
                                         'parameters': {},
                                         },
                               
                               'confidence_threshold': 0,
                               },
            
            'outlier': {'enroll': {'method': '',
                                   'parameters': {},
                                   },
                        
                        'id': {'method': '',
                               'parameters': {},
                               },
                        },
            
            'autoAuth': {'threshold': 0,
                         'k': 0,
                         },
            
            'validation': {
                           'enroll': {'time': 0,
                                      'minimum_templates': 0,
                                      },
                           
                           'id': {'time': 0,
                                  'minimum_templates': 0,
                                  },
                           },
            }


ERR_MSG = {'ID-0': 'Identification: Success',
           'ID-1': 'Identification: Error: UntrainedError',
           'ID-2': 'Identification: Error: Default subject',
           'ID-3': 'Identification: Error: Combination confidence below threshold',
           'ID-4': 'Identification: Error: Empty combination',
           
           'AUT-0': 'Authentication: Success',
           'AUT-1': 'Authentication: Error: UntrainedError',
           'AUT-2': 'Authentication: Error: UnknownSubjectError',
           'AUT-3': 'Authentication: Error: Combination confidence below threshold',
           'AUT-4': 'Authentication: Error: Empty combination',
           
           'TR-0': 'Train: Success',
           'TR-1': 'Train: Error',
           
           'RM-0': 'Remove: Success',
           'RM-1': 'Remove: Error',
           
           'VAL-0': 'Validation: Success',
           'VAL-1': 'Validation: Error: Not enough segments',
           'VAL-2': 'Validation: Error: Not enough good segments',
           'VAL-3': 'Validation: Error: Auto-authentication failed',
           }



class BaseBrain(object):
    """
    Base class for Vitalidi brains.
    
    """
    
    CLF_NAMES = []
    CLF_WEIGHTS = {}
    
    def __init__(self, path, settings, ignorePrevious=False, deployLocations=None):
        # instantiate or load the individual classifiers
        
        # self things
        self.path = config.folder = os.path.join(os.path.normpath(path), 'Brain')
        self.clfs = {}
        self.clfPath = os.path.join(self.path, 'clfs')
        
        # load settings
        self.loadSettings(settings)
        
        # load classifiers
        if ignorePrevious:
            # ignore previously saved classifiers
            try:
                shutil.rmtree(self.clfPath)
            except OSError:
                pass
            
            # deploy at path
            self.deploy(deployLocations)
            self.createClassifiers()
        else:
            # deploy at path
            self.deploy(deployLocations)
            
            # load classifiers
            for n in self.CLF_NAMES:
                cls = self.clfClass[n]
                try:
                    self.clfs[n] = cls.load(os.path.join(self.clfPath, n))
                except IOError:
                    # at least one of the individual classifiers is missing
                    self.createClassifiers()
                    break
    
    def deploy(self, deployLocations=None):
        # deploy at path
        
        if deployLocations is None:
            config.deploy(['clfs', 'data', 'data/Unknown', 'validation'])
        else:
            if 'clfs' not in deployLocations:
                deployLocations.append('clfs')
            
            config.deploy(deployLocations)
    
    def loadSettings(self, settings):
        # load settings
        
        if isinstance(settings, basestring):
            # load settings from file
            with open(settings, 'r') as fid:
                self.settings = json.load(fid)
        elif isinstance(settings, dict):
            # dict settings
            self.settings = settings
        else:
            raise TypeError, "Unsupported settings type."
        
        # verify classifier settings, update io
        self.clfClass = {}
        for n in self.CLF_NAMES:
            try:
                self.clfClass[n] = classifiers.selector(self.settings['classification'][n]['method'])
                aux = self.settings['classification'][n]['parameters']
            except KeyError:
                raise ValueError, "No settings for %s classifier." % n
            else:
                aux['io'] = self.clfPath
        
        # quick settings
        self.confThr = self.settings['classification']['confidence_threshold']
    
    def createClassifiers(self):
        # create new classifier instances
        
        for n in self.CLF_NAMES:
            # instantiate
            cls = self.clfClass[n]
            parameters = copy.deepcopy(self.settings['classification'][n]['parameters'])
            self.clfs[n] = cls(**parameters)
            # rebrand
            try:
                self.clfs[n]._rebrand(n)
            except OSError, e:
                print e
                ### think about this
            # save
            self.clfs[n].save(self.clfPath)
    
    def prepareData(self, data):
        # prepare the data for classification
        
        out = {}
        
        return out
    
    def saveOutliersPlot(self, segments, partition, currentTime=None):
        # save outliers plot
        
        if currentTime is None:
            currentTime = self.getCurrentTime()
        
        # plot
        fig = vis.plotOutliers(segments, partition)
        
        # save
        fig.savefig(os.path.join(self.path, 'validation', '%s.png' % currentTime), bbox_inches='tight')
        vis.close(fig)
        
        print "Outlier figure saved."
    
    def saveFile(self, command, name, Rawsignal=None, FilteredSignal=None, Segments=None,
                 currentTime=None, eventValue=None, good_segments=None, res=None, mode='a'):
        # save to hdf5
        
        if currentTime is None:
            currentTime = self.getCurrentTime()
        signalName = 'rec-' + currentTime
        
        # file name
        if name == 'tmp':
            fpath = os.path.join(self.path, 'data', 'tmpRecord.hdf5')
            uid = -1
        else:
            path = os.path.join(self.path, 'data', str(name))
            if not os.path.exists(path):
                os.makedirs(path)
            
            fpath = os.path.join(path, '%s.hdf5' % signalName)
            uid = 1 + len(glob.glob(os.path.join(path, '*.hdf5')))
        
        with h5db.hdf(fpath, mode) as fid:
            # add header
            header = {'command': command, 'name': signalName, 'date': currentTime, 'id': uid}
            fid.addInfo(header)
            
            # add raw signal
            if Rawsignal is not None:
                dtype = '/raw'
                dname = 'signal%d' % len(fid.listSignals(dtype)['signalsList'])
                mdata = {'type': dtype, 'date':  currentTime , 'name': dname}
                fid.addSignal(dtype, Rawsignal, mdata, dname)
            
            # add filtered signal
            if FilteredSignal is not None:
                dtype = '/filtered'
                dname = 'signal%d' % len(fid.listSignals(dtype)['signalsList'])
                mdata = {'type': dtype, 'date': currentTime , 'name': dname}
                fid.addSignal(dtype, FilteredSignal, mdata, dname)
            
            # add segments
            if Segments is not None:
                if res is None:
                    res = self.settings['outlier'][command]
                dtype = '/segments'
                dname = 'signal%d' % len(fid.listSignals(dtype)['signalsList'])
                mdata = {'type': dtype, 'date': currentTime , 'name': dname,
                         'good_segments': good_segments, 'outlierDetection': res}
                fid.addSignal(dtype, Segments, mdata, dname)
            
            if eventValue is not None:
                # add R peaks indexes
                dtype = '/filtered'
                R = numpy.array([eventValue, eventValue]).T
                mdata = {'type': dtype, 'date': currentTime, 'name': 'R'}
                fid.addEvent(dtype, timeStamps=R, mdata=mdata, eventName='R')
        
        print 'File Saved.'
    
    def updateFile(self, name):
        # update uid of the tmp record, rename and copy to correct destination
        
        srcFileName = os.path.join(self.path, 'data', 'tmpRecord.hdf5')
        with h5db.hdf(srcFileName, 'a') as fid:
            header = fid.getInfo()['header']
            header.update({'id': 1 + len(glob.glob(os.path.join(self.path, 'data', name, '*.hdf5')))})
            out = header['name'] + '.hdf5'
            cmd = header['command']
            fid.addInfo(header)
        
        dstFileName = os.path.join(self.path, 'data', name, out)
        
        # copy
        shutil.copy(srcFileName, dstFileName)
        
        # remove tmp
        os.remove(srcFileName)
        
        print 'File Updated (%s).' % name
        
        return cmd
    
    def validateAutoAuth(self, segments):
        # validate auto-authentication
        
        # settings
        threshold = self.settings['autoAuth']['threshold']
        k = self.settings['autoAuth']['k']
        
        length = len(segments)
        dists = []
        
        for i in xrange(length):
            for j in xrange(i+1, length):
                dists.append(misc.msedistance(segments[i], segments[j]))
        
        # min distances
        dists = numpy.array(dists)
        dists.sort()
        dists = dists[:k]
        
        if numpy.all(dists < threshold):
            success = True
        else:
            success = False
        
        return success
    
    def validation(self, nsegments, segments, case, autoAuth=False):
        # Validate ECG templates
        
        # settings
        outlierSettings = self.settings['outlier'][case]
        min_nb_segments = self.settings['validation'][case]['minimum_templates']
        
        success = False
        output = None
        
        # verify minimum number of segments
        if nsegments >= min_nb_segments:
            # outlier detection and removal
            output = outlier.runMethod(segments, outlierSettings['method'], **outlierSettings['parameters'])
            
            # verify minimum number of good segments
            if output['nbGood'] >= min_nb_segments:
                if autoAuth:
                    # validate auto-authentication
                    if self.validateAutoAuth(output['templates']):
                        success = True
                        message = ERR_MSG['VAL-0']
                    else:
                        message = ERR_MSG['VAL-3']
                else:
                    success = True
                    message = ERR_MSG['VAL-0']
            else:
                message = ERR_MSG['VAL-2']
        else:
            message = ERR_MSG['VAL-1']
        
        return output, success, message
    
    def validationStatus(self, nsegments, case):
        # compute a measure of progress until validation is successful
        
        min_nb_segments = self.settings['validation'][case]['minimum_templates']
        
        # the last 10% are the hardest
        if min_nb_segments > 0:
            rate = (90. * nsegments) / min_nb_segments
            if rate > 90:
                rate = 90
        else:
            rate = 90
        
        return rate
    
    @classmethod
    def getCurrentTime(cls):
        # current date and time (UTC)
        return datetime.datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S')
    
    @staticmethod
    def listFolderSubjects(path):
        # list subjects in folder
        
        folders = filter(lambda i: os.path.isdir(os.path.join(path, i)), os.listdir(path))
        folders = filter(lambda i: i not in ['json', 'outlierdet', 'Unknown', 'journal'], folders)
        folders.sort()
        
        return folders
    
    def listSubjects(self):
        # list the subjects
        
        return self.listFolderSubjects(os.path.join(self.path, 'data'))
    
    def listClassSubjects(self):
        # list the enrolled subjects
        
        subjects = set()
        for n in self.CLF_NAMES:
            for item in self.clfs[n].listSubjects():
                subjects.add(item)
        
        subjects = list(subjects)
        subjects.sort()
                
        return subjects
    
    def checkSubject(self, subject):
        aux = []
        for c in self.clfs.itervalues():
            aux.append(c.checkSubject(subject))
        
        return any(aux)
    
    def _combination(self, results):
        # private combination
        
        # compile results to find all classes
        vec = numpy.concatenate(results.values())
        unq = numpy.unique(vec)
        
        nb = len(unq)
        if nb == 0:
            # empty array
            raise classifiers.EmptyCombination
        elif nb == 1:
            # unanimous result
            decision = unq[0]
            confidence = 1.
            counts = [1.]
        else:
            # multi-class
            counts = numpy.zeros(nb, dtype='float')
            
            for n in results.iterkeys():
                # ensure array
                res = numpy.array(results[n])
                ns = float(len(res))
                
                # get count for each unique class
                for i in xrange(nb):
                    aux = float(numpy.sum(res == unq[i]))
                    counts[i] += ((aux / ns) * self.CLF_WEIGHTS[n])
            
            # most frequent class
            predMax = counts.argmax()
            counts /= counts.sum()
            
            decision = unq[predMax]
            confidence = counts[predMax]
        
        return decision, confidence, counts, unq
    
    def combination(self, results):
        # combine the result (from identification or authentication) => most frequent result
        
        decision, confidence, _, _ = self._combination(results)
        
        return decision, confidence
    
    def train(self, data, subject, updateThresholds=True):
        # data is list of feature vectors
        
        # prepare data
        aux = self.prepareData(data)
        
        for n in self.CLF_NAMES:
            if len(aux[n]) == 0:
                continue
            
            # train
            self.clfs[n].train({subject: aux[n]}, updateThresholds=updateThresholds)
            
            # save
            self.clfs[n].save(self.clfPath)
        
        return True, ERR_MSG['TR-0']
    
    def removeSubject(self, subject, updateThresholds=False):
        # remove a subject from the classifiers
        
        for n in self.CLF_NAMES:
            # train
            self.clfs[n].train({subject: None}, updateThresholds=updateThresholds)
            
            # save
            self.clfs[n].save(self.clfPath)
        
        return True, ERR_MSG['RM-0']
    
    def _authenticate(self, data, subject, threshold=None):
        # private authentication routine
        # data has been previously prepared
        
        out = {}
        for n in self.CLF_NAMES:
            if len(data[n]) == 0:
                continue
            
            # authenticate
            try:
                out[n] = self.clfs[n].authenticate(data[n], subject, threshold=threshold)
            except classifiers.UnknownSubjectError:
                continue
            except classifiers.UntrainedError:
                continue
        
        return out
    
    def authenticate(self, data, subject, ready=False):
        # data is a list of feature vectors, allegedly belonging to the given subject
        
        # check if subject exists
        if not self.checkSubject(subject):
            return False, False, ERR_MSG['AUT-2']
        
        # prepare data
        if not ready:
            aux = self.prepareData(data)
        else:
            aux = data
        
        out = self._authenticate(aux, subject, threshold=None) ### review threshold
        
        # combine results
        try:
            decision, confidence = self.combination(out)
        except classifiers.EmptyCombination:
            return False, False, ERR_MSG['AUT-4']
        
        # check confidence
        if confidence < self.confThr:
            return False, True, ERR_MSG['AUT-3'] + ': %2.4f' % confidence
        
        return bool(decision), True, ERR_MSG['AUT-0']
    
    def groupAuthenticate(self, data, names=None, ready=False):
        # authenticate against a group of subjects
        
        # check if subjects exist
        if names is None:
            names = self.listClassSubjects()
        else:
            names = filter(self.checkSubject, names)
        
        if len(names) == 0:
            return False
        
        # prepare data
        if not ready:
            aux = self.prepareData(data)
        else:
            aux = data
        
        # authenticate each subject
        auths = [self.authenticate(aux, sub, ready=True)[0] for sub in names]
        
        # final decision
        ### we can also check if there is only one accepted subject
        decision = any(auths)
        
        return decision
    
    def _identify(self, data, threshold=None):
        # private identification routine
        # data has been previously prepared
        
        out = {}
        for n in self.CLF_NAMES:
            if len(data[n]) == 0:
                continue
            
            # identify
            try:
                out[n] = self.clfs[n].identify(data[n], threshold=threshold)
            except classifiers.UntrainedError:
                continue
        
        return out
    
    def identify(self, data, ready=False):
        # data is list of feature vectors
        
        # prepare data
        if not ready:
            aux = self.prepareData(data)
        else:
            aux = data
        
        out = self._identify(aux, threshold=None) ### review threshold
        
        # combine results
        try:
            subject, confidence = self.combination(out)
        except classifiers.EmptyCombination:
            return '', False, ERR_MSG['ID-4']
        
        # check confidence
        if confidence < self.confThr:
            return '', False, ERR_MSG['ID-3'] + ': %2.4f' % confidence
        
        if subject == '':
            return '', False, ERR_MSG['ID-2']
        
        return subject, True, ERR_MSG['ID-0']
    
    def debugTest(self, data):
        # perform identification and authentication, returning information for debugging
        
        # prepare data
        aux = self.prepareData(data)
        
        out = {}
        
        # identification
        bux = self._identify(aux, threshold=None)
        decision, confidence, counts, unq = self._combination(bux)
        out['identification'] = {'classifiers': bux,
                                 'decision': decision,
                                 'confidence': confidence,
                                 'counts': counts,
                                 'countLabels': unq,
                                 }
        
        # authentication
        out['authentication'] = {}
        for sub in self.listClassSubjects():
            bux = self._authenticate(aux, sub, threshold=None)
            decision, confidence, counts, unq = self._combination(bux)
            out['authentication'][sub] = {'classifiers': bux,
                                          'decision': decision,
                                          'confidence': confidence,
                                          'counts': counts,
                                          'countLabels': unq,
                                          }
        
        return out
    
    def evaluate(self, recordTypes=None):
        # evaluate the performance of the brain
        
        # check inputs
        if recordTypes is None:
            recordTypes = ['auth', 'id']
        
        # get subjects
        # subjects = self.listSubjects()
        subjects = self.listClassSubjects()
        
        # choose the ID or AUT records
        data = {}
        counts = {}
        for sub in subjects:
            data[sub] = []
            counts[sub] = [0, 0] # (test, enroll)
            
            # check if subject is enrolled
            if not self.checkSubject(sub):
                continue
            
            # get signals
            files = glob.glob(os.path.join(self.path, 'data', sub, '*.hdf5'))
            files.sort(reverse=True)
            for f in files:
                with h5db.hdf(f, 'r') as fid:
                    cmd = fid.getInfo()['header']['command']
                    if cmd in recordTypes:
                        counts[sub][0] += 1
                        aux = fid.getSignal('/segments', 'signal0')
                        data[sub].append(self.prepareData(aux['signal'][aux['mdata']['good_segments']]))
                    else:
                        counts[sub][1] += 1
            
            # check if there are no ID or AUT records
            if len(data[sub]) == 0:
                # only get one (the newest)
                for f in files:
                    with h5db.hdf(f, 'r') as fid:
                        aux = fid.getSignal('/segments', 'signal0')
                        data[sub].append(self.prepareData(aux['signal'][aux['mdata']['good_segments']]))
                    break
        
        # filter out subjects without data
        subjects = filter(lambda i: len(data[i]) > 0, subjects)
        
        # thresholds
        prevTh = self.confThr
        thresholds = numpy.linspace(0, 1, 20)
        
        results = {'subjectList': subjects,
                   'subjectDict': classifiers.SubjectDict((sub, sub) for sub in subjects),
                   }
        for sub in subjects:
            id_res = []
            auth_res = []
            
            for th in thresholds:
                self.confThr = th
                
                # identification
                aux = [self.identify(item, ready=True)[0] for item in data[sub]]
                id_res.append(numpy.array([-1 if item == '' else item for item in aux], dtype='object'))
                
                # authentication
                bux = []
                for sub_tst in subjects:
                    bux.append([self.authenticate(item, sub_tst, ready=True)[0] for item in data[sub]])
                auth_res.append(numpy.array(bux))
            
            # save to dict
            results[sub] = {'identification': numpy.array(id_res),
                            'authentication': numpy.array(auth_res),
                            }
        
        # restore threshold
        self.confThr = prevTh
        
        # assess results
        output = evaluation.assessClassification(results, thresholds)
        
        return output, counts
    
    def subjectPerformance(self, subject):
        # compute authentication performance for given subject
        
        res = []
        for _, item in self.loadSegments(subject, ['enroll', 'auth', 'id']):
            accept, _, _ = self.authenticate(item['signal'][item['mdata']['good_segments']], subject)
            res.append(accept)
        
        # randomly discard some tests
        nb = len(res) - 3
        if nb <= 0:
            prf = 0
        else:
            res = config.random.choice(res, size=nb, replace=False)
            prf = numpy.sum(res) / float(nb)
        
        return prf
    
    def loadSegments(self, subject, recordTypes):
        # iterator that yields the stored segments of a user used for a given purpose
        
        dataPath = os.path.join(self.path, 'data')
        files = glob.glob(os.path.join(dataPath, subject, '*.hdf5'))
        
        for f in files:
            fname = os.path.split(f)[1]
            with h5db.hdf(f, 'r') as fid:
                cmd = fid.getInfo()['header']['command']
                
                if cmd in recordTypes:
                    yield fname, fid.getSignal('/segments', 'signal0')
    
    def trainFromFiles(self, recordTypes=None, ignoreSubjects=None):
        # train brain from data files
        
        # check inputs
        if recordTypes is None:
            recordTypes = ['enroll']
        if ignoreSubjects is None:
            ignoreSubjects = []
        
        # filter out subjects to ignore
        subjects = filter(lambda item: item not in ignoreSubjects, self.listSubjects())
        subjects.sort()
        
        # load data
        for sub in subjects:
            data = []
            for _, item in self.loadSegments(sub, recordTypes):
                # get only good segments
                data.append(item['signal'][item['mdata']['good_segments']])
            
            # concatenate
            try:
                data = numpy.concatenate(data, axis=0)
            except ValueError:
                print "No data available for subject %s!" % sub
                continue
            else:
                # train
                self.train(data, sub)
    
    def trainFromFolder(self, path=None, ignoreSubjects=None):
        # train brain from an external folder
        
        # check inputs
        if path is None:
            raise TypeError, "Please specify an input folder."
        if ignoreSubjects is None:
            ignoreSubjects = []
        
        # filter out subjects to ignore
        subjects = filter(lambda item: item not in ignoreSubjects, self.listFolderSubjects(path))
        subjects.sort()
        
        # load data
        for sub in subjects:
            data = []
            
            files = glob.glob(os.path.join(path, sub, '*.hdf5'))
            for f in files:
                with h5db.hdf(f, 'r') as fid:
                    aux = fid.getSignal('/segments', 'signal0')
                    
                    # get only good segments
                    data.append(aux['signal'][aux['mdata']['good_segments']])
            
            # concatenate
            try:
                data = numpy.concatenate(data, axis=0)
            except ValueError:
                print "No data available for subject %s!" % sub
                continue
            else:
                # train
                self.train(data, sub)
    
    def updateThresholds(self):
        # update the classifier thresholds
        
        for n in self.CLF_NAMES:
            try:
                clf = self.clfs[n]
            except classifiers.UntrainedError:
                continue
            else:
                clf.updateThresholds(overwrite=True, fraction=0.5)
                clf.save(self.clfPath)


class MyBrain(BaseBrain):
    """
    Brain using SVM classifiers, combining all segments, 3 mean-waves and 5 mean-waves.
    """
    
    CLF_NAMES = ['segments', '3-meanWaves', '5-meanWaves']
    CLF_WEIGHTS = {'segments': 1.0,
                   '3-meanWaves': 1.0,
                   '5-meanWaves': 1.0,
                   }
    
    def prepareData(self, data):
        # prepare the data for classification
        
        out = {'segments': data,
               '3-meanWaves': misc.mean_waves(data, 3),
               '5-meanWaves': misc.mean_waves(data, 5),
               }
        
        return out


class QuickBrain(BaseBrain):
    """
    Brain using SVM classifiers, combining all segments, 3 mean-waves and 5 mean-waves.
    """
    
    CLF_NAMES = ['5-meanWaves']
    CLF_WEIGHTS = {'5-meanWaves': 1.0}
    
    def prepareData(self, data):
        # prepare the data for classification
        
        out = {'5-meanWaves': misc.mean_waves(data, 5)}
        
        return out


class KeyBrain(BaseBrain):
    """
    Brain using SVM classifiers, combining all segments, 3 median waves and 5 median waves.
    Templates are scaled with a gaussian window.
    """
    
    CLF_NAMES = ['segments', '3-medianWaves', '5-medianWaves']
    CLF_WEIGHTS = {'segments': 1.0,
                   '3-medianWaves': 2.0,
                   '5-medianWaves': 3.0,
                   }
    
    def prepareData(self, data):
        # prepare the data for classification
        
        # normalize
        data = preprocessing.norm9(data, R=200, sigma=1.5)
        
        out = {'segments': data,
               '3-medianWaves': misc.median_waves(data, 3),
               '5-medianWaves': misc.median_waves(data, 5),
               }
        
        return out


class MixBrain(BaseBrain):
    """
    Brain
    """
    
    CLF_NAMES = ['segments', '3-medianWaves-norm', '3-medianWaves', '5-medianWaves']
    CLF_WEIGHTS = {'segments': 1.0,
                   '3-medianWaves-norm': 1.0,
                   '3-medianWaves': 1.0,
                   '5-medianWaves': 2.0,
                   }
    
    def prepareData(self, data):
        # prepare the data for classification
        
        aux = preprocessing.norm9(data, R=200, sigma=1.5)
        
        out = {'segments': data,
               '3-medianWaves-norm': misc.median_waves(aux, 3),
               '3-medianWaves': misc.median_waves(data, 3),
               '5-medianWaves': misc.median_waves(data, 5),
               }
        
        return out


class DoorBrain(BaseBrain):
    """
    Brain using an SVM classifier with 5 median waves.
    Templates are scaled with a gaussian window.
    """
    
    CLF_NAMES = ['5-medianWaves']
    CLF_WEIGHTS = {'5-medianWaves': 1.0}
    
    def prepareData(self, data):
        # prepare the data for classification
        
        # normalize
        data = preprocessing.norm9(data, R=200, sigma=1.5)
        
        out = {'5-medianWaves': misc.median_waves(data, 5)}
        
        return out


class RunEvaluate(Process):
    """
    Run classifier evaluation
    """
    
    def __init__(self, brainCls=None, stateDict=None, *args, **kwargs):
        # run parent __init__
        super(RunEvaluate, self).__init__()
        
        # check inputs
        if brainCls is None:
            raise TypeError, "Please provide the Brain class."
        
        if stateDict is None:
            raise TypeError, "Please provide the state dictionary."
        
        # self inputs
        self.brain = brainCls(*args, **kwargs)
        self.stateDict = stateDict
    
    def run(self):
        # run evaluation
        
        # current time
        currentTime = self.brain.getCurrentTime()
        self.stateDict['evalImg'] = 'Brain/validation/eval_%s.png' % currentTime
        
        # run evaluation
        out, counts = self.brain.evaluate()
        
        # save
        datamanager.skStore(os.path.join(self.brain.path, 'validation', 'eval_%s.dict' % currentTime), out)
        
        # plot
        try:
            # subjects
            subjects = out['subject'].keys()
            subjects.sort()
            
            xticklabels = ['Global']
            xticklabels.extend(subjects)
            data = []
            # EID
            aux = []
            aux.append(out['global']['identification']['rates']['Err'] * 100.)
            aux.extend(out['subject'][sub]['identification']['rates']['Err'] * 100. for sub in subjects)
            data.append(aux)
#            # Acc
#            aux = []
#            aux.append(out['global']['authentication']['rates']['Acc'] * 100.)
#            aux.extend(out['subject'][sub]['authentication']['rates']['Acc'] * 100. for sub in subjects)
#            data.append(aux)
            # FAR
            aux = []
            aux.append(out['global']['authentication']['rates']['FAR'] * 100.)
            aux.extend(out['subject'][sub]['authentication']['rates']['FAR'] * 100. for sub in subjects)
            data.append(aux)
            # FRR
            aux = []
            aux.append(out['global']['authentication']['rates']['FRR'] * 100.)
            aux.extend(out['subject'][sub]['authentication']['rates']['FRR'] * 100. for sub in subjects)
            data.append(aux)
            
            # counts
            fmt = '%d T // %d E'
            xl = len(xticklabels)
            dcounts = [''] * xl
            gtest, genroll = 0, 0
            for i in xrange(1, xl):
                dcounts[i] = fmt % (counts[xticklabels[i]][0], counts[xticklabels[i]][1])
                gtest += counts[xticklabels[i]][0]
                genroll += counts[xticklabels[i]][1]
            
            dcounts[0] = fmt % (gtest, genroll)
            
            # make figure
            fig = vis.figure(figsize=(16, 9))
            ax = fig.add_subplot(111)
            
            # bar plot
            vis.multiBarPLotAxis(ax, data, xticklabels, labels=['EID', 'FAR', 'FRR'],
                                 ylabel='Percentage', ylim=(0, 130), loc='upper right',
                                 rotation=25, xtickssize=10, vlines=True)
#            vis.multiBarPLotAxis(ax, data, xticklabels, labels=['EID', 'Acc', 'FAR', 'FRR'],
#                                 ylabel='Percentage', ylim=(0, 120), loc='upper right',
#                                 rotation=25, xtickssize=10, vlines=True)
            
            # number of test and enroll records
            vis.multiBarPLotAxisText(ax, dcounts, ypos=101, textalpha=5.0)
            
            # save figure
            fig.savefig(os.path.join(self.brain.path, 'validation', 'eval_%s.png' % currentTime), dpi=250, bbox_inches='tight')
            vis.close(fig)
        except Exception:
            with open(os.path.join(self.brain.path, 'validation', 'debug.txt'), 'w') as fid:
                traceback.print_exc(file=fid)

