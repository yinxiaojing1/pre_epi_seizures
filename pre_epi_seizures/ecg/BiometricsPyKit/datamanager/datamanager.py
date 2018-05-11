"""
.. module:: datamanager
   :platform: Unix, Windows
   :synopsis: This module provides various functions to load and store data.

.. moduleauthor:: Carlos Carreiras


"""

# TODO:
# Falta implementar classes para MonitorPlux e GZIP
# mod


# imports
# built-in
import csv
import errno
import cPickle
import glob
import gzip
import os
import shutil
import warnings
import zipfile

# 3rd party
from sklearn.externals import joblib
import numpy as np
import h5py

# BiometricsPyKit
import config

# BioSPPy
from database import biomesh
from database import h5db




def Store(config):
    """
    Method to obtain the appropriate database manager: StorageBIT (online or offline), MonitorPlux, GZIP.
    
    Input:
        config (dict): Configuration parameters.
    
    Output:
        unit (DBUnit, H5Unit, MPUnit or GZUnit): The database manager instance.
    
    Configurable fields:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    """
    
    # check inputs
    if config is None:
        raise TypeError, "Please specify a configuration."
    
    # data source
    try:
        source = config['source']
    except KeyError:
        raise KeyError, "Please specify the source ('StorageBIT', 'HDF5', 'MonitorPlux', or 'gzip') of the data in the configuration under 'source'."
    
    # mapper
    try:
        mapper = config['mapper']
    except KeyError:
        raise KeyError, "Please specify the mapping in the configuration under 'mapper'."
    
    # switch
    if source == 'StorageBIT':
        # connection parameters
        try:
            dbConn = config['DBConnection']
        except KeyError:
            raise KeyError, "Please specify the DB connection parameters in the configuration under 'DBConnection'."
        
        # experiments
        try:
            experiments = config['experiments']
        except KeyError:
            raise KeyError, "Please specify the experiment(s) in the configuration under 'experiments'."
        
        return DBUnit(dbConn, mapper, experiments)
        
    elif source == 'HDF5':
        # get the path
        try:
            path = config['path']
        except KeyError:
            raise KeyError, "Please specify the path to the HDF5 files in the configuration under 'path'."
        
        # get the experiments
        try:
            experiments = config['experiments']
        except KeyError:
            raise KeyError, "Please specify the experiment(s) in the configuration under 'experiments'."
        
        return H5Unit(path, mapper, experiments)
            
    elif source == 'MonitorPlux':
        # get the path
        try:
            path = config['path']
        except KeyError:
            raise KeyError, "Please specify the path to the MonitorPlux files in the configuration under 'path'."
        
        return MPUnit(path, mapper)
            
    elif source == 'gzip':
        # get the path
        try:
            path = config['path']
        except KeyError:
            raise KeyError, "Please specify the path to the gzip files in the configuration under 'path'."
        
        # return GZUnit(path, mapper)
        raise ValueError, "'MonitorPlux' not yet implemented."
    else:
        raise TypeError, "Unsupported source type."



class DBUnit:
    """
    StorageBIT (online) database manager class.
    
    Input:
        None
    
    Output:
        None
    
    Configurable fields:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    """
    
    def __init__(self, dbConn, mapper, experiments):
        """
        Instantiate the class.
        
        Input:
            dbConn (dict): The DB connection parameters.
            
            mapper (dict): To map user variables to the database representation.
            
            experiments (string or list of strings): The experiment(s) to load.
        
        Output:
            None
        
        Configurable fields:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            dbConn = {'host': '193.136.222.234',
                    'dstPath': '/path/to/files',
                    'dbName': 'database'}
            mapper = {'raw': 'signals/ECG/hand/raw',
                      'filtered': 'signals/ECG/hand/filtered'}
            experiments = ['T1', 'T2', 'test']
            store = DBUnit(dbConn, mapper, experiments)
        
        """
        
        # connect to db
        # dbConn.update({'sync': True})
        db = biomesh.biomesh(**dbConn)
        
        # get the subjects
        subjects = []
        if isinstance(experiments, list):
            pass
        elif isinstance(experiments, basestring):
            experiments = [experiments]
        else:
            raise TypeError, "Unsupported type in 'experiments' key."
        
        for exp in experiments:
            aux = db.subsInExp(exp)['idList']
            subjects = subjects + list(set(aux) - set(subjects))
        
        subjects.sort()
        
        # get the records (sorted)
        records = db.records.getAll(restrict={'experiment': experiments})['idList']
        
        # self things
        self.db = db
        self.mapper = mapper
        self.experiments = experiments
        self.subjects = subjects
        self.records = records
    
    
    def dbmetada(self, refine=None):
        """
        Get metadata of records from database. Refine rules may be applied.
        
        Input:
            refine (dict): Refine rules
        
        Output:
            output (dict): Output metadata where keys correspond to record id numbers.
        
        Configurable fields:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            output = store.dbmetada()
            output is {0: record 0 metadata, 1: record 1 metadata, ...}
            output = store.dbmetada(refine={'_id': [0]})
            output is {0: record 0 metadata}
        
        """
        
        output = {}
        
        if refine is None:
            records = self.records
        else:
            records = refine['_id']
        
        for rec in records:
            output[rec] = self.db.records.getById(rec, {'signals': 0, 'events': 0})['doc']
        
        return output
    
    
    def data2db(self, data, data_type, updateDuration=False):
        """
        Insert data of specified types in the database.
        
        Input:
            data (dict): Input data where keys correspond to record IDs.
                         There is also one key 'info' with specific information about the data.
            
            data_type (list of strings): Input data types to be added.
            
            updateDuration (bool): Flag to update the duration of the record. Default: False.
        
        Output:
            None
        
        Configurable fields:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            data = {'info': {'smapleRate': 1000.}, 0: {'raw': [0, 1, 2], 'filtered': [2, 3, 4]}, 1: {'raw': [0, 1, 2], 'filtered': [2, 3, 4]}}
            store.data2db(data, ['raw', 'filtered'])
        
        """
        
        # metadata
        try:
            mdata = data.pop('info')
        except KeyError:
            raise KeyError, "No info."
        
        for tpe in data_type:
            try:
                weg = self.mapper[tpe]
            except KeyError:
                warnings.warn("The key %s does not exist in the mapper.\nCorresponding data will NOT be saved." % str(data_type))
                continue
            
            wegits = weg.split('/')
            case = wegits[0]
            weg = '/' + '/'.join(wegits[1:])
            
            # cycle the records
            for key, value in data.iteritems():
                try:
                    aux = value[tpe]
                except KeyError:
                    continue
                if case == 'signals':
                    self.db.records.addSignal(key, weg, aux, mdata, updateDuration=updateDuration)
                elif case == 'events':
                    self.db.records.addEvent(key, weg, aux, [], mdata)
    
    
    def db2data(self, data_type, refine=None, tasks=False, mdataSplit=False, concatenate=False, axis=0):
        """
        Retrieves data of the specified type from database. Refine rules may be applied.
        
        Input:
            data_type (string): The data type to retrieve.
            
            refine (dict): Refine retrieved information.
            
            tasks (bool): If True, also return task indices for each data element.
            
            mdataSplit (bool): Flag to separate the metadata from the signals or events.
            
            concatenate (bool): If True, concatenate the existing datasets into a single array, along axis.
            
            axis (int): Axis along which to concatenate the datasets.
        
        Output:
            output (dict): Output data of data_type where keys correspond to record id numbers.
        
        Configurable fields:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            output = store.db2data('raw')
            output is {0: [{'signal': [0, 1, 2], 'mdata': {'sampleRate': 1000.}}, ...], 1: ...}
            output = store.db2data('raw', refine={'_id': [0]})
            output is {0: [{'signal': [0, 1, 2], 'mdata': {'sampleRate': 1000.}}, ...]}
            output = store.db2data('R', refine={'_id': [0]})
            output is {0: [{'timeStamps': [0, 1, 2], 'mdata': {'sampleRate': 1000.}}, ...]}
        
        """
        
        try:
            weg = self.mapper[data_type]
        except KeyError:
            raise KeyError, "The key %s does not exist in the mapper." % str(data_type)
        
        # auxiliary functions
        def auxSignals():
            pass
        def auxEvents():
            pass
        
        # determine the case
        wegits = weg.split('/')
        case = wegits[0]
        wegC = '/' + '/'.join(wegits[1:])
        
        if case == 'signals':
            if mdataSplit:
                def fcnGet(out, md, rec, weg, ref):
                    data = self.db.records.getSignal(rec, weg, ref)
                    md.append(data.pop('mdata'))
                    out.append(data['signal'])
                    return None
            else:
                def fcnGet(out, md, rec, weg, ref):
                    data = self.db.records.getSignal(rec, weg, ref)
                    out.append(data)
                    return None
        elif case == 'events':
            if mdataSplit:
                def fcnGet(out, md, rec, weg, ref):
                    data = self.db.records.getEvent(rec, weg, ref)
                    md.append(data.pop('mdata'))
                    out.append(data)
                    return None
            else:
                def fcnGet(out, md, rec, weg, ref):
                    data = self.db.records.getEvent(rec, weg, ref)
                    out.append(data)
                    return None
        
        # get the records
        if refine is None:
            records = self.records
        else:
            records = refine['_id']
        
        output = {}
        mdata = {}
        for rec in records:
            try:
                aux = self.db.records[rec][weg].list()['local']
            except KeyError:
                continue
            output[rec] = []
            mdata[rec] = []
            for item in aux:
                fcnGet(output[rec], mdata[rec], rec, wegC, item[1])
        
        # concatenate
        if concatenate:
            if not mdataSplit:
                raise ValueError, "In order to concatenate the data, the metadata splitting flag must be set to True."
            elif case == 'events':
                raise ValueError, "Concatenation does not work with Events, only with signals."
            for rec in output.keys():
                data = tuple(output[rec])
                output[rec] = np.concatenate(data, axis=axis)
        
        # tasks
        if tasks:
            taskIDs = taskify(output)
            if mdataSplit:
                return output, taskIDs, mdata
            else:
                return output, taskIDs
        else:
            if mdataSplit:
                return output, mdata
            else:
                return output
    
    
    def subjectTTSets(self, info):
        """
        Returns subjects information according to the parameters in info.
        
        Input:
            info (dict): Input dict with the following keys:
                'train_set' (list): train tags
                'test_set' (list): test tags
                'train_time' (tuple of ints): train record time in seconds
                'test_time' (tuple of ints): test record time in seconds
        
        Output:
            output (list): Output list where each entry is tuple (subject id, train record id(s), test record id(s)).
        
        Configurable fields:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            info = {'train_set': ['T1'],
                    'test_set': ['T2'],
                    'train_time': (0, 60),
                    'test_time': (0, 60)}
            output = store.subjectTTSets(info)
            output is [(0, [0, 1], [2]), (1, [3], [4, 5]), (2, [6], [7]), ...]
        
        """
        
        try:
            train_set = info['train_set']
        except KeyError:
            raise KeyError, "No train_set."
        
        try:
            test_set = info['test_set']
        except KeyError:
            raise KeyError, "No test_set."
        
        try:
            train_time = info['train_time']
        except KeyError:
            raise KeyError, "No train_time."
        
        try:
            test_time = info['test_time']
        except KeyError:
            raise KeyError, "No test_time."
        
        # get train records
        train_recs = self.db.records.listAndTags(train_set)['idList']
        train_recs_meta = self.dbmetada(refine={'_id': train_recs})
        
        # verify duration
        for item in train_recs_meta:
            duration = train_recs_meta[item]['duration']
            if duration < train_time[1]:
                train_recs.remove(item)
        
        # get the subjects
        subTrn = {}
        for rec in train_recs:
            sub = train_recs_meta[rec]['subject']
            try:
                subTrn[sub].append(rec)
            except KeyError:
                subTrn[sub] = [rec]
        
        # get test records
        test_recs = self.db.records.listAndTags(test_set)['idList']
        test_recs_meta = self.dbmetada(refine={'_id': test_recs})
        
        # verify duration
        for item in test_recs_meta:
            duration = test_recs_meta[item]['duration']
            if duration < test_time[1]:
                test_recs.remove(item)
        
        # get the subjects
        subTst = {}
        for rec in test_recs:
            sub = test_recs_meta[rec]['subject']
            try:
                subTst[sub].append(rec)
            except KeyError:
                subTst[sub] = [rec]
        
        # harmonize the users
        users = set(subTrn.keys()).intersection(set(subTst.keys()))
        
        # finally build output
        output = [(item, subTrn[item], subTst[item]) for item in users]
        
        return output
    
    
    def getDataSet(self, tags=None):
        """
        Returns the IDs of the records that have the given tags (AND operator).
        
        Input:
            tags (list): List of tags.
        
        Output:
            output (list): Output list with the record IDs.
        
        Configurable fields:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            output = store.getDataSet(['Tag 1', 'Tag 2', ...])
            output is e.g. [0, 1, 2, ...]
        
        """
        
        # check inputs
        if tags is None:
            raise TypeError, "Please specify the list of tags."
        
        output = set(self.db.records.listAndTags(tags)['idList'])
        # limit to preloaded records
        output = list(output.intersection(set(self.records)))
        # sort
        output.sort()
        
        return output
    
    
    def getTTSets(self, data, tasks, trainTags, testTags):
        """
        Split the data and tasks dictionaries into train and test sets.
        
        Input:
            data (dict): Data dictionary.
            
            tasks (dict): Tasks dictionary.
            
            trainTags (list): List of tags for the train set.
            
            testTags (list): List of tags for the test set.
        
        Output:
            output (dict): Output dictionary with keys:
                trainData (dict): Train data.
                trainTasks (dict): Train tasks.
                testData (dict): Test data.
                testTasks (dict): Test tasks.
        
        Configurable fields:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        """
        
        # train
        trainRecs = self.getDataSet(trainTags)
        trainData = {}
        trainTasks = {}
        for rec in trainRecs:
            trainData[rec] = data[rec]
            trainTasks[rec] = tasks[rec]
        
        # test
        testRecs = self.getDataSet(testTags)
        testData = {}
        testTasks = {}
        for rec in testRecs:
            testData[rec] = data[rec]
            testTasks[rec] = tasks[rec]
        
        output = {'trainData': trainData,
                  'trainTasks': trainTasks,
                  'testData': testData,
                  'testTasks': testTasks}
        
        return output
    
    
    def getSubjectInfo(self, refine=None):
        """
        Get metadata of subjects from database. Refine rules may be applied.
        
        Input:
            refine (dict): Refine rules
        
        Output:
            output (dict): Output metadata where keys correspond to subject id numbers.
        
        Configurable fields:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            output = store.getSubjectInfo()
            output is {0: subject 0 metadata, 1: subject 1 metadata, ...}
            output = store.getSubjectInfo(refine={'_id': [0]})
            output is {0: subject 0 metadata}
        
        """
        
        output = {}
        
        if refine is None:
            subjects = self.subjects
        else:
            subjects = refine['_id']
        
        for sub in subjects:
            output[sub] = self.db.subjects.getById(sub)['doc']
        
        return output
    
    
    def addRecord(self, header):
        """
        Add a new record to the database.
        
        Input:
            header (dict): The new record to add. Has to have 'subject' and 'experiment' keys.
        
        Output:
            recordId (int): The ID of the new record.
        
        Configurable fields:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            header = {'experiment': 'test', 'subject': 1, 'date': 'yyyy-mm-ddThh:mm:ss'}
            store.addRecord(header)
        
        """
        
        recordId = self.db.records.add(header)['recordId']
        
        # self things
        if header['experiment'] in self.experiments:
            self.records.append(recordId)
            self.records.sort()
            if header['subject'] not in self.subjects:
                self.subjects.append(header['subject'])
                self.subjects.sort()
        
        return recordId
    
    
    def addExperiment(self, experiment=None):
        """
        Add a new experiment to the database.
        
        Input:
            experiment (dict): The new experiment to add. Has to have the key 'name'.
        
        Output:
            experimentName (str): The name of the new experiment.
        
        Configurable fields:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            experiment = {'name': 'Test', 'goals': 'To test the addition of experiments.', 'description': 'This is a sample experiment.'}
            experimentName = store.addExperiment(experiment)
        
        """
        
        # add experiment to db
        self.db.experiments.add(experiment)
        
        # self things
        experimentName = experiment['name']
        if experimentName not in self.experiments:
            self.experiments.append(experimentName)
        
        return experimentName
    
    
    def addSubject(self, subject=None):
        """
        Add a new subject to the database.
        
        Input:
            experiment (dict): The new subject to add. Has to have the key 'name'.
        
        Output:
            subjectId (int): The ID of the new subject.
        
        Configurable fields:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            subject = {'name': 'John Smith', 'birthdate': 'yyyy-mm-ddThh:mm:ss', 'email': 'john.smith@domain.com'}
            subjectId = store.addExperiment(subject)
        
        """
        
        # add subject to db
        subjectId = self.db.subjects.add(subject)['subjectId']
        
        # self things
        if subjectId not in self.subjects:
            self.subjects.append(subjectId)
            self.subjects.sort()
        
        return subjectId
    
    
    def close(self):
        """
        Close the manager.
        
        Input:
            None
        
        Output:
            None
        
        Configurable fields:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            store.close()
        
        """
        
        # close the db
        self.db.close()



class H5Unit:
    """
    StorageBIT (offline) database manager class.
    
    Input:
        None
    
    Output:
        None
    
    Configurable fields:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    """
    
    def __init__(self, path, mapper, experiments):
        """
        Instantiate the class.
        
        Input:
            path (string): The path to the HDF5 files.
            
            mapper (dict): To map user variables to the database representation.
            
            experiments (string or list of strings): The experiment(s) to load.
        
        Output:
            None
        
        Configurable fields:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            path = '/path/to/files'
            mapper = {'raw': 'signals/ECG/hand/raw',
                      'filtered': 'signals/ECG/hand/filtered'}
            experiments = ['T1', 'T2', 'test']
            store = H5Unit(dbConn, mapper, experiments)
        
        """
        
        # expand the user and/or properly format the path
        if '~' in path:
            path = os.path.abspath(os.path.expanduser(path))
        else:
            path = os.path.abspath(path)
        
        # check if folder exists, create otherwise
        if not os.path.exists(path):
            os.makedirs(path)
        
        # check if ExpSub exists
        metaFile = os.path.join(path, 'ExpSub.hdf5')
        if not os.path.exists(metaFile):
            # add the ExpSub file
            with h5db.meta(metaFile) as fid:
                fid.setDB('None')
                fid.addSubject({'_id':0, 'name': 'unnamed', 'records': []})
                fid.addExperiment({'_id':0, 'name': 'unknown', 'description': 'Generic experiment.', 'goals': 'To store records from unknown experiments.', 'records':[]})
        
        # get subjects and records
        subjects = []
        records = []
        with h5db.meta(metaFile) as meta:
            if isinstance(experiments, list):
                pass
            elif isinstance(experiments, basestring):
                experiments = [experiments]
            else:
                raise TypeError, "Unsupported type in 'experiments' key."
            
            for exp in experiments:
                doc = meta.getExperiment(exp)['experiment']
                try:
                    doc = doc['records']
                except KeyError:
                    continue
                else:
                    aux = []
                    for item in doc:
                        aux.append(item['subject'])
                        records.extend(item['list'])
                    subjects = subjects + list(set(aux) - set(subjects))
        
        subjects.sort()
        records.sort()
        
        # self things
        self.path = path
        self.mapper = mapper
        self.experiments = experiments
        self.subjects = subjects
        self.records = records
    
    
    def dbmetada(self, refine=None):
        """
        Get metadata of records from database. Refine rules may be applied.
        
        Input:
            refine (dict): Refine rules
        
        Output:
            output (dict): Output metadata where keys correspond to record id numbers.
        
        Configurable fields:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            output = store.dbmetada()
            output is {0: record 0 metadata, 1: record 1 metadata, ...}
            output = store.dbmetada(refine={'_id': [0]})
            output is {0: record 0 metadata}
        
        """
        
        output = {}
        
        if refine is None:
            records = self.records
        else:
            records = refine['_id']
        
        for rec in records:
            try:
                with h5db.hdf(os.path.join(self.path, 'rec_%d.hdf5' % rec), 'r') as fid:
                    output[rec] = fid.getInfo()['header']
            except IOError:
                output[rec] = None
        
        return output
    
    
    def data2db(self, data, data_type, updateDuration=False):
        """
        Insert data of specified types in the database.
        
        Input:
            data (dict): Input data where keys correspond to record IDs.
                         There is also one key 'info' with specific information about the data.
            
            data_type (list of strings): Input data types to be added.
            
            updateDuration (bool): Flag to update the duration of the record. Default: False.
        
        Output:
            None
        
        Configurable fields:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            data = {'info': {'smapleRate': 1000.}, 0: {'raw': [0, 1, 2], 'filtered': [2, 3, 4]}, 1: {'raw': [0, 1, 2], 'filtered': [2, 3, 4]}}
            store.data2db(data, ['raw', 'filtered'])
        
        """
        
        # metadata
        try:
            mdata = data.pop('info')
        except KeyError:
            raise KeyError, "No info."
        
        # data type
        for tpe in data_type:
            try:
                weg = self.mapper[tpe]
            except KeyError:
                warnings.warn("The key %s does not exist in the mapper.\nCorresponding data will NOT be saved." % str(data_type))
                continue
            
            wegits = weg.split('/')
            case = wegits[0]
            weg = '/' + '/'.join(wegits[1:])
            
            mdata.update({'type': weg})
            
            # cycle the records
            for key, value in data.iteritems():
                try:
                    aux = value[tpe]
                except KeyError:
                    continue
                with h5db.hdf(os.path.join(self.path, 'rec_%d.hdf5' % key)) as fid:
                    if case == 'signals':
                        signals = fid.listSignals(weg)['signalsList']
                        name = 'signal%d' % len(signals)
                        mdata.update({'name': name})
                        source = fid.addSignal(weg, aux, mdata, name)
                        
                        header = fid.getInfo()['header']
                       
                        # duration
                        if updateDuration:
                            try:
                                Fs = float(mdata['sampleRate'])
                            except KeyError:
                                pass
                            else:
                                header.update({'duration': len(aux) / Fs, 'durationReference': source})
                        # add tags
                        try:
                            tags = header['tags']
                        except KeyError:
                            tags = wegits[1:]
                        else:
                            tags = list(set(tags) | set(wegits[1:]))
                        header['tags'] = tags
                        
                        fid.addInfo(header)
                    elif case == 'events':
                        events = fid.listEvents(weg)['eventsList']
                        name = 'event%d' % len(events)
                        mdata.update({'name': name})
                        fid.addEvent(weg, aux, [], mdata, name)
                        # add tags
                        header = fid.getInfo()['header']
                        try:
                            tags = header['tags']
                        except KeyError:
                            tags = wegits[1:]
                        else:
                            tags = list(set(tags) | set(wegits[1:]))
                        header['tags'] = tags
                        fid.addInfo(header)
    
    
    def db2data(self, data_type, refine=None, tasks=False, mdataSplit=False, concatenate=False, axis=0):
        """
        Retrieves data of the specified type from database. Refine rules may be applied.
        
        Input:
            data_type (string): The data type to retrieve.
            
            refine (dict): Refine retrieved information.
            
            tasks (bool): If True, also return task indices for each data element.
            
            mdataSplit (bool): Flag to separate the metadata from the signals or events.
            
            concatenate (bool): If True, concatenate the existing datasets into a single array, along axis.
            
            axis (int): Axis along which to concatenate the datasets.
        
        Output:
            output (dict): Output data of data_type where keys correspond to record id numbers.
        
        Configurable fields:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            output = store.db2data('raw')
            output is {0: [{'signal': [0, 1, 2], 'mdata': {'sampleRate': 1000.}}, ...], 1: ...}
            output = store.db2data('raw', refine={'_id': [0]})
            output is {0: [{'signal': [0, 1, 2], 'mdata': {'sampleRate': 1000.}}, ...]}
            output = store.db2data('R', refine={'_id': [0]})
            output is {0: [{'timeStamps': [0, 1, 2], 'mdata': {'sampleRate': 1000.}}, ...]}
        
        """
        
        try:
            weg = self.mapper[data_type]
        except KeyError:
            raise KeyError, "The key %s does not exist in the mapper." % str(data_type)
        
        # determine the case
        wegits = weg.split('/')
        case = wegits[0]
        weg = '/' + '/'.join(wegits[1:])
        
        if case == 'signals':
            if mdataSplit:
                def fcnGet(out, md, fname, weg):
                    with h5db.hdf(fname, 'r') as fid:
                        aux = fid.listSignals(weg)['signalsList']
                        for item in aux:
                            data = fid.getSignal(weg, item)
                            md.append(data.pop('mdata'))
                            out.append(data['signal'])
                    return None
            else:
                def fcnGet(out, md, fname, weg):
                    with h5db.hdf(fname, 'r') as fid:
                        aux = fid.listSignals(weg)['signalsList']
                        for item in aux:
                            data = fid.getSignal(weg, item)
                            out.append(data)
                    return None
        elif case == 'events':
            if mdataSplit:
                def fcnGet(out, md, fname, weg):
                    with h5db.hdf(fname, 'r') as fid:
                        aux = fid.listEvents(weg)['eventsList']
                        for item in aux:
                            data = fid.getEvent(weg, item)
                            md.append(data.pop('mdata'))
                            out.append(data)
                    return None
            else:
                def fcnGet(out, md, fname, weg):
                    with h5db.hdf(fname, 'r') as fid:
                        aux = fid.listEvents(weg)['eventsList']
                        for item in aux:
                            data = fid.getEvent(weg, item)
                            out.append(data)
                    return None
        
        # get the records
        if refine is None:
            records = self.records
        else:
            records = refine['_id']
            
        output = {}
        mdata = {}
        for rec in records:
            output[rec] = []
            mdata[rec] = []
            fname = os.path.join(self.path, 'rec_%d.hdf5' % rec)
            try:
                fcnGet(output[rec], mdata[rec], fname, weg)
            except IOError:
                continue
        
        # concatenate
        if concatenate:
            if not mdataSplit:
                raise ValueError, "In order to concatenate the data, the metadata splitting flag must be set to True."
            elif case == 'events':
                raise ValueError, "Concatenation does not work with Events, only with signals."
            for rec in output.keys():
                data = tuple(output[rec])
                output[rec] = np.concatenate(data, axis=axis)
        
        # tasks
        if tasks:
            taskIDs = taskify(output)
            if mdataSplit:
                return output, taskIDs, mdata
            else:
                return output, taskIDs
        else:
            if mdataSplit:
                return output, mdata
            else:
                return output
    
    
    def subjectTTSets(self, info):
        """
        Returns subjects information according to the parameters in info.
        
        Input:
            info (dict): Input dict with the following keys:
                'train_set' (list): train tags
                'test_set' (list): test tags
                'train_time' (tuple of ints): train record time in seconds
                'test_time' (tuple of ints): test record time in seconds
        
        Output:
            output (list): Output list where each entry is tuple (subject id, train record id(s), test record id(s)).
        
        Configurable fields:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            info = {'train_set': ['T1'],
                    'test_set': ['T2'],
                    'train_time': (0, 60),
                    'test_time': (0, 60)}
            output = store.subjectTTSets(info)
            output is [(0, [0, 1], [2]), (1, [3], [4, 5]), (2, [6], [7]), ...]
        
        """
        
        try:
            train_set = info['train_set']
        except KeyError:
            raise KeyError, "No train_set."
        
        try:
            test_set = info['test_set']
        except KeyError:
            raise KeyError, "No test_set."
        
        try:
            train_time = info['train_time']
        except KeyError:
            raise KeyError, "No train_time."
        
        try:
            test_time = info['test_time']
        except KeyError:
            raise KeyError, "No test_time."
        
        # get all records
        records = self.dbmetada()
        
        # get train and test records and verify duration
        subTrn = {}
        subTst = {}
        for item in records:
            try:
                tags = records[item]['tags']
                duration = records[item]['duration']
                subject = records[item]['subject']
            except KeyError:
                continue
            if all(x in tags for x in train_set) and (duration >= train_time[1]):
                try:
                    subTrn[subject].append(item)
                except KeyError:
                    subTrn[subject] = [item]
            if all(x in tags for x in test_set) and (duration >= test_time[1]):
                try:
                    subTst[subject].append(item)
                except KeyError:
                    subTst[subject] = [item]
        
        # harmonize the users
        users = set(subTrn.keys()).intersection(set(subTst.keys()))
        
        # finally build output
        output = [(item, subTrn[item], subTst[item]) for item in users]
        
        return output
    
    
    def getDataSet(self, tags=None):
        """
        Returns the IDs of the records that have the given tags (AND operator).
        
        Input:
            tags (list): List of tags.
        
        Output:
            output (list): Output list with the record IDs.
        
        Configurable fields:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            output = store.getDataSet(['Tag 1', 'Tag 2', ...])
            output is e.g. [0, 1, 2, ...]
        
        """
        
        # check inputs
        if tags is None:
            raise TypeError, "Please specify the list of tags."
        
        # get all records
        records = self.dbmetada()
        
        output = []
        for item in records:
            try:
                recTags = item['tags']
            except KeyError:
                continue
            if all(x in recTags for x in tags):
                output.append(item['_id'])
        # sort
        output.sort()
        
        return output
    
    
    def getTTSets(self, data, tasks, trainTags, testTags):
        """
        Split the data and tasks dictionaries into train and test sets.
        
        Input:
            data (dict): Data dictionary.
            
            tasks (dict): Tasks dictionary.
            
            trainTags (list): List of tags for the train set.
            
            testTags (list): List of tags for the test set.
        
        Output:
            output (dict): Output dictionary with keys:
                trainData (dict): Train data.
                trainTasks (dict): Train tasks.
                testData (dict): Test data.
                testTasks (dict): Test tasks.
        
        Configurable fields:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        """
        
        # train
        trainRecs = self.getDataSet(trainTags)
        trainData = {}
        trainTasks = {}
        for rec in trainRecs:
            trainData[rec] = data[rec]
            trainTasks[rec] = tasks[rec]
        
        # test
        testRecs = self.getDataSet(testTags)
        testData = {}
        testTasks = {}
        for rec in testRecs:
            testData[rec] = data[rec]
            testTasks[rec] = tasks[rec]
        
        output = {'trainData': trainData,
                  'trainTasks': trainTasks,
                  'testData': testData,
                  'testTasks': testTasks}
        
        return output
    
    
    def getSubjectInfo(self, refine=None):
        """
        Get metadata of subjects from database. Refine rules may be applied.
        
        Input:
            refine (dict): Refine rules
        
        Output:
            output (dict): Output metadata where keys correspond to subject id numbers.
        
        Configurable fields:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            output = store.getSubjectInfo()
            output is {0: subject 0 metadata, 1: subject 1 metadata, ...}
            output = store.getSubjectInfo(refine={'_id': [0]})
            output is {0: subject 0 metadata}
        
        """
        
        output = {}
        
        if refine is None:
            subjects = self.subjects
        else:
            subjects = refine['_id']
        
        for sub in subjects:
            with h5db.meta(os.path.join(self.path, 'ExpSub.hdf5')) as fid:
                output[sub] = fid.getSubject(sub)['subject']
        
        return output
    
    
    def addRecord(self, header):
        """
        Add a new record to the database.
        
        Input:
            header (dict): The new record to add. Has to have 'subject' and 'experiment' keys.
        
        Output:
            recordId (int): The ID of the new record.
        
        Configurable fields:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            header = {'experiment': 'test', 'subject': 1, 'date': 'yyyy-mm-ddThh:mm:ss'}
            store.addRecord(header)
        
        """
        
        # generate ID
        recordId = len(glob.glob(os.path.join(self.path, 'rec_*.hdf5')))
        npath = os.path.join(self.path, 'rec_%d.hdf5' % recordId)
        header.update({'_id': recordId})
        
        # check header
        try:
            experiment = header['experiment']
        except KeyError:
            print("Record not assigned to an experiment.\nStoring in 'unknown' experiments.")
            header.update({'experiment': 'unknown'})
            experiment = 'unknown'
        
        try:
            subject = header['subject']
        except KeyError:
            print("Record has no subject.\nStoring in 'unnamed' subject.")
            header.update({'subject': 0})
            subject = 0
        
        # check if experiment and subject exist in database
        with h5db.meta(os.path.join(self.path, 'ExpSub.hdf5')) as fidM:
            # experiment
            if isinstance(experiment, int):
                # we have to search in the database
                found = False
                subList = fidM.listSubjects()
                for item in subList:
                    if item['_id'] == experiment:
                        # found it
                        found = True
                        break
                if found:
                    experiment = item['name']
                    header.update({'experiment': experiment})
                else:
                    raise ValueError, "Experiment %s does not exist in the database.\nAdd it before the record." % str(experiment)
            else:
                # just try
                try:
                    fidM.getExperiment(experiment)['experiment']['name']
                except KeyError:
                    raise ValueError, "Experiment %s does not exist in the database.\nAdd it before the record." % str(experiment)
            
            if isinstance(subject, basestring):
                # we have to search the database
                found = False
                expList = fidM.listExperiment()
                for item in expList:
                    if item['_name'] == subject:
                        # found it
                        found = True
                        break
                if found:
                    subject = item['_id']
                    header.update({'subjet': subject})
                else:
                    raise ValueError, "Subject %s does not exist in the database.\nAdd it before the record." % str(subject)
            else:
                # just try
                try:
                    fidM.getSubject(subject)['subject']['_id']
                except KeyError:
                    raise TypeError, "Subject %s does not exist in the database.\nAdd it before the record." % str(subject)
            
            # add record to experiment and subject and generate session number
            expDoc = fidM.getExperiment(experiment)['experiment']
            # find the subject
            found = False
            for item in expDoc['records']:
                if item['subject'] == subject:
                    # found it
                    found = True
                    break
            if found:
                session = len(item['list'])
                item['list'].append(recordId)
            else:
                expDoc['records'].append({'subject': subject, 'list': [recordId]})
                session = 0
            fidM.updateExperiment(experiment, expDoc)
            
            subDoc = fidM.getSubject(subject)['subject']
            # find the experiment
            found = False
            for item in subDoc['records']:
                if item['experiment'] == experiment:
                    # found it
                    found = True
                    break
            if found:
                item['list'].append(recordId)
            else:
                subDoc['records'].append({'experiment': experiment, 'list': [recordId]})
            fidM.updateSubject(subject, subDoc)
        
        # tags
        if not header.has_key('tags'):
            header.update({'tags': [experiment]})
        else:
            tags = set(header['tags'])
            tags.add(experiment)
            header['tags'] = list(tags)
        
        # duration
        if not header.has_key('duration'):
            header.update({'duration': np.inf})
        
        # save the new record
        header.update({'session': session})
        with h5db.hdf(npath) as fid:
            fid.addInfo(header)
        
        # self things
        if header['experiment'] in self.experiments:
            self.records.append(recordId)
            self.records.sort()
            if header['subject'] not in self.subjects:
                self.subjects.append(header['subject'])
                self.subjects.sort()
        
        return recordId
    
    
    def addExperiment(self, experiment=None):
        """
        Add a new experiment to the database.
        
        Input:
            experiment (dict): The new experiment to add. Has to have the key 'name'.
        
        Output:
            experimentName (str): The name of the new experiment.
        
        Configurable fields:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            experiment = {'name': 'Test', 'goals': 'To test the addition of experiments.', 'description': 'This is a sample experiment.'}
            experimentName = store.addExperiment(experiment)
        
        """
        
        # check inputs
        if experiment is None:
            raise TypeError, "A JSON experiment must be provided."
        
        # avoid non-ascii characters
        experimentName = h5db.latin1ToAscii(experiment['name'])
        
        with h5db.meta(os.path.join(self.path, 'ExpSub.hdf5')) as fid:
            # check if experiment already exists (by name)
            expList = fid.listExperiments()['expList']
            expList = [item['name'] for item in expList]
            if experimentName in expList:
                print "Warning: experiment already exists in DB; skipping new insertion."
                return experimentName
            if not experiment.has_key('_id'):
                # create a new id (starts at zero)
                newId = len(expList)
                experiment.update({'_id': newId})
            if not experiment.has_key('records'):
                # add the "records" field
                experiment.update({'records': []})
            # add experiment
            fid.addExperiment(experiment)
        
        # self things
        if experimentName not in self.experiments:
            self.experiments.append(experimentName)
        
        return experimentName
    
    
    def addSubject(self, subject=None):
        """
        Add a new subject to the database.
        
        Input:
            experiment (dict): The new subject to add. Has to have the key 'name'.
        
        Output:
            subjectId (int): The ID of the new subject.
        
        Configurable fields:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            subject = {'name': 'John Smith', 'birthdate': 'yyyy-mm-ddThh:mm:ss', 'email': 'john.smith@domain.com'}
            subjectId = store.addExperiment(subject)
        
        """
        
        # check inputs
        if subject is None:
            raise TypeError, "A JSON subject must be provided."
        
        # avoid non-ascii characters
        subjectName = h5db.latin1ToAscii(subject['name'])
        
        with h5db.meta(os.path.join(self.path, 'ExpSub.hdf5')) as fid:
            # check if subject aleady exists (by name)
            subList = fid.listSubjects()['subList']
            subDict = {}
            for item in subList:
                subDict[item['name']] = item['_id']
            if subjectName in subDict.keys():
                print "Warning: subject already exists in DB; skipping new insertion."
                return subDict[subjectName]
            if not subject.has_key('_id'):
                # get a new id (starts at zero)
                newId = len(subList)
                subject.update({'_id':newId})
            
            if not subject.has_key('records'):
                # add the "records" field
                subject.update({'records':[]})
            # add the subject
            fid.addSubject(subject)
        
        # self things
        if subject['_id'] not in self.subjects:
            self.subjects.append(subject['_id'])
            self.subjects.sort()
        
        return subject['_id']
    
    
    def close(self):
        """
        Close the manager.
        
        Input:
            None
        
        Output:
            None
        
        Configurable fields:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            store.close()
        
        """
        
        return



class MPUnit:
    # MonitorPlux
    
    def __init__(self, path, mapper):
        raise ValueError, "'MonitorPlux' not yet implemented."


def zipArchiveStore(srcPath, zipPath):
    # store the contents of a directory into a zip file
    # do not include the extension in the zip file
    
    # save current working dir
    save_cwd = os.getcwd()
    
    # change working dir to source
    os.chdir(srcPath)
    base_dir = os.curdir
    
    # open zip file
    zipPath += ".zip"
    with zipfile.ZipFile(zipPath, 'w', allowZip64=True) as fid:
        for dirpath, dirnames, filenames in os.walk(base_dir):
            for d in dirnames:
                path = os.path.normpath(os.path.join(dirpath, d))
                if os.path.isdir(path):
                    fid.write(path, path)
            
            for name in filenames:
                path = os.path.normpath(os.path.join(dirpath, name))
                if os.path.isfile(path):
                    fid.write(path, path)
    
    # restore working dir
    os.chdir(save_cwd)
    
    return zipPath


def zipArchiveLoad(zipPath, dstPath):
    # extract the contents of a zip file into a directory
    # do not include the extension in the zip file
    
    with zipfile.ZipFile(zipPath + '.zip', 'r', allowZip64=True) as fid:
        fid.extractall(dstPath)


def gzStore(path, data):
    with gzip.open(path, 'wb') as fid:
        cPickle.dump(data, fid)


def gzLoad(path):
    with gzip.open(path, 'rb') as fid:
        data = cPickle.load(fid)
    
    return data


def cpStore(path, data):
    with open(path, 'wb') as fid:
        cPickle.dump(data, fid)


def cpLoad(path):
    with open(path, 'rb') as fid:
        data = cPickle.load(fid)
    
    return data


def skStore(path, data, compress=3):
    # opimized pickle for numpy arrays
    
    joblib.dump(data, path, compress=compress)

def skLoad(path):
    # optimized pickle for numpy arrays
    
    return joblib.load(path)


def allocH5(path):
    with h5py.File(path):
        pass


def h5Store(path, label, data):
    try:
        fid = h5py.File(path)
        label = str(label)
        try:
            fid.create_dataset(label, data=data)
        except (RuntimeError, ValueError):
            del fid[label]
            fid.create_dataset(label, data=data)
    finally:
        fid.close()


def h5Load(path, label):
    try:
        fid = h5py.File(path)
        label = str(label)
        try:
            return fid[label][...]
        except KeyError:
            return None
    finally:
        fid.close()


def csvStore(path, data):
    # save numpy array to csv file
    
    # ensure numpy
    data = np.array(data)
    
    # data type
    dtype = data.dtype.name
    if 'string' in dtype:
        dtype = 'string'
    
    # add dtype to path name
    if '.csv' in path:
        pass
    else:
        path += '.' + dtype + '.csv'
    
    path = os.path.abspath(path)
    
    # write to file
    with open(path, 'wb') as fid:
        writer = csv.writer(fid, delimiter=',', quotechar='"')
        
        # write rows
        writer.writerows(data)
    
    return path


def csvLoad(path):
    # load numpy array from csv file
    
    # get data type
    dtype = path.split('.')[-2]
    
    # read from file
    data = []
    with open(path, 'rb') as fid:
        reader = csv.reader(fid)
        
        for row in reader:
            data.append(row)
    
    # convert to numpy
    data = np.array(data, dtype=dtype)
    
    return data


def fileBenchmark(path, it):
    # quick benchmark for file load and store
    import time
        
    # test data
    data = np.zeros((1000, 1000), dtype='float')
    
    # test gz
    tic = time.time()
    for i in xrange(it):
        # write
        gzStore(os.path.join(path, 'tst-%d' % i), data)
    tac = time.time()
    for i in xrange(it):
        # read
        _ = gzLoad(os.path.join(path, 'tst-%d' % i))
    toc = time.time()
    
    gzWops = it / (tac - tic)
    gzRops = it / (toc - tac)
    gzTops = 2 * it / (toc - tic)
    
    print "GZ - Write: %f, Read: %f, Total: %f" % (gzWops, gzRops, gzTops)
    
    # test h5
    h5path = os.path.join(path, 'h5bench.hdf5')
    tic = time.time()
    for i in xrange(it):
        # write
        h5Store(h5path, 'tst-%d' % i, data)
    tac = time.time()
    for i in xrange(it):
        # read
        _ = h5Load(h5path, 'tst-%d' % i)
    toc = time.time()
    
    h5Wops = it / (tac - tic)
    h5Rops = it / (toc - tac)
    h5Tops = 2 * it / (toc - tic)
    
    print "H5 - Write: %f, Read: %f, Total: %f" % (h5Wops, h5Rops, h5Tops)


def deleteFile(filePath):
    # delete a file if it exists
    
    try:
        os.remove(filePath)
    except OSError, e:
        if e.errno != errno.ENOENT:
            # errno.ENOENT => no such file or directory
            raise


def listFiles(path, spec='*'):
    # list files (with glob)
    
    return glob.iglob(os.path.join(path, spec))


def export2csv(fname, data):
    with open(fname, 'wb') as fid:
        writer = csv.writer(fid, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for item in data:
            writer.writerow(item[:])


def taskify(data):
    taskid = 0
    tasks = {}
    for recid in data.keys():
        if recid == 'info': continue
        nb = len(data[recid])
        tasks[recid] = range(taskid, taskid + nb)
        taskid += nb
    return tasks


def copyExpResults(basePath, expID, dstPath, images=False):
    # copy the results of a biometrics experiment
    
    # source folder
    srcPath = os.path.join(basePath, 'Exp-%s' % str(expID))
    
    # create destination folder
    config.folder = os.path.join(dstPath, 'Exp-%s' % str(expID))
    config.deploy()
    
    # copy configs
    print "Copying config files..."
    confs = ['config.dict', 'harmonizedSubjects.dict',
             'populationStatistics.dict', 'subjects.dict',
             'testTasks.dict', 'trainTasks.dict']
    for f in confs:
        src = os.path.join(srcPath, f)
        dst = config.getFilePath((f, ))
        try:
            shutil.copy(src, dst)
        except IOError:
            print "Copy of file %s failed." % f
    
    # copy logs
    print "Copying log files..."
    for f in glob.glob(os.path.join(srcPath, 'log', '*.log')):
        fname = os.path.split(f)[1]
        dst = config.getFilePath(('log', fname))
        try:
            shutil.copy(f, dst)
        except IOError:
            print "Copy of file %s failed." % fname
    
    # copy results
    print "Copying results files..."
    ft = lambda item: item != ''
    for dirpath, _, filenames in os.walk(os.path.join(srcPath, 'results')):
        for f in filenames:
            if '.png' in f and not images:
                continue
            src = os.path.join(dirpath, f)
            cm = os.path.commonprefix([srcPath, src])
            aux = src.replace(cm, '').split(os.path.sep)
            aux = filter(ft, aux)
            dst = config.getFilePath(aux)
            try:
                shutil.copy(src, dst)
            except IOError:
                print "Copy of file %s failed." % f
    
    return None



if __name__ == '__main__':
    import pylab as pl
    print "Data Manager Module"
    
    config = {'source': 'StorageBIT',
              'DBConnection': {'dbName': 'storageTest',
                               'host': '193.136.222.234',
                               'port': 27017,
                               'dstPath': 'D:/BioMESH'},
              'experiments': ['T1', 'T2', 'T3'],
              'mapper': {'var1': 'signals/random/raw',
                         'var2': 'signals/random/a',
                         'var3': 'events/random/annotation',
                         'var4': 'signals/random/b'}
              }
    
#    config = {'source': 'HDF5',
#              'path': 'C:\\Users\\Carlos\\tmp\\datamanager',
#              'experiments': 'Test',
#              'mapper': {'var1': 'signals/test',
#                         'var2': 'signals/test/a',
#                         'var3': 'events/test/annotation',
#                         'var4': 'signals/test/c'}
#              }
    
    st = Store(config)
    
    # dbmetada
    out = st.dbmetada()
    out = st.dbmetada(refine={'_id': [0]})
    
    print out
    
    # db2data
    out = st.db2data('var1')
    out = st.db2data('var1', refine={'_id': [0]})
    for item in out[0]:
        signal = item['signal']
        pl.plot(signal, label='signal')
       
    out = st.db2data('var3', refine={'_id': [0]})
    for item in out[0]:
        pl.plot(item['timeStamps'], item['values'], 'ro', label='annotation')
    pl.legend()
    
    # data2db
    signal = np.ones(len(signal))
    mdata = {'comments': 'ones'}
    data = {0: {'var4': signal}, 'info': mdata}
    st.data2db(data, ['var4'])
    
    # subjectTTSets
    out = st.subjectTTSets({'train_set': ['A'],
                            'test_set': ['B'],
                            'train_time': (0, 0.1),
                            'test_time': (0, 0.1)})
    
    print out
    
    # getSubjectInfo
    out = st.getSubjectInfo()
    out = st.getSubjectInfo(refine={'_id': [1]})
    
    print out
    
    # add record
    record = {'experiment': 'T1', 'subject': 1, 'comments': 'new record', 'tags': ['A']}
    st.addRecord(record)
    
    st.close()
    
    pl.show()
    
