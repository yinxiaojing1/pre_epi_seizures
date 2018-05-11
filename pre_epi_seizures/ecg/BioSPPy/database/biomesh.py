"""
.. module:: biomesh
   :platform: Unix, Windows
   :synopsis: This module provides an API to use the BioMESH specification. See the specification at http://camoes.lx.it.pt/MediaWiki/index.php/Database_Specification#Data_Model.

.. moduleauthor:: Carlos Carreiras
"""


import pymongo as pmg
import h5db as h5
import syncdb as sdb
import os, string, warnings
import re, sympy
import numpy as np
# import glob

from multiprocessing import Process, Queue
#from Queue import Empty

# splitting regular expression for symbolic tags
RETAG = re.compile(r"\(|\)|~|\||&")



class biomesh():
    
    """
    
    Class to operate on a BioMESH DB database.
    
    Kwargs:
        
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1]
        
    """
    
    def __init__(self, dbName=None, host='localhost', port=27017, dstPath='~', srvPath='/BioMESH', sync=False, altSync=False):
        """
        Establish a connection to a BioMESH DB server. If the database (DB) does not exist, one is created with the necessary basic structures.
        
        Kwargs:
            dbName (str): Name of the DB to connect to.
            
            host (str): Network address of the MongoDB server. Default: 'localhost'.
            
            port (int): Port the MongoDB server is listening on. Default: 27017.
            
            dstPath (str): Path to store the HDF5 files. Default: '~'.
            
            srvPath (str): Path to store the HDF5 files on the remote server. Default: '/BioMESH'.
            
            sync (bool): Flag to perform synchronization with remote server. Default: True.
            
            altSync (bool): New experimental synchronization framework. Default: False.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            If the 'sync' flag is True, the connection to the database may take longer to establish due to the synchronization step. This is especially true if it is the first time the database is being accessed.
        
        Example:
            db = biomesh(dbName='biomesh_tst', host='193.136.222.234', port=27017, dstPath='~/tmp/biomesh', srvPath='/biomesh_tst', sync=False)
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if dbName is None:
            raise TypeError, "A DB name must be specified."
        
        # establish connection
        connection = pmg.Connection(host, port)
        db = connection[dbName]
        
        # path to write HDF5 files
        dstPath = _hdf5Folder(dbName, dstPath)
        
        # self things
        self.dbName = dbName
        self.db = db
        self.dstPath = dstPath
        self.srvPath = srvPath
        self.sync = sync
        self.altSync = altSync
        
        # perform sync with remote server
        if sync:
            self.syncNow()
        
        # experimental sync
        if altSync:
            # launch the sync client
            self.queue = Queue()
            self.process = Process(target=sdb.altSyncNow, args=(self.queue, 'admin', '193.136.222.234', 27018, dbName))
            self.process.start()
            
        
        # meta HDF5
        meta = os.path.join(dstPath, 'ExpSub.hdf5')
        if not os.path.exists(meta):
            fid = h5.meta(meta)
            fid.setDB(dbName)
            fid.close()
        
        # check basic structures
        self.experiments = experiments(db, meta, self)
        self.subjects = subjects(db, meta, self)
        self.records = records(db, dstPath, srvPath, meta, self)
        self.idTracker = db['IDTracker']
        
        
        if db.collection_names() == []:
            # DB is empty, create ID trackers, unknown experiment and subject
            self.idTracker.insert([{'_id': 'subjects', 'nextID': 0, 'recycle': [], 'rlen': 0},
                                   {'_id': 'experiments', 'nextID': 0, 'recycle': [], 'rlen': 0},
                                   {'_id': 'records', 'nextID': 0, 'recycle': [], 'rlen': 0}
                                   ])
            self.experiments.add({'name': 'unknown', 'description': 'Generic experiment.', 'goals': 'To store records from unknown experiments.', 'records':[]})
            self.subjects.add({'name': 'unnamed', 'records': []})
        
        # ensure indexes
        db.experiments.ensure_index('name')
        db.subjects.ensure_index('name')
        db.records.ensure_index('date')
        db.records.ensure_index('experiment')
        db.records.ensure_index('subject')
    
    def _getNewId(self, collection):
        """
        
        Generate a new ID for the given collection.
        
        Allows re-use of previously deleted IDs.
        
        Kwargs:
            collection (str): Name of the collection.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            ID = db._getNewId('records')
        
        References:
            .. [1]
            
        """
        
        if collection not in ['subjects', 'experiments', 'records']:
            return None
        
        ID = None
        
        # look for recycled ID
        doc = self.idTracker.find_and_modify(query={'_id': collection, 'rlen': {'$gt': 0}},
                                             update={'$inc': {'rlen': -1}, '$pop': {'recycle': -1}})
        if doc is None:
            # recycle is empty
            ID = self.idTracker.find_and_modify(query={'_id': collection},
                                                update={'$inc': {'nextID': 1}})['nextID']
        else:
            # get ID from recycle
            ID = doc['recycle'][0]
        
        return ID
    
    def _deleteId(self, collection, docId):
        """
    
        Close the connection to the database.
        
        Allows re-use of previously deleted IDs.
        
        Kwargs:
            collection (str): Name of the collection.
            
            docId (int): ID to delete.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            db._deleteId('records', 0)
        
        References:
            .. [1]
            
        """
        
        if collection in ['subjects', 'experiments', 'records']:
            # add deleted ID
            self.idTracker.update({'_id': collection},
                                  {'$addToSet': {'recycle': docId},
                                   '$inc': {'rlen': 1}})
    
    def close(self):
        """
    
        Close the connection to the database.
        
        Kwargs:
            
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            db.close()
        
        References:
            .. [1]
            
        """
        
        # perform sync with remote server
        if self.sync:
            self.syncNow()
        
        if self.altSync:
            self.queue.put({'function': 'SIGTERM'})
            self.process.join()
            self.process.terminate()
        
        # close connection
        self.db.connection.close()
    
    
    def syncNow(self):
        """
        
        Function to perform the synchronization with the remote server.
        
        Kwargs:
            None
        
        Kwrvals:
            None
        
        See Also:
            
        
        Notes:
            
        
        Example:
            db.sync()
        
        References:
            .. [1]
            
        """
        
        thread, queue = sdb.syncDB(self.dbName, self.db.connection.host, self.dstPath, self.srvPath)
        print "Synchronizing with remote server."
        thread.join(2)
        if thread.is_alive():
            print "This may take some time.\nPlease wait..."
            thread.join()
            out = queue.get()
            if not out['sync']:
                warnings.warn("Synchronization failed: check ssh configuration.\n" + out['stderr'])
            else:
                print "Done!"
        else:
            warnings.warn("Synchronization failed.")
    
    
    def drop(self, flag=True):
        """
        
        Function to remove the database from the server. The HDF5 files are preserved.
        
        Kwargs:
            flag (bool): Flag to override user confirmation about removal. Set to False to avoid user prompt. Default: True.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            db.drop()
        
        References:
            .. [1]
            
        """
        
        # make sure
        prompt = 'Are you sure you want to drop the databse "{0}" from the server at "{1}"? (y/n)\nRemoved data cannot be restored!\n'.format(self.dbName, self.db.connection.host)
        sure = True
        if flag:
            var = raw_input(prompt)
            if var != 'y' and var != 'Y':
                sure = False
        
        out = None
        if sure:
            # drop command
            out = self.db.command('dropDatabase')
        
        return out
    
    
    def h5Add(self, filePath=None):
        """
        
        Procedure to add already created HDF5 files (records) to MongoDB.
        
        Kwargs:
            filePath (str): Location of the file to add.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if filePath is None:
            raise TypeError, "A path to the HDF5 file must be provided."
        
        # helper
        def filldict(x, y): D[x] = y
        
        # open the file
        f = h5.hdf(filePath, mode='r')
        recId = int(os.path.split(filePath)[1].split('.')[0].split('_')[-1])
        
        # loop the data
        D = {}
        f.signals.visititems(filldict)
        for key in D.iterkeys():
            if isinstance(D[key], h5.h5py.Dataset):
                out = f.getSignalInfo(key)
                # update the DB
                self.records.addSignal(recId, out['mdata']['type'], [], out['mdata'], flag=False)
        
        # loop the events
        D = {}
        f.events.visititems(filldict)
        aux = []
        for val in D.itervalues():
            if isinstance(val, h5.h5py.Dataset):
                aux.append(val.parent.name)
        aux = set(aux)
        for item in aux:
            out = f.getEventInfo(item)
            # update the DB
            self.records.addEvent(recId, out['mdata']['type'], [], [], out['mdata'], flag=False)
        
        # close the file
        f.close()
    
    
    def metaH5Add(self, filePath=None):
        """
        
        Procedure to add an already created meta HDF5 file (experiments and subjects) to MongoDB.
        
        Kwargs:
            filePath (str): Location of the file to add.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if filePath is None:
            raise TypeError, "A path to the HDF5 file must be provided."
        
        # open the file
        f = h5.meta(filePath, mode='r')
        expList = f.listExperiments()['expList']
        subList = f.listSubjects()['subList']
        f.close()
        
        # experiments
        for item in expList:
            self.experiments.add(item, flag=False)
        
        # subjects
        for item in subList:
            self.subjects.add(item, flag=False)


    def subsInExp(self, experimentName):
        """
        
        List all the subjects in the database belonging to a given experiment.
        
        Kwargs:
            experimentName (str): The name of the experiment.
        
        Kwrvals:
            idList (list): List with the IDs of the subjects.
        
        See Also:
            
        
        Notes:
            
        
        Example:
            subList = db.subsInExp('unknown')['idList']
        
        References:
            .. [1]
            
        """
        
        exp = self.experiments.getByName(experimentName, {'records': 1})['doc']
        try:
            out = [item['subject'] for item in exp['records']]
        except KeyError:
            out = []
        
        return {'idList': out}
    
    
    def expsInSub(self, subjectId):
        """
        
        List all the experiments in the database belonging to a given subject.
        
        Kwargs:
            subjectId (int): The ID of the subject.
        
        Kwrvals:
            nameList (list): List with the names of the experiments in the database.
        
        See Also:
            
        
        Notes:
            
        
        Example:
            expList = db.expsInSub(0)['nameList']
        
        References:
            .. [1]
            
        """
        sub = self.subjects.getById(subjectId, {'records': 1})['doc']
        try:
            out = [item['experiment'] for item in sub['records']]
        except KeyError:
            out = []
        
        return {'nameList': out}


def listDB(host='localhost', port=27017):
    """
    
    Lists the databases present in a given server.
    
    Kwargs:
        host (str): Network address of the MongoDB server. Default: 'localhost'.
        
        port (int): Port the MongoDB server is listening on. Default: 27017.
    
    Kwrvals:
        dbList (list): A list with DB names.
    
    See Also:
        
    
    Notes:
        
    
    Example:
        listDB('193.136.222.234', 27017)
    
    References:
        .. [1]
        
    """
    
    # connect to DB
    connection = pmg.Connection(host, port)
    
    # list the DBs
    dbList = connection.database_names()
    
    # remove system DBs from list
    protect = ['admin', 'config', 'local']
    for item in protect:
        try:
            dbList.remove(item)
        except ValueError:
            continue
    
    # close connection
    connection.close()
    
    # kwrvals
    kwrvals = {}
    kwrvals['dbList'] = dbList
    
    return kwrvals


def _hdf5Folder(dbName=None, dstPath=None):
    """
    
    Check if the path for the HDF5 files exists, create if needed.
    
    Kwargs:
        dbName (str): Name of the DB.
        
        dstPath (str): Path to store the HDF5 files.
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1]
        
    """
    
    # check inputs
    if dbName is None:
        raise TypeError, "A DB name must be provided."
    if dstPath is None:
        raise TypeError, "A path must be provided."
    
    # expand the user or properly format the path
    if '~' in dstPath:
        dstPath = os.path.abspath(os.path.expanduser(dstPath))
    else:
        dstPath = os.path.abspath(dstPath)
    
    # add the DB name
    dstPath = os.path.join(dstPath, 'Databases', dbName)
    
    # make sure the path exists
    if not os.path.exists(dstPath):
        os.makedirs(dstPath)
    
    return dstPath



class subjects:
    """
    
    Class to operate on the Subjects collection.
    
    Kwargs:
        
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1]
        
    """
    
    def __init__(self, db=None, meta=None, parent=None, **kwargs):
        """
        
        Initialize the subjects class.
        
        Kwargs:
            db (pymongo.database.Database): An instance representing the DB.
            
            meta (str): The path to the meta HDF5 file.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            For subjects, the key '_id' is the preferred indexer.
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if db is None:
            raise TypeError, "A DB instance must be provided."
        if meta is None:
            raise TypeError, "A path to the meta HDF5 file must be provided."
        
        self.collection = db['subjects']
        self.meta = meta
        self.parent = parent
    
    
    def add(self, subject=None, flag=True):
        """
        
        To add a subject to the 'subjects' collection.
        
        Kwargs:
            subject (dict): Subject (JSON) to add.
            
            flag (bool): Flag to store in HDF5. Default: True.
        
        Kwrvals:
            subjectId (int): ID of the subject in the DB.
        
        See Also:
            
        
        Notes:
            If the subject already exists (i.e. there is a subject with the same name) the ID of the subject in the DB is returned.
        
        Example:
            subId = db.subjects.add({'name': 'subject'})['subjectId']
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if subject is None:
            raise TypeError, "A JSON subject must be provided."
        
        # check if subject already exists (by name)
        res = self.getByName(subject['name'], restrict={'_id': 1})
        
        # kwrvals
        kwrvals = {}
        
        if not res['doc'] is None:
            # the subject already exists
            print 'Warning: subject already exists in DB; skipping new insertion.'
            kwrvals['subjectId'] = res['doc']['_id']
            return kwrvals
        
        # get a new id (starts at zero)
        # newId = self.collection.count()
        newId = self.parent._getNewId('subjects')
        
        # add id to subject document
        subject.update({'_id': newId})
        
        if not subject.has_key('records'):
            # add the "records" field
            subject.update({'records': []})
        
        # save to meta file
        if flag:
            fid = h5.meta(self.meta)
            fid.addSubject(subject)
            subject = fid.getSubject(subject['_id'])['subject']
            fid.close()
        
        # save to db
        try:
            self.collection.insert(subject)
        except Exception, _:
            self.parent._deleteId('subjects', newId)
            raise
        
        # experimental sync
        if self.parent.altSync:
            self.parent.queue.put({'function': 'subjects.add'})
        
        kwrvals['subjectId'] = subject['_id']
        return kwrvals
        
    
    def _addExperiment(self, subjectId, experimentName, flag=True):
        """
        
        To add an experiment to the given subject.
        
        Kwargs:
            subjectId (int): ID of the subject.
            
            experimentName (str): Name of the experiment.
            
            flag (bool): Flag to store in HDF5. Default: True.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        if not self._hasExperiment(subjectId, experimentName):
            self.collection.update({'_id':subjectId}, {'$push': {'records': {'experiment':experimentName, 'list':[]}}})
            if flag:
                fid = h5.meta(self.meta)
                info = fid.getSubject(subjectId)['subject']
                info['records'].append({'experiment':experimentName, 'list':[]})
                fid.updateSubject(subjectId, info)
                fid.close()
        
        
    def _hasExperiment(self, subjectId, experimentName):
        """
        
        To check if a given experiment is already included in a given subject.
        
        Kwargs:
            subjectId (int): ID of the subject.
            
            experimentName (str): Name of the experiment.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        res = self.getById(subjectId, restrict={'records.experiment':1})
        
        if res['doc'] is None:
            # subject does not exist
            #### maybe print a warning???
            return False
        
        try:
            doc = res['doc']['records']
        except KeyError:
            ### maybe print a warning
            self.collection.update({'_id': res['doc']['_id']}, {'$set': {'records':[]}})
            return False
        
        for item in doc:
            if item['experiment'] == experimentName:
                return True
        
        return False
    
    def _addRecord(self, subjectId, experimentName, recordId, flag=True):
        """
        
        To add a record link to a given subject.
        
        Kwargs:
            subjectId (int): ID of the subject.
            
            experimentName (str): Name of the experiment.
            
            recordId (int): ID of the record.
            
            flag (bool): Flag to store in HDF5. Default: True.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        self._addExperiment(subjectId, experimentName)
        
        if not self._hasRecord(subjectId, experimentName, recordId):
            self.collection.update({'_id':subjectId, 'records.experiment':experimentName}, {'$push':{'records.$.list':recordId}})
            if flag:
                fid = h5.meta(self.meta)
                info = fid.getSubject(subjectId)['subject']
                for item in info['records']:
                    if item['experiment'] == experimentName:
                        break
                item['list'].append(recordId)
                fid.updateSubject(subjectId, info)
                fid.close()
        
    
    def _hasRecord(self, subjectId, experimentName, recordId):
        """
        
        Check if subject has a given record associated with a certain experiment.
        
        Kwargs:
            subjectId (int): ID of the subject.
            
            experimentName (str): Name of the experiment.
            
            recordId (int): ID of the record.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        # get the subject
        res = self.getById(subjectId, restrict={'records':1})
        
        if res['doc'] is None:
            # subject does not exist
            #### maybe print a warning???
            return False
        
        try:
            doc = res['doc']['records']
        except KeyError:
            # there is no 'records' field => adding one
            ### maybe print a warning???
            ### but now 'records' is always added...
            self.collection.update({'_id': res['doc']['_id']}, {'$set': {'records':[]}})
            return False
        
        # check under the given experiment
        aux = None
        for item in doc:
            if item['experiment'] == experimentName:
                aux = item['list']
                break
        
        if aux == None:
            raise ValueError("The experiment is not in subject '" + str(experimentName) + "'.")
        
        return (recordId in aux)
        
    
    def getByName(self, subjectName=None, restrict={}):
        """
        
        Query the 'subjects' collection by name.
        
        Kwargs:
            subjectName (str): Name of the subject to query.
            
            restrict (dict): Dictionary to restrict the information sent by the DB. Default: {}.
        
        Kwrvals:
            doc (dict): The document with the results of the query. None if no match is found.
        
        See Also:
            
        
        Notes:
            
        
        Example:
            # get everything
            doc = db.subjects.getByName('subject')['doc']
            # only return the ID
            doc = db.subjects.getByName('subject', restrict={'_id': 1})['doc']
            # only return the name (ID is also returned!)
            doc = db.subjects.getByName('subject', restrict={'name': 1})['doc']
            # only return the name and don't return the ID
            doc = db.subjects.getByName('subject', restrict={'name': 1, '_id': 0})['doc']
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if subjectName is None:
            raise TypeError, "A subject name must be provided."
        
        # avoid non-ascii characters
        subjectName = h5.latin1ToAscii(subjectName)
        
        search = {'name':subjectName}
        
        if restrict == {}:
            doc = self.collection.find_one(search)
        else:
            doc = self.collection.find_one(search, restrict)
        
        # kwrvals
        kwrvals = {}
        kwrvals['doc'] = doc
        
        return kwrvals
        
    def getById(self, subjectId=None, restrict={}):
        """
        
        Query the 'subjects' collection by ID.
        
        Kwargs:
            subjectId (int): ID of the subject to query.
            
            restrict (dict): Dictionary to restrict the information sent by the DB. Default: {}.
        
        Kwrvals:
            doc (dict): The document with the results of the query. None if no match is found.
        
        See Also:
            
        
        Notes:
            
        
        Example:
            # get everything
            doc = db.subjects.getById(0)['doc']
            # only return the ID
            doc = db.subjects.getById(0, restrict={'_id': 1})['doc']
            # only return the name (ID is also returned!)
            doc = db.subjects.getById(0, restrict={'name': 1})['doc']
            # only return the name and don't return the ID
            doc = db.subjects.getById(0, restrict={'name': 1, '_id': 0})['doc']
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if subjectId is None:
            raise TypeError, "A subject ID must be provided."
        
        search = {'_id':subjectId}
        
        if restrict == {}:
            doc = self.collection.find_one(search)
        else:
            doc = self.collection.find_one(search, restrict)
        
        # kwrvals
        kwrvals = {}
        kwrvals['doc'] = doc
        
        return kwrvals
        
    
    def get(self, refine={}, restrict={}):
        """
        
        Make a general query to the 'subjects' collection.
        
        Kwargs:
            refine (dict): Dictionary to refine the search. Default: {}.
            
            restrict (dict): Dictionary to restrict the information sent by the DB. Default: {}.
        
        Kwrvals:
            docList (list): The list of dictionaries with the results of the query. Empty list if no match is found.
        
        See Also:
            
        
        Notes:
            
        
        Example:
            # get all subjects
            res = db.subjects.get()['docList']
            # get subjects with 'field' set to 'new'
            res = db.subjects.get(refine={'field': 'new'})['docList']
            # get subjects with 'field' set to 'new' and 'flag' set to False
            res = db.subjects.get(refine={'field': 'new', 'flag': False})['docList']
        
        References:
            .. [1]
            
        """
        
        # avoid latin1 characters in name
        try:
            name = refine['name']
        except KeyError:
            pass
        else:
            name = h5.latin1ToAscii(name)
            refine.update({'name': name})
        
        if restrict == {}:
            doc = self.collection.find(refine)
        else:
            doc = self.collection.find(refine, restrict)
        
        res = []
        for item in doc:
            res.append(item)
        
        # kwrvals
        kwrvals = {}
        kwrvals['docList'] = res
        
        return kwrvals
    
    
    def update(self, subjectId=None, info={}):
        """
        
        Update a subject with the given information. Fields can be added, and its type changed, but not deleted.
        
        Kwargs:
            subjectId (int): ID of the subject to update.
            
            info (dict): Dictionary with the information to update. Default: {}.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            db.subjects.update(0, {'new': 'field'})
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if subjectId is None:
            raise TypeError, "A subject ID must be provided."
        
        # don't update the following fields
        fields = ['_id', 'records']
        for item in fields:
            if info.has_key(item):
                info.pop(item)
        
        # avoid latin1 characters in name
        try:
            name = info['name']
        except KeyError:
            pass
        else:
            name = h5.latin1ToAscii(name)
            info.update({'name': name})
        
        # update the DB
        self.collection.update({'_id': subjectId}, {'$set': info})
        
        # update the meta file
        fid = h5.meta(self.meta)
        fid.updateSubject(subjectId, info)
        fid.close()
        
        # experimental sync
        if self.parent.altSync:
            self.parent.queue.put({'function': 'subjects.update'})
    
    
    def _delExperiment(self, subjectId, experimentName):
        """
        
        Remove an experiment from a subject.
        
        Kwargs:
            subjectId (int): ID of the subject.
            
            experimentName (str): Name of the experiment to remove.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        self.collection.update({'_id':subjectId}, {'$pull':{'records':{'experiment':experimentName}}})
    
    
    def _delRecord(self, subjectId, experimentName, recordId, flag=True):
        """
        
        Remove a record (belonging to a certain experiment) from a subject.
        
        Kwargs:
            subjectId (int): ID of the subject.
            
            experimentName (str): Name of the experiment the record belengs to.
            
            recordId (int): ID of the record to remove.
            
            flag (bool): Flag to store in HDF5. Default: True.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        # update databse
        self.collection.update({'_id': subjectId, 'records.experiment': experimentName}, {'$pull': {'records.$.list': recordId}})
        
        # update HDF5
        if flag:
            fid = h5.meta(self.meta)
            info = fid.getSubject(subjectId)['subject']
            for item in info['records']:
                if item['experiment'] == experimentName:
                    break
            item['list'].remove(recordId)
            fid.updateSubject(subjectId, info)
            fid.close()
    
    def list(self):
        """
        
        List all the subjects in the database.
        
        Kwargs:
            
        
        Kwrvals:
            idList (list): List with the IDs of the subjects in the database.
        
        See Also:
            
        
        Notes:
            
        
        Example:
            subList = db.subjects.list()['idList']
        
        References:
            .. [1]
            
        """
        
        aux = self.get(restrict={'_id': 1})['docList']
        
        res = [item['_id'] for item in aux]
        
        # kwargs
        kwargs = {}
        kwargs['idList'] = res
        
        return kwargs



class experiments:
    """
    
    Class to operate on the Experiments collection.
    
    Kwargs:
        
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1]
        
    """
    
    def __init__(self, db=None, meta=None, parent=None, **kwargs):
        """
        
        Initialize the experiments class.
        
        Kwargs:
            db (pymongo.database.Database): An instance representing the DB.
            
            meta (str): The path to the meta HDF5 file.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            For experiments, the key 'name' is the preferred indexer.
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if db is None:
            raise TypeError, "A DB instance must be provided."
        if meta is None:
            raise TypeError, "A path to the meta HDF5 file must be provided."
        
        self.collection = db['experiments']
        self.meta = meta
        self.parent = parent
        
    
    def add(self, experiment=None, flag=True):
        """
        
        To add an experiment to the DB's 'experiments' collection.
        
        Kwargs:
            experiment (dict): Experiment (JSON) to add.
            
            flag (bool): Flag to store in HDF5. Default: True.
        
        Kwrvals:
            experimentId (int): ID of the experiment in the DB.
        
        See Also:
            
        
        Notes:
            If the experiment already exists (i.e. there is an experiment with the same name) the ID of the experiment in the DB is returned.
        
        Example:
            expId = db.experiments.add({'name': 'experiment'})['experimentId']
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if experiment is None:
            raise TypeError, "A JSON experiment must be provided."
        
        # check if experiment already exists (by name)
        res = self.getByName(experiment['name'], restrict={'_id': 1})
        
        # kwrvals
        kwrvals = {}
        
        if not res['doc'] is None:
            # the experiment already exists
            print 'Warning: experiment already exists in DB; skipping new insertion.'
            kwrvals['experimentId'] = res['doc']['_id']
            return kwrvals
        
        # get a new id (starts at zero)
        # newId = self.collection.count()
        newId = self.parent._getNewId('experiments')
        
        # add id to subject document
        experiment.update({'_id': newId})
        
        if not experiment.has_key('records'):
            # add the "records" field
            experiment.update({'records': []})
        
        # save to meta file
        if flag:
            fid = h5.meta(self.meta)
            fid.addExperiment(experiment)
            fid.close()
        
        # save to db
        try:
            self.collection.insert(experiment)
        except Exception, _:
            self.parent._deleteId('experiments', newId)
            raise
        
        # experimental sync
        if self.parent.altSync:
            self.parent.queue.put({'function': 'experiments.add'})
        
        kwrvals['experimentId'] = experiment['_id']
        return kwrvals

        
    def _addSubject(self, experimentName, subjectId, flag=True):
        """
        
        To add a subject to the given experiment.
        
        Kwargs:
            experimentName (str): Name of the experiment.
            
            subjectId (int): ID of the subject.
            
            flag (bool): Flag to store in HDF5. Default: True.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        if not self._hasSubject(experimentName, subjectId):
            self.collection.update({'name':experimentName}, {'$push': {'records': {'subject':subjectId, 'list':[]}}})
            if flag:
                fid = h5.meta(self.meta)
                info = fid.getExperiment(experimentName)['experiment']
                info['records'].append({'subject': subjectId, 'list':[]})
                fid.updateExperiment(experimentName, info)
                fid.close()
        
    def _hasSubject(self, experimentName, subjectId):
        """
        
        To check if a given subject is already included in a given experiment.
        
        Kwargs:
            experimentName (str): Name of the experiment.
            
            subjectId (int): ID of the subject.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        res = self.getByName(experimentName, restrict={'records.subject':1})
        
        if res['doc'] is None:
            # experiment does not exist
            #### maybe print a warning???
            return False
        
        try:
            doc = res['doc']['records']
        except KeyError:
            ### maybe print a warning???
            self.collection.update({'_id': res['doc']['_id']}, {'$set': {'records':[]}})
            return False
        
        for item in doc:
            if item['subject'] == subjectId:
                return True
        
        return False
    
    
    def _addRecord(self, experimentName, subjectId, recordId, flag=True):
        """
        
        To add a record link to a given experiment.
        
        Kwargs:
            experimentName (str): Name of the experiment.
            
            subjectId (int): ID of the subject.
            
            recordId (int): ID of the record.
            
            flag (bool): Flag to store in HDF5. Default: True.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        self._addSubject(experimentName, subjectId)
        
        if not self._hasRecord(experimentName, subjectId, recordId):
            self.collection.update({'name':experimentName, 'records.subject':subjectId}, {'$push':{'records.$.list':recordId}})
            if flag:
                fid = h5.meta(self.meta)
                info = fid.getExperiment(experimentName)['experiment']
                for item in info['records']:
                    if item['subject'] == subjectId:
                        break
                item['list'].append(recordId)
                fid.updateExperiment(experimentName, info)
                fid.close()
    
    
    def _hasRecord(self, experimentName, subjectId, recordId):
        """
        
        Check if experiment has a given record associated with a certain subject.
        
        Kwargs:
            experimentName (str): Name of the experiment.
            
            subjectId (int): ID of the subject.
            
            recordId (int): ID of the record.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        # get the experiment
        res = self.getByName(experimentName, restrict={'records':1})
        
        if res['doc'] is None:
            # experiment does not exist
            #### maybe print a warning???
            return False
        
        try:
            doc = res['doc']['records']
        except KeyError:
            # there is no records field => adding one
            ### maybe print a warning???
            self.collection.update({'_id': res['doc']['_id']}, {'$set': {'records':[]}})
            return False
        
        aux = None
        for item in doc:
            if item['subject'] == subjectId:
                aux = item['list']
                break
        
        if aux == None:
            raise ValueError("The subject is not in experiment '" + str(experimentName) + "'.")
        
        return (recordId in aux)
        
    def getByName(self, experimentName=None, restrict={}):
        """
        
        Query the 'experiments' collection by name.
        
        Kwargs:
            experimentName (str): Name of the experiment to query.
            
            restrict (dict): Dictionary to restrict the information sent by the DB. Default: {}.
        
        Kwrvals:
            doc (dict): The document with the results of the query. None if no match in found.
        
        See Also:
            
        
        Notes:
            
        
        Example:
            # get everything
            doc = db.experiments.getByName('experiment')['doc']
            # only return the ID
            doc = db.experiments.getByName('experiment', restrict={'_id': 1})['doc']
            # only return the name (ID is also returned!)
            doc = db.experiments.getByName('experiment', restrict={'name': 1})['doc']
            # only return the name and don't return the ID
            doc = db.experiments.getByName('experiment', restrict={'name': 1, '_id': 0})['doc']
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if experimentName is None:
            raise TypeError, "An experiment name must be specified."
        
        # avoid non-ascii characters
        experimentName = h5.latin1ToAscii(experimentName)
        
        search = {'name':experimentName}
        
        if restrict == {}:
            doc = self.collection.find_one(search)
        else:
            doc = self.collection.find_one(search, restrict)
        
        # kwrvals
        kwrvals = {}
        kwrvals['doc'] = doc
        
        return kwrvals
        
    def getById(self, experimentId=None, restrict={}):
        """
        
        Query the 'experiments' collection by ID.
        
        Kwargs:
            experimentId (int): ID of the experiment to query.
            
            restrict (dict): Dictionary to restrict the information sent by the DB. Default: {}.
        
        Kwrvals:
            doc (dict): The document with the results of the query. None if no match is found.
        
        See Also:
            
        
        Notes:
            
        
        Example:
            # get everything
            doc = db.experiments.getById(0)['doc']
            # only return the ID
            doc = db.experiments.getById(0, restrict={'_id': 1})['doc']
            # only return the name (ID is also returned!)
            doc = db.experiments.getById(0, restrict={'name': 1})['doc']
            # only return the name and don't return the ID
            doc = db.experiments.getById(0, restrict={'name': 1, '_id': 0})['doc']
        
        References:
            .. [1]
            
        """    
        
        # check inputs
        if experimentId is None:
            raise TypeError, "An experiment ID must be specified."
        
        search = {'_id':experimentId}
        
        if restrict == {}:
            doc = self.collection.find_one(search)
        else:
            doc = self.collection.find_one(search, restrict)
        
        # kwrvals
        kwrvals = {}
        kwrvals['doc'] = doc
        
        return kwrvals
    
    
    def get(self, refine={}, restrict={}):
        """
        
        Make a general query the 'experiments' collection.
        
        Kwargs:
            refine (dict): Dictionary to refine the search. Default: {}.
            
            restrict (dict): Dictionary to restrict the information sent by the DB. Default: {}.
        
        Kwrvals:
            docList (list): The list of documents with the results of the query. Empty list if no match is found.
        
        See Also:
            
        
        Notes:
            
        
        Example:
            # get all experiments
            res = db.experiments.get()['docList']
            # get experiments with 'field' set to 'new'
            res = db.experiments.get(refine={'field': 'new'})['docList']
            # get experiments with 'field' set to 'new' and 'flag' set to False
            res = db.experiments.get(refine={'field': 'new', 'flag': False})['docList']
        
        References:
            .. [1]
            
        """    
        
        # avoid latin1 characters in name
        try:
            name = refine['name']
        except KeyError:
            pass
        else:
            name = h5.latin1ToAscii(name)
            refine.update({'name': name})
        
        if restrict == {}:
            doc = self.collection.find(refine)
        else:
            doc = self.collection.find(refine, restrict)
        
        res = []
        for item in doc:
            res.append(item)
        
        # kwrvals
        kwrvals = {}
        kwrvals['docList'] = res
        
        return kwrvals
    
    
    def update(self, experimentName=None, info={}):
        """
        
        Update an experiment with the given information. Fields can be added, and its type changed, but not deleted.
        
        Kwargs:
            experimentName (str): Name of the experiment to update.
            
            info (dict): Dictionary with the information to update. Default: {}.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            db.experiments.update('experiment', {'new': 'field'})
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if experimentName is None:
            raise TypeError, "An experiment name must be provided."
        
        # don't update the following fields
        fields = ['_id', 'records']
        for item in fields:
            if info.has_key(item):
                info.pop(item)
        
        
        # update the DB
        self.collection.update({'name': experimentName}, {'$set': info})
        
        # update the meta file
        fid = h5.meta(self.meta)
        fid.updateExperiment(experimentName, info)
        fid.close()
        
        # experimental sync
        if self.parent.altSync:
            self.parent.queue.put({'function': 'experiments.update'})
    
    
    def _delSubject(self, experimentId, subjectId):
        """
        
        Remove a subject from an experiment.
        
        Kwargs:
            experimentId (int): ID of the experiment.
            
            subjectId (int): ID of the subject.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        self.collection.update({'_id':experimentId}, {'$pull':{'records':{'subject': subjectId}}})
    
    
    def _delRecord(self, experimentName, subjectId, recordId, flag=True):
        """
        
        Remove a record (belonging to a certain subject) from an experiment.
        
        Kwargs:
            experimentName (str): Name of the experiment.
            
            subjectId (int): ID of the subject.
            
            recordId (int): ID of the record to remove.
            
            flag (bool): Flag to store in HDF5. Default: True.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        # update database
        self.collection.update({'name': experimentName, 'records.subject': subjectId}, {'$pull': {'records.$.list': recordId}})
        
        # update HDF5
        if flag:
            fid = h5.meta(self.meta)
            info = fid.getExperiment(experimentName)['experiment']
            for item in info['records']:
                if item['subject'] == subjectId:
                    break
            item['list'].remove(recordId)
            fid.updateExperiment(experimentName, info)
            fid.close()
    
    def list(self):
        """
        
        List all the experiments in the database.
        
        Kwargs:
            
        
        Kwrvals:
            nameList (list): List with the names of the experiments in the database.
        
        See Also:
            
        
        Notes:
            
        
        Example:
            expList = db.experiments.list()['nameList']
        
        References:
            .. [1]
            
        """
        
        aux = self.get(restrict={'name': 1})['docList']
        
        res = [item['name'] for item in aux]
        
        # kwargs
        kwargs = {}
        kwargs['nameList'] = res
        
        return kwargs
    


class records:
    """
    
    Class to operate on the Records collection.
    
    Kwargs:
        
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1]
        
    """
    
    def __init__(self, db=None, path=None, srvPath=None, meta=None, parent=None, **kwargs):
        """
        
        Initialize the records class.
        
        Kwargs:
            db (pymongo.database.Database): An instance representing the DB.
            
            path (str): The path to store the HDF5 files.
            
            srvPath (str): The path to store the HDF5 files on the remote server.
            
            meta (str): The path to the meta HDF5 file.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            For records, the key '_id' is the preferred indexer.
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if db is None:
            raise TypeError, "A DB instance must be provided."
        if path is None:
            raise TypeError, "A path to the HDF5 files must be privided."
        if srvPath is None:
            raise TypeError, "A path to the HDF5 files on the remote server must be privided."
        
        self.dbName = db.name
        self.host = db.connection.host
        self.srvPath = srvPath
        self.collection = db['records']
        self.experiments = experiments(db, meta)
        self.subjects = subjects(db, meta)
        self.name = 'rec_%d.hdf5'
        self.path = path
        self.parent = parent
    
    def __getitem__(self, key):
        """
        
        To get Records from the DB.
        
        x.__getitem__(y) <==> x[y]
        
        Kwargs:
            key (int, slice, list): Items to retrieve.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            container = db.records[0]
            containerList = db.records[:]
            containerList = db.records[[0, 3, 4]]
        
        References:
            .. [1]
            
        """
        
        if type(key) is int:
            res = self.getById(key, restrict={'_id': 1})
            if res['doc'] is None:
                raise KeyError, "No Record found with given ID (%d)." % key
            return dataSelector(self, key)
        elif type(key) is slice:
            contnr = []
            for item in self._recordsIter(key):
                contnr.append(item)
            return contnr
        elif type(key) is list:
            contnr = []
            for item in self._recordsIter(key):
                contnr.append(item)
            return contnr
        else:
            raise KeyError, "Unsupported key type."
    
    def __len__(self):
        """
        
        Returns the number of records on the DB.
        
        x.__len__() <==> len(x)
        
        Kwargs:
            
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            len(db.records)
        
        References:
            .. [1]
            
        """
        
        res = self.getAll()
        
        return len(res['idList'])
    
    def __iter__(self):
        """
        
        Iterator operator over the Records in the DB.
        
        Kwargs:
            
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            for item in db.records:
                signal = item['signals']['test'][0].signal
                metadataS = item['signals']['test'][0].metadata
                timeStamps = item['events']['test'][0].timeStamps
                values = item['events']['test'][0].values
                metadataE = item['events']['test'][0].metadata
        
        References:
            .. [1]
            
        """
        
        return self._recordsIter()
    
    def _recordsIter(self, sel=None):
        """
        
        Iterator over the Records in the DB.
        
        Kwargs:
            sel (int, slice, list): Selector.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        aux = np.array(self.getAll()['idList'])
        aux = aux[sel].flatten()
        for item in aux:
            yield self.__getitem__(int(item))

    def add(self, record=None, flag=True):
        """
        
        To add a record to the DB's 'records' collection.
        
        Kwargs:
            record (dict): Record (JSON) to add.
            
            flag (bool): Flag to store in HDF5. Default: True.
        
        Kwrvals:
            recordId (int): ID of the record in the DB.
        
        See Also:
            
        
        Notes:
            A new record is always added, regardless of the existence of records with the same name. If the record to add has no 'experiment' and/or 'subject' fields, it will be linked to a generic experiment ('unknown') and/or to a generic subject ('unnamed').
        
        Example:
            recId = db.records.add({'name': 'record'})['recordId']
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if record is None:
            raise TypeError, "A JSON record must be provided."
        
        # get experiment name/ID
        if not record.has_key('experiment'):
            print("Record not assigned to an experiment.\nStoring in 'unknown' experiments.")
            record.update({'experiment':'unknown'})
        experiment = record['experiment']
        try:
            if type(experiment) is int:
                experiment = self.experiments.getById(experiment, restrict={'name': 1})['doc']['name']
            else:
                experiment = self.experiments.getByName(experiment, restrict={'name': 1})['doc']['name']
        except KeyError:
            raise ValueError, "Experiment %s does not exist in the database.\nAdd it before the record." % str(experiment)
        record.update({'experiment': experiment})
        
        # get subject ID/name
        if not record.has_key('subject'):
            print("Record has no subject.\nStoring in 'unnamed' subject.")
            record.update({'subject': 'unnamed'})
        subject = record['subject']
        try:
            if type(subject) is not int:
                subject = self.subjects.getByName(subject, restrict={'_id': 1})['doc']['_id']
            else:
                subject = self.subjects.getById(subject, restrict={'_id': 1})['doc']['_id']
        except KeyError:
            raise ValueError, "Subject %s does not exist in the database.\nAdd it before the record." % str(subject)
        record.update({'subject': subject})
        
        # generate session number
        if not record.has_key('session'):
            session = self._counter(subject, experiment)
            record.update({'session':session})
        session = record['session']
        
        # generate ID (starts at zero)
        # newId = self.collection.count()
        newId = self.parent._getNewId('records')
        # add id to subject document
        record.update({'_id': newId})
        recordId = record['_id']
        
        # tags
        if not record.has_key('tags'):
            record.update({'tags': [experiment]})
        else:
            tags = set(record['tags'])
            tags.add(experiment)
            record['tags'] = list(tags)
        
        # duration
        if not record.has_key('duration'):
            record.update({'duration': 0., 'durationReference': None})
        
        if flag:
            # save to new HDF5
            record.update({'dbName': self.dbName}) # store DB name
            name = self.name % recordId
            filePath = os.path.join(self.path, name)
            f = h5.hdf(filePath, mode='w')
            f.addInfo(record)
            f.close()
            # sync file to server
            # thread = sdb.syncH5(self.dbName, name, self.host, self.path, self.srvPath)
            # thread.join(3)
        
        # other things for the DB
        # if not record.has_key('audit'):
            # # add the "audit" field
            # record.update({'audit':[]})
            
        if not record.has_key('signals'):
            # add the "signals" field
            record.update({'signals':{}})
            
        if not record.has_key('events'):
            # add the "events" field
            record.update({'events':{}})
        
        # save to db
        try:
            self.collection.insert(record)
        except Exception, _:
            self.parent._deleteId('records', newId)
            raise
        self.subjects._addRecord(subject, experiment, recordId)
        self.experiments._addRecord(experiment, subject, recordId)
        
        # experimental sync
        if self.parent.altSync:
            self.parent.queue.put({'function': 'records.add'})
        
        # kwrvals
        kwrvals = {}
        kwrvals['recordId'] = recordId
        
        return kwrvals
    
    def _counter(self, subjectId, experimentName, debug=False):
        """
        
        Count the number of sessions a subject atended a given experiment.
        
        Kwargs:
            subjectId (int): ID of the subject.
            
            experimentName (str): Name of the experiment.
            
            debug (bool): For debug. Default: False.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        # via experiments
        res = self.experiments.getByName(experimentName, restrict={'records':1})
        
        session = 0
        if res['doc'] is not None:
            doc = res['doc']['records']
            for item in doc:
                if item['subject'] == subjectId:
                    session = len(item['list'])
                    break
        
        if debug:
            # via subjects
            res = self.subjects.getById(subjectId, restrict={'records':1})
            
            sessionD = 0
            if res['doc'] is not None:
                doc = res['doc']['records']
                for item in doc:
                    if item['experiment'] == experimentName:
                        sessionD = len(item['list'])
                        break
            print 'Counter debug: everything OK? ', session == sessionD
        
        return session
    
    def addAudit(self):
        """
        
        To add information to 'audit' field.
        
        TO DO!
        
        Kwargs:
            
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        print "TO DO - Function to add information to 'audit' field."
    
    def addSignal(self, recordId=None, signalType='/', signal=None, mdata={}, flag=True, compress=False, updateDuration=False):
        """
        
        To add signal (synchronous) data to a record.
        
        Kwargs:
            recordId (int): ID of the record.
            
            signalType (str): Type of the signal to add. Default: '/'.
            
            signal (array): Signal to add. Default: [].
            
            mdata (dict): Dictionary (JSON) with metadata about the signal. Default: {}.
            
            flag (bool): Flag to store in HDF5. Default: True.
            
            compress (bool): Flag to compress the data (GZIP). Default: False.
            
            updateDuration (bool): Flag to update the duration of the record. Default: False.
        
        Kwrvals:
            recordId (int): ID of the record.
            
            signalRef (str): Storge name of the signal.
            
            signalType (str): Type of the inserted signal.
        
        See Also:
            
        
        Notes:
            
        
        Example:
            res = db.records.addSignal(0, '/test', [0, 1, 2, 3], {'comments': 'test signal'})
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if recordId is None:
            raise TypeError, "A record ID must be provided."
        
        # get record to add data to
        res = self.getById(recordId, restrict={'signals':1, 'duration': 1})
        
        if res['doc'] is None:
            raise ValueError("No record found with given ID ({0}).".format(recordId))
        
        # correctly format the type
        try:
            weg = signalType
            if weg is '':
                weg = '/'
            elif not weg[0] == '/':
                weg = '/' + weg
        except KeyError:
            weg = '/'
        mdata.update({'type': weg})
        
        # generate name
        try:
            signalName = mdata['name']
        except KeyError:
            signalName = None
        signalName = self._nameChecker('signals', res['doc'], weg, signalName)
        mdata.update({'name': signalName})
        
        # kwrvals
        kwrvals = {}
        kwrvals['recordId'] = recordId
        kwrvals['signalRef'] = signalName
        kwrvals['signalType'] = weg
        
        if flag:
            # add to HDF5 file
            name = self.name % recordId
            filePath = os.path.join(self.path, name)
            f = h5.hdf(filePath, mode='a')
            source = f.addSignal(weg, signal, mdata, signalName, compress)
            f.close()
        
        # add tags
        wegits = weg.split('/')
        self.addTags(recordId, wegits[1:], flag)
        
        # duration
        if updateDuration:
            try:
                Fs = float(mdata['sampleRate'])
            except KeyError:
                pass
            else:
                self.update(recordId, {'duration': len(signal) / Fs, 'durationReference': source}, flag=flag)
                    
        
        # update DB
        mdata.pop('type')
        if weg == '/':
            weg = 'signals' + '.loc'
        else:
            weg = 'signals' + weg.replace('/', '.') + '.loc'
        self.collection.update({'_id': recordId}, {'$push': {weg: mdata}})
        
        # experimental sync
        if self.parent.altSync:
            self.parent.queue.put({'function': 'records.addSignal'})
        
        return kwrvals
    
    def addEvent(self, recordId=None, eventType='/', timeStamps=None, values=None, mdata={}, flag=True, compress=False):
        """
        
        To add events (asynchronous data) to a record.
        
        Kwargs:
            recordId (int): ID of the record.
            
            eventType (str): Type of the events to add. Default: '/'.
            
            timeStamps (array): Array of time stamps. Default: [].
            
            values (array): Array with data for each time stamp. Default: [].
            
            mdata (dict): JSON with metadata about the events. Default: {}.
            
            flag (bool): Flag to store in HDF5. Default: True.
            
            compress (bool): Flag to compress the data (GZIP). Default: False.
        
        Kwrvals:
            recordId (int): ID of the record.
            
            eventRef (str): Storge name of the events.
            
            eventType (str): Type of the inserted events.
        
        See Also:
            
        
        Notes:
            
        
        Example:
            res = db.records.addEvent(0, '/test', [0, 1, 2], [[0, 1], [2, 3], [4, 5]], {'comments': 'test event'})
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if recordId is None:
            raise TypeError, "A record ID must be provided."
        
        # get record to add data to
        res = self.getById(recordId, restrict={'events':1})
        
        if res['doc'] is None:
            raise ValueError("No record found with given ID ({0}).".format(recordId))
        
        # correctly format the type
        try:
            weg = eventType
            if weg is '':
                weg = '/'
            elif not weg[0] == '/':
                weg = '/' + weg
        except KeyError:
            weg = '/'
        mdata.update({'type': weg})
        
        # generate name
        try:
            eventName = mdata['name']
        except KeyError:
            eventName = None
        eventName = self._nameChecker('events', res['doc'], weg, eventName)
        mdata.update({'name': eventName})
        
        # kwrvals
        kwrvals = {}
        kwrvals['recordId'] = recordId
        kwrvals['eventRef'] = eventName
        kwrvals['eventType'] = weg
        
        if flag:
            # add to HDF5 file
            name = self.name % recordId
            filePath = os.path.join(self.path, name)
            f = h5.hdf(filePath, mode='a')
            f.addEvent(weg, timeStamps, values, mdata, eventName, compress)
            f.close()
        
        # add tags
        wegits = weg.split('/')
        self.addTags(recordId, wegits[1:], flag)
        
        # update DB
        mdata.pop('type')
        if weg == '/':
            weg = 'events' + '.loc'
        else:
            weg = 'events' + weg.replace('/', '.') + '.loc'
        self.collection.update({'_id': recordId}, {'$push': {weg: mdata}})
        
        # experimental sync
        if self.parent.altSync:
            self.parent.queue.put({'function': 'records.addEvent'})
        
        return kwrvals
    
    def addTags(self, recordId=None, tags=None, flag=True):
        """
        
        Add tags to a record.
        
        Kwargs:
            recordId (int): ID of the record.
            
            tags (list): Tags to add.
            
            flag (bool): Flag to store in HDF5. Default: True.
        
        Kwrvals:
            None
        
        See Also:
            
        
        Notes:
            
        
        Example:
            db.records.addTags(0, ['a', 'b', 'c'])
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if recordId is None:
            raise TypeError, "A record ID must be provided."
        if tags is None:
            return
        
        # update the db
        doc = self.collection.find_and_modify(query={'_id': recordId},
                                              update={'$addToSet': {'tags': {'$each': tags}}},
                                              new=True)
        
        # update the HDF5
        if flag:
            name = self.name % recordId
            filePath = os.path.join(self.path, name)
            f = h5.hdf(filePath, mode='a')
            header = f.getInfo()['header']
            try:
                header['tags'] = doc['tags']
            except KeyError:
                pass
            f.addInfo(header)
            f.close()
        
        # experimental sync
        if self.parent.altSync:
            self.parent.queue.put({'function': 'records.addTags'})
    
    def getByName(self, recordName=None, restrict={}):
        """
        
        Query the 'records' collection by name.
        
        Kwargs:
            recordName (str): Name of the record.
            
            restrict (dict): Dictionary to restrict the information sent by the DB. Default: {}.
        
        Kwrvals:
            doc (dict): The document with the results of the query. None if no match in found.
        
        See Also:
            
        
        Notes:
            
        
        Example:
            # get everything
            doc = db.records.getByName('record')['doc']
            # only return the ID
            doc = db.records.getByName('record', restrict={'_id': 1})['doc']
            # only return the name (ID is also returned!)
            doc = db.records.getByName('record', restrict={'name': 1})['doc']
            # only return the name and don't return the ID
            doc = db.records.getByName('record', restrict={'name': 1, '_id': 0})['doc']
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if recordName is None:
            raise TypeError, "A record name must be specified."
        
        search = {'name':recordName}
        
        if restrict == {}:
            doc = self.collection.find_one(search)
        else:
            doc = self.collection.find_one(search, restrict)
        
        # kwrvals
        kwrvals = {}
        kwrvals['doc'] = doc
        
        return kwrvals
        
    def getById(self, recordId=None, restrict={}):
        """
        
        Query the 'records' collection by ID.
        
        Kwargs:
            recordId (int): ID of the record.
            
            restrict (dict): Dictionary to restrict the information sent by the DB. Default: {}.
        
        Kwrvals:
            doc (dict): The document with the results of the query. None if no match in found.
        
        See Also:
            
        
        Notes:
            
        
        Example:
            # get everything
            doc = db.records.getById(0)['doc']
            # only return the ID
            doc = db.records.getById(0, restrict={'_id': 1})['doc']
            # only return the name (ID is also returned!)
            doc = db.records.getById(0, restrict={'name': 1})['doc']
            # only return the name and don't return the ID
            doc = db.records.getById(0, restrict={'name': 1, '_id': 0})['doc']
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if recordId is None:
            raise TypeError, "A rocord ID must be provided."
        
        search = {'_id':recordId}
        
        if restrict == {}:
            doc = self.collection.find_one(search)
        else:
            doc = self.collection.find_one(search, restrict)
        
        # kwrvals
        kwrvals = {}
        kwrvals['doc'] = doc
        
        return kwrvals
    
    def get(self, refine={}, restrict={}):
        """
        
         Make a general query the 'records' collection.
        
        Kwargs:
            refine (dict): Dictionary to refine the search. Default: {}.
            
            restrict (dict): Dictionary to restrict the information sent by the DB. Default: {}.
        
        Kwrvals:
            docList (list): The list of documents with the results of the query. Empty list if no match is found.
        
        See Also:
            
        
        Notes:
            
        
        Example:
            # get all records
            res = db.records.get()['docList']
            # get records with 'field' set to 'new'
            res = db.records.get(refine={'field': 'new'})['docList']
            # get records with 'field' set to 'new' and 'flag' set to False
            res = db.records.get(refine={'field': 'new', 'flag': False})['docList']
        
        References:
            .. [1]
            
        """
        
        if restrict == {}:
            doc = self.collection.find(refine)
        else:
            doc = self.collection.find(refine, restrict)
        
        res = []
        for item in doc:
            res.append(item)
        
        # kwrvals
        kwrvals = {}
        kwrvals['docList'] = res
        
        return kwrvals
    
    def getSignal(self, recordId=None, signalType='/', signalRef=None):
        """
        
        Retrive a signal(synchronous data) from a record and corresponding metadata.
        
        Kwargs:
            recordId (int): ID of the record.
            
            signalType (str): Type of the desired signal. Default: '/'.
            
            signalRef (int, str): Storge name (global) or index (local) of the desired signal.
        
        Kwrvals:
            signal (array): Array with the signal.
            
            mdata (dict): Dictionary with the signal's accompanying metadata.
        
        See Also:
            
        
        Notes:
            
        
        Example:
            res = db.records.getSignal(0, '/test', 0)
            signal = res['signal']
            metadata = res['mdata']
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if recordId is None:
            raise TypeError, "A record ID must be provided."
        if signalRef is None:
            raise TypeError, "A signal reference must be provided."
        
        # correctly format the type
        weg = signalType
        if weg == '':
            weg = '/'
        elif not weg[0] == '/':
            weg = '/' + weg
        
        if isinstance(signalRef, basestring):
            # signalRef is a name
            dataName = signalRef
        elif type(signalRef) is int:
            # signalRef is an index
            if weg == '/':
                wegM = 'signals' + '.loc'
            else:
                wegM = 'signals' + weg.replace('/', '.') + '.loc'
            res = self.getById(recordId, restrict={wegM:1})
            aux = res['doc']
            wegits = wegM.split('.')
            for item in wegits: aux = aux[item]
            dataName = aux[signalRef]['name']
        else:
            raise TypeError, "Input argument 'signalRef' must be of type str or int."
        
        # access the HDF5
        name = self.name % recordId
        filePath = os.path.join(self.path, name)
        f = h5.hdf(filePath)
        kwrvals = f.getSignal(weg, dataName)
        f.close()
        
        return kwrvals
    
    def getEvent(self, recordId=None, eventType='/', eventRef=None):
        """
        
        Retrieve events (asynchronous data) from a record.
        
        Kwargs:
            recordId (int): ID of the record.
            
            eventType (str): Type of the desired event. Default: '/'.
            
            eventRef (str, int): Storage name (global) or index (local) of the desired events.
        
        Kwrvals:
            timeStamps (array): Array of time stamps.
            
            values (array): Array with data for each time stamp.
            
            mdata (dict): Dictionary with metadata about the events.
        
        See Also:
            
        
        Notes:
            
        
        Example:
            res = db.records.getEvent(0, '/test', 0)
            timeStamps = res['timeStamps']
            values = res['values']
            metadata = res['mdata']
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if recordId is None:
            raise TypeError, "A record ID must be provided."
        if eventRef is None:
            raise TypeError, "An event reference must be provided."
        
        # correctly format the type
        weg = eventType
        if weg == '':
            weg = '/'
        elif not weg[0] == '/':
            weg = '/' + weg
        
        if isinstance(eventRef, basestring):
            # eventRef is a name
            eventName = eventRef
        elif type(eventRef) is int:
            # eventRef is an index
            if weg == '/':
                wegM = 'events' + '.loc'
            else:
                wegM = 'events' + weg.replace('/', '.') + '.loc'
            res = self.getById(recordId, restrict={wegM:1})
            aux = res['doc']
            wegits = wegM.split('.')
            for item in wegits: aux = aux[item]
            eventName = aux[eventRef]['name']
        
        # access the HDF5
        name = self.name % recordId
        filePath = os.path.join(self.path, name)
        f = h5.hdf(filePath)
        kwrvals = f.getEvent(weg, eventName)
        f.close()
        
        return kwrvals
    
    def getAll(self, restrict={}, count=-1, randomFlag=False):
        """
        
        Generate a list of records present in the DB. The list may include all records, records pertaining to an experiment (or list of experiments), record pertaining to a subject (or list of subjects) or a combination of both experiments and subjects.
        
        Kwargs:
            restrict (dict): To restrict the results. Can have the following keys:
                experiment (int, str, list): Experiment ID, experiment name, list of experiment IDs, list of experiment names, or list of experiment names and IDs.
                subject (int, str, list): Subject ID, subject name, list of subject IDs, list of subject names, or list of experiment names and IDs.
                Default: {}
            
            count (int): The resulting list has, at most, 'count' items. Set to -1 to output all records found. Default: -1.
            
            randomFlag (bool): Set this flag to True to randomize the output list. Default: False.
        
        Kwrvals:
            idList (list): List with the records' IDs that match the search.
        
        See Also:
            
        
        Notes:
            
        
        Example:
            # get all records
            idList = db.records.getAll()['idList']
            # get at most 5 records
            idList = db.records.getAll(count=5)['idList']
            # get all records from experiment 'exp'
            idList = db.records.getAll(restrict={'experiment': 'exp'})['idList']
            # get all records from experiment 'exp' and 'dxp'
            idList = db.records.getAll(restrict={'experiment': ['exp', 'dxp']})['idList']
            # get all records from subject 0
            idList = db.records.getAll(restrict={'subject': 0})['idList']
            # get all records from subject 0 and 1
            idList = db.records.getAll(restrict={'subject': [0, 1]})['idList']
            # get all records from subject 0 and 1 and 'John Smith'
            idList = db.records.getAll(restrict={'subject': [0, 1, 'John Smith']})['idList']
            # get all records from experiment 'exp', but only from subjects 0 and 1
            idList = db.records.getAll(restrict={'experiment': 'exp', 'subject': [0, 1]})['idList']
        
        References:
            .. [1]
            
        """
        
        # helper functions
        def select(IDs, count, randomFlag=False):
            # to select up to count
            IDs.sort()
            count = int(count)
            if count >= 0:
                if count == 0:
                    IDs = []
                elif randomFlag:
                    # random selection
                    from random import sample
                    count = min([count, len(IDs)])
                    IDs = sample(IDs, count)
                    IDs.sort()
                else:
                    # sequential
                    IDs = IDs[0:count]
            return IDs
            
        def check(exp=[], sub=[]):
            # always put to list
            
            if isinstance(exp, basestring) or isinstance(exp, int):
                exp = [exp]
            if isinstance(sub, basestring) or isinstance(sub, int):
                sub = [sub]
                
            return exp, sub
        
        def worker1(self, tpe, thing):
            res = []
            for item in thing:
                if tpe == 'experiment':
                    if isinstance(item, basestring):
                        doc = self.experiments.getByName(item, restrict={'records': 1})['doc']
                    else:
                        doc = self.experiments.getById(item, restrict={'records': 1})['doc']
                    if doc == None:
                        continue
            
                elif tpe == 'subject':
                    if isinstance(item, basestring):
                        doc = self.subjects.getByName(item, restrict={'records': 1})['doc']
                    else:
                        doc = self.subjects.getById(item, restrict={'records': 1})['doc']
                    if doc == None:
                        continue
            
                sigs = doc['records']
                for jtem in sigs:
                    res.extend(iter(jtem['list']))
            return res
        
        def worker2(self, exp, sub):
            # really does the job
            expSigs = worker1(self, 'experiment', exp)
            subSigs = worker1(self, 'subject', sub)
            
            expSigs = set(expSigs)
            subSigs = set(subSigs)
            res = list(expSigs.intersection(subSigs))
            
            return res
            
        # kwrvals
        kwrvals = {}
        
        IDs = []
        if restrict == {}:
            # get all records (up to count) regardless of experiment or subject
            cursor = self.collection.find({}, {'_id':1})
            for item in cursor: IDs.append(item['_id'])
            IDs = select(IDs, count, randomFlag)
            kwrvals['idList'] = IDs
            return kwrvals
        
        flag1 = restrict.has_key('experiment')
        flag2 = restrict.has_key('subject')
        
        if flag1:
            experiment = restrict['experiment']
            if flag2:
                subject = restrict['subject']
                # both experiment and subject
                exp, sub = check(experiment, subject)
                IDs = worker2(self, exp, sub)
                IDs = select(IDs, count, randomFlag)
                kwrvals['idList'] = IDs
                return kwrvals
            
            # only experiment
            exp, sub = check(exp=experiment)
            IDs = worker1(self, 'experiment', exp)
            IDs = list(set(IDs))
            IDs = select(IDs, count, randomFlag)
            kwrvals['idList'] = IDs
            return kwrvals
        
        if flag2:
            subject = restrict['subject']
            # only subject
            exp, sub = check(sub=subject)
            IDs = worker1(self, 'subject', sub)
            IDs = list(set(IDs))
            IDs = select(IDs, count, randomFlag)
            kwrvals['idList'] = IDs
            return kwrvals
        
        # other cases
        kwrvals['idList'] = IDs
        return kwrvals
    
    def listTypes(self, recordId=None):
        """
        
        To list the types of signals and events.
        
        Kwargs:
            recordId (int): ID of the record.
        
        Kwrvals:
            signalTypes (list): List of data types.
            
            eventTypes (list): List of event types.
        
        See Also:
            
        
        Notes:
            
        
        Example:
            res = db.records.listTypes(0)
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if recordId is None:
            raise TypeError, "A recordId must be provided."
        
        # helper
        def walkStruct(st, new, ty='/'):
            for key in st.iterkeys():
                if key == 'loc':
                    new.extend([ty+item['name'] for item in st[key]])
                else:
                    walkStruct(st[key], new, ty+key+'/')
        
        # get the record
        res = self.getById(recordId, restrict={'signals': 1, 'events': 1})
        if res['doc'] is None:
            raise ValueError, "No record found with given ID ({0}).".format(recordId)
        data = res['doc']['signals']
        events = res['doc']['events']
        
        # walk the structure
        dataL = []
        walkStruct(data, dataL)
        dataL.sort()
        eventsL = []
        walkStruct(events, eventsL)
        eventsL.sort()
        
        # kwrvals
        kwrvals = {}
        kwrvals['signalTypes'] = dataL
        kwrvals['eventTypes'] = eventsL
        
        return kwrvals
    
    def listSymbolicTags(self, query=None):
        # lists the records that match the symbolic query
        
        # check inputs
        if query is None:
            raise TypeError, "A query must be provided."
        
        # split query with the reg ex
        bits = RETAG.split(query)
        
        # strip white space, remove empty strings
        symbols = []
        for item in bits:
            item = item.strip()
            
            if item != '':
                symbols.append(item)
        
        # unique items
        symbols = list(set(symbols))
        
        # replace symbols
        varD = {}
        for i in xrange(len(symbols)):
            s = 'v%d' % i
            query = query.replace(symbols[i], s)
            varD[s] = symbols[i]
        
        # sympify ans simplyfy query
        query = sympy.simplify(sympy.sympify(query))
        
        # execute query
        def helper(query):
            op = query.class_key()[2]
            
            if op == 'Not':
                tags = [varD[str(item)] for item in query.args]
                return self.listNotTags(tags)['idList']
            
            elif op == 'And':
                tags = []
                aux = []
                for a in query.args:
                    if a.class_key()[2] == 'Symbol':
                        tags.append(varD[str(a)])
                    else:
                        aux.append(set(helper(a)))
                res = set(self.listAndTags(tags)['idList'])
                return list(res.intersection(*aux))
                
            elif op == 'Or':
                tags = []
                aux = []
                for a in query.args:
                    if a.class_key()[2] == 'Symbol':
                        tags.append(varD[str(a)])
                    else:
                        aux.append(set(helper(a)))
                res = set(self.listOrTags(tags)['idList'])
                return list(res.union(*aux))
            
            elif op == 'Symbol':
                tags = [varD[str(query)]]
                return self.listAndTags(tags)['idList']
            
            return []
        
        idList = helper(query)
        idList.sort()
        
        return {'idList': idList}
    
    def listAndTags(self, tags=None):
        """
        
        Lists the records that simultaneously have all the given tags (AND operator).
        
        Kwargs:
            tags (list): Tags to match.
        
        Kwrvals:
            idList (list): List with the records' IDs that match the search.
        
        See Also:
            
        
        Notes:
            
        
        Example:
            res = db.records.listAndTags(['a', 'b', 'c'])['idList']
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if tags is None:
            return {'idList': []}
        
        res = self.get({'tags': {'$all': tags}}, {'_id': 1})['docList']
        idList = [item['_id'] for item in res]
        
        return {'idList': idList}
    
    def listOrTags(self, tags=None):
        """
        
        Lists the records that have, at least, one of the given tags (OR operator).
        
        Kwargs:
            tags (list): Tags to match.
        
        Kwrvals:
            idList (list): List with the records' IDs that match the search.
        
        See Also:
            
        
        Notes:
            
        
        Example:
            res = db.records.listOrTags(['a', 'b', 'c'])['idList']
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if tags is None:
            return {'idList': []}
        
        res = self.get({'tags': {'$in': tags}}, {'_id': 1})['docList']
        idList = [item['_id'] for item in res]
        
        return {'idList': idList}
    
    def listNotTags(self, tags=None):
        # not operator
        
        # check inputs
        if tags is None:
            return {'idList': []}
        
        res = self.get({'tags': {'$nin': tags}}, {'_id': 1})['docList']
        idList = [item['_id'] for item in res]
        
        return {'idList': idList}
    
    def update(self, recordId=None, info={}, flag=True):
        """
        
        Update a record with the given information. Fields can be added, and its type changed, but not deleted.
        
        Kwargs:
            recordId (int): ID of the record to update.
            
            info (dict): Dictionary with the information to update. Default: {}.
            
            flag (bool): Flag to store in HDF5. Default: True.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            db.records.update(0, {'new': 'field'})
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if recordId is None:
            raise TypeError, "A record ID must be provided."
        
        # don't update the following fields
        fields = ['_id', 'experiment', 'subject', 'signals', 'events', 'tags']
        for item in fields:
            if info.has_key(item):
                info.pop(item)
        
        # update DB
        self.collection.update({'_id': recordId}, {'$set': info})
        
        # update HDF5
        if flag:
            name = self.name % recordId
            filePath = os.path.join(self.path, name)
            f = h5.hdf(filePath, mode='a')
            aux = f.getInfo()
            aux['header'].update(info)
            f.addInfo(**aux)
            f.close()
        
        # experimental sync
        if self.parent.altSync:
            self.parent.queue.put({'function': 'records.update'})
    
    def _nameChecker(self, case=None, doc=None, path=None, name=None):
        """
        
        To check if name exists in the record, ot to generate a new name.
        
        Kwargs:
            case (str): Case ('signals' or 'events') to check.
            
            doc (dict): Document to check.
            
            path (str): Type path.
            
            name (str): Name to check.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if case is None:
            raise TypeError, "A case ('signals' or 'events') must be provided."
        if doc is None:
            raise TypeError, "A record doc must be provided."
        
        # navigate to type
        weg = case + path
        wegits = weg.split('/')
        aux = doc
        for item in wegits:
            try:
                aux = aux[item]
            except KeyError:
                aux = None
                break
        
        # get the forbidden names
        forbidden = []
        if aux is not None:
            try:
                aux = aux['loc']
            except KeyError:
                aux = None
            else:
                for item in aux:
                    forbidden.append(item['name'])
        
        # is it necessary to generate new name?
        flag = True
        if (name is not None) and (name not in forbidden):
            flag = False
        
        if flag:
            # generate new name
#            name_o = name
            i = len(forbidden)
            name = case[:-1] + str(i)
            while name in forbidden:
                i += 1
                name = case[:-1] + str(i)
            
#            if name_o is not None:
#                print "The given name is already in use. A new name was generated: %s" % name
        
        return name
    
    def delete(self, recordId=None, keepFile=True):
        """
        
        Remove a record from the database.
        
        Kwargs:
            recordId (int): ID of the record.
            
            keepFile (bool): Flag to keep local file. Default: True.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            db.records.delete(0)
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if recordId is None:
            raise TypeError, "A record ID must be provided."
        
        # get subject and experiment
        doc = self.getById(recordId, {'subject': 1, 'experiment': 1})['doc']
        subjectId = doc['subject']
        experimentName = doc['experiment']
        
        # remove record from subject and experiment
        self.subjects._delRecord(subjectId, experimentName, recordId)
        self.experiments._delRecord(experimentName, subjectId, recordId)
        
        # remove from database
        self.collection.remove({'_id': recordId})
        
        # recycle ID
        self.parent._deleteId('records', recordId)
        
        # delete HDF5
        if not keepFile:
            name = self.name % recordId
            filePath = os.path.join(self.path, name)
            os.remove(filePath)
    
    def delSignal(self, recordId=None, signalType='/', signalRef=None):
        """
        
        Remove a signal (synchronous data) from a record.
        
        Kwargs:
            recordId (int): ID of the record.
            
            signalType (str): Type of the desired data. Default: '/'.
            
            signalRef (str, int): Storge name (global) or index (local) of the desired signal.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            db.records.delSignal(0, '/test', 0)
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if recordId is None:
            raise TypeError, "A record ID must be provided."
        if signalRef is None:
            raise TypeError, "A signal reference must be provided."
        
        # correctly format the type
        weg = signalType
        if weg == '':
            weg = '/'
        elif not weg[0] == '/':
            weg = '/' + weg
        
        # type for DB
        if weg == '/':
            wegM = 'signals' + '.loc'
        else:
            wegM = 'signals' + weg.replace('/', '.') + '.loc'
        
        wegits = wegM.split('.')
        res = self.getById(recordId, restrict={wegM:1})
        loc = res['doc']
        for item in wegits: loc = loc[item]
        
        # get the name of the dataset
        if isinstance(signalRef, basestring):
            # signalRef is a name
            dataName = signalRef
        elif type(signalRef) is int:
            # signalRef is an index
            dataName = loc[signalRef]['name']
        else:
            raise TypeError, "Input argument 'signalRef' must be of type str or int."
        
        # access the HDF5
        name = self.name % recordId
        filePath = os.path.join(self.path, name)
        f = h5.hdf(filePath, mode='r+')
        f.delSignal(weg, dataName)
        f.close()
        
        # update the DB
        self.collection.update({'_id': recordId}, {'$pull': {wegM: {'name': dataName}}}, multi=True)
        
        # experimental sync
        if self.parent.altSync:
            self.parent.queue.put({'function': 'records.delSignal'})
    
    def delSignalType(self, recordId=None, signalType='/'):
        """
        
        Remove a signal type from a record (including all sub-types).
        
        Kwargs:
            recordId (int): ID of the record.
            
            signalType (str): Type of the desired data. Drfault: '/'
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            db.records.delSignalType(0, '/test')
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if recordId is None:
            raise TypeError, "A record ID must be provided."
        
        # helper
        def walkStruct(st):
            for key in st.iterkeys():
                if key == 'loc':
                    for item in st[key]: nameList.append(item['name'])
                else:
                    walkStruct(st[key])
        
        # correctly format the type
        weg = signalType
        if weg == '':
            weg = '/'
        elif not weg[0] == '/':
            weg = '/' + weg
        
        # type for DB
        if weg == '/':
            wegM = 'signals'
        else:
            wegM = 'signals' + weg.replace('/', '.')
        
        wegits = wegM.split('.')
        res = self.getById(recordId, restrict={wegM:1})
        aux = res['doc']
        for item in wegits: aux = aux[item]
        
        # get the names
        nameList = []
        walkStruct(aux)
        
        # access the HDF5
        name = self.name % recordId
        filePath = os.path.join(self.path, name)
        f = h5.hdf(filePath, mode='r+')
        f.delSignalType(weg)
        f.close()
        
        # update the DB
        if len(wegits) <= 1:
            op = '$set'
            val = {}
        else:
            op = '$unset'
            val = 1
        self.collection.update({'_id': recordId}, {op: {wegM: val}})
        
        # experimental sync
        if self.parent.altSync:
            self.parent.queue.put({'function': 'records.delSignalType'})
    
    def delEvent(self, recordId=None, eventType='/', eventRef=None):
        """
        
        Remove events from a record.
        
        Kwargs:
            recordId (int): ID of the record.
            
            eventType (str): Type of the desired event. Default: '/'.
            
            eventRef (str, int): Storage name (global) or index (local) of the desired events.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            db.records.delEvent(0, '/test', 0)
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if recordId is None:
            raise TypeError, "A record ID must be provided."
        if eventRef is None:
            raise TypeError, "An event reference must be provided."
        
        # correctly format the type
        weg = eventType
        if weg == '':
            weg = '/'
        elif not weg[0] == '/':
            weg = '/' + weg
        
        # type for DB
        if weg == '/':
            wegM = 'events' + '.loc'
        else:
            wegM = 'events' + weg.replace('/', '.') + '.loc'
        wegits = wegM.split('.')
        res = self.getById(recordId, restrict={wegM:1})
        loc = res['doc']
        for item in wegits: loc = loc[item]
        
        # get the name of the dataset
        if isinstance(eventRef, basestring):
            # eventRef is a name
            eventName = eventRef
        elif type(eventRef) is int:
            # eventRef is an index
            eventName = loc[eventRef]['name']
        else:
            raise TypeError, "Input argument 'eventRef' must be of type str or int."
        
        # access the HDF5
        name = self.name % recordId
        filePath = os.path.join(self.path, name)
        f = h5.hdf(filePath, mode='r+')
        f.delEvent(weg, eventName)
        f.close()
        
        # update the DB
        self.collection.update({'_id': recordId}, {'$pull': {wegM: {'name': eventName}}}, multi=True)
        
        # experimental sync
        if self.parent.altSync:
            self.parent.queue.put({'function': 'records.delEvent'})
    
    def delEventType(self, recordId=None, eventType='/'):
        """
        
        Remove an event type from a record (including all sub-types).
        
        Kwargs:
            recordId (int): ID of the record.
            
            eventType (str): Type of the desired event. Default: '/'.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            db.records.delEventType(0, '/test')
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if recordId is None:
            raise TypeError, "A record ID must be provided."
        
        # helper
        def walkStruct(st):
            for key in st.iterkeys():
                if key == 'loc':
                    for item in st[key]: nameList.append(item['name'])
                else:
                    walkStruct(st[key])
        
        # correctly format the type
        weg = eventType
        if weg == '':
            weg = '/'
        elif not weg[0] == '/':
            weg = '/' + weg
        
        # type for DB
        if weg == '/':
            wegM = 'events'
        else:
            wegM = 'events' + weg.replace('/', '.')
        
        wegits = wegM.split('.')
        res = self.getById(recordId, restrict={wegM:1})
        aux = res['doc']
        for item in wegits: aux = aux[item]
        
        # get the names
        nameList = []
        walkStruct(aux)
        
        # access the HDF5
        name = self.name % recordId
        filePath = os.path.join(self.path, name)
        f = h5.hdf(filePath, mode='r+')
        f.delEventType(weg)
        f.close()
        
        # update the DB
        if len(wegits) <= 1:
            op = '$set'
            val = {}
        else:
            op = '$unset'
            val = 1
        self.collection.update({'_id': recordId}, {op: {wegM: val}})
        
        # experimental sync
        if self.parent.altSync:
            self.parent.queue.put({'function': 'records.delEventType'})
    
    def delTags(self, recordId=None, tags=None):
        """
        
        Delete tags from a record.
        
        Kwargs:
            recordId (int): ID of the record.
            
            tags (list): Tags to add.
        
        Kwrvals:
            None
        
        See Also:
            
        
        Notes:
            
        
        Example:
            db.records.delTags(0, ['a', 'b', 'c'])
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if recordId is None:
            raise TypeError, "A record ID must be provided."
        if tags is None:
            return
        
        # update the db
        doc = self.collection.find_and_modify(query={'_id': recordId},
                                              update={'$pullAll': {'tags': tags}},
                                              new=True)
        
        # update the HDF5
        name = self.name % recordId
        filePath = os.path.join(self.path, name)
        f = h5.hdf(filePath, mode='a')
        header = f.getInfo()['header']
        try:
            header['tags'] = doc['tags']
        except KeyError:
            pass
        f.addInfo(header)
        f.close()
        
        # experimental sync
        if self.parent.altSync:
            self.parent.queue.put({'function': 'records.delTags'})



#class biomesh_off():
#    
#    def __init__(self, dbName=None, dstPath='~'):
#        
#        # check inputs
#        if dbName is None:
#            raise TypeError, "A DB name must be specified."
#        
#        # path to write HDF5 files
#        dstPath = _hdf5Folder(dbName, dstPath)
#        meta = os.path.join(dstPath, 'ExpSub.hdf5')
#        
#        # self things
#        self.dbName = dbName
#        self.dstPath = dstPath
#        self.experiments = experiments_off(dstPath, meta)
#        self.subjects = subjects_off(dstPath, meta)
#        self.records = records_off(dstPath, meta)
#        
#        # meta HDF5
#        if not os.path.exists(meta):
#            fid = h5.meta(meta)
#            fid.setDB(dbName)
#            fid.close()
#            self.experiments.add({'_id':0,
#                                  'name': 'unknown',
#                                  'description': 'Generic experiment.',
#                                  'goals': 'To store records from unknown experiments.',
#                                  'records':[]})
#            self.subjects.add({'_id':0,
#                               'name': 'unnamed',
#                               'records': []})
#    
#    
#    def subsInExp(self):
#        pass
#    
#    
#    def expsInSub(self):
#        pass
#    
#    
#    def close(self):
#        return None
#
#
#
#class subjects_off():
#    
#    def __init__(self, dstPath, meta):
#        pass
#
#
#
#class experiments_off():
#    
#    def __init__(self, dstPath, meta):
#        pass
#
#
#
#class records_off():
#    
#    def __init__(self, dstPath, meta):
#        pass



class dataContainer():
    """
    
    Data container for signals and events.
    
    Kwargs:
        
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1]
        
    """
    
    def __init__(self, case=None, mdata={}, signal=[], timeStamps=[], values=[]):
        """
        
        Initialize the container according to case (signal or events).
        
        Kwargs:
            case (str): Case ('signals' or 'events') to store.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if case is None:
            raise TypeError, "A case ('signals' or 'events') must be specified."
        
        if case == 'signals':
            self.signal = signal
        elif case == 'events':
            self.timeStamps = timeStamps
            self.values = values
        else:
            raise ValueError, "Undefined case ('%s')!" % str(case)
        
        self.metadata = mdata
        self._case = case
    
    
    def __str__(self):
        """
        
        str operator.
        
        Kwargs:
            
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        return str((self._case, self.metadata))



class dataSelector():
    """
    
    Wrapper for operator overloading to records.
    
    Kwargs:
        
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1]
        
    """
    
    def __init__(self, recordsInst, recordId, dataType='/'):
        """
        
        Initiate the class.
        
        Kwargs:
            recordsInst (biomesh.records): Instance of the records collection.
            
            recordId (int): ID of the record.
            
            dataType (str): Type of the data. Default: '/'.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        self.records = recordsInst
        self.recordId = recordId
        self.type = dataType
        
    def __getitem__(self, key):
        """
        
        x.__getitem__(y) <==> x[y]
        
        Kwargs:
            key (str, int, slice, list): Item to retrieve.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        if isinstance(key, basestring):
            # add to the type
            if self.type == '/':
                weg = self.type + key
            else:
                weg = self.type + '/' + key
            wegits = weg.split('/')
            wegits = filter(lambda a: a != '', wegits)
            if wegits[0] not in ['signals', 'events']:
                raise KeyError, "Specify the case first: 'signals' or 'events'."
            return dataSelector(self.records, self.recordId, weg)
        elif type(key) is int:
            # get the data (single)
            if self.type == '/':
                raise KeyError, "Specify the case first: 'signals' or 'events'."
            wegits = self.type.split('/')
            wegits = filter(lambda a: a != '', wegits)
            weg = '/' + string.join(wegits[1:], '/')
            case = wegits[0]
            if case == 'signals':
                out = self.records.getSignal(self.recordId, weg, key)
                contnr = dataContainer(case=case, mdata=out['mdata'], signal=out['signal'])
            elif case == 'events':
                out = self.records.getEvent(self.recordId, weg, key)
                contnr = dataContainer(case=case, mdata=out['mdata'], timeStamps=out['timeStamps'], values=out['values'])
            return contnr
        elif type(key) is slice:
            # get the data (slice)
            contnr = []
            refList = range(self.__len__())
            refList = refList[key]
            for item in refList:
                try:
                    contnr.append(self.__getitem__(item))
                except (KeyError, IndexError):
                    contnr.append(None)
            return contnr
        elif type(key) is list:
            # get the data (list)
            contnr = []
            for item in key:
                try:
                    contnr.append(self.__getitem__(item))
                except (KeyError, IndexError):
                    contnr.append(None)
            return contnr
        else:
            print "Key type not supported."
    
    
    def __len__(self):
        """
        
        Number of datasets in the record.
        
        x.__len__() <==> len(x)
        
        Kwargs:
            
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            If the case (signal or events) is still unspecified, returns the total number of datasets. Otherwise, returns the number of datasets on the current type.
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        res = self.list()
        
        try:
            return len(res['signalTypes']) + len(res['eventTypes'])
        except KeyError:
            return len(res['local'])
    
    
    def __iter__(self):
        """
        
        Iterator operator.
        
        Kwargs:
            
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        return self._dataSelectorIter()
    
    
    def _dataSelectorIter(self):
        """
        
        Iterator.
        
        Kwargs:
            
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        item = 0
        while True:
            try:
                yield self.__getitem__(item)
            except (KeyError, IndexError):
                break
            item += 1
    
    
    def __str__(self):
        """
        
        str operator.
        
        Kwargs:
            
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        return str((self.recordId, self.type))
    
    
    def list(self):
        """
        
        List the datasets.
        
        Kwargs:
            
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        if self.type == '/':
            return self.records.listTypes(self.recordId)
        else:
            wegits = self.type.split('/')
            wegits = filter(lambda a: a != '', wegits)
            res = self.records.getById(self.recordId, restrict={wegits[0]: 1})
            if res['doc'] is None:
                raise ValueError, "No record found with given ID ({0}).".format(self.recordId)
            
            aux = res['doc']
            for item in wegits:
                aux = aux[item]
            
            # cycle the keys
            loc = []
            sub = []
            for key in aux:
                if key == 'loc':
                    for i in range(len(aux[key])):
                        loc.append((i, aux[key][i]['name']))
                else:
                    sub.append(key)
            
            sub.sort()
            return {'local': loc, 'subTypes': sub}


    
if __name__ == '__main__':
    # Example
    import datetime
    import time
    
    # open connection to DB
    db = biomesh(dbName='biomesh_tst', host='193.136.222.234', port=27017, dstPath='~/tmp/biomesh', srvPath='/BioMESH', sync=True, altSync=True)
    
    
    
    # subjects
    print "Testing subjects\n"
    subject = {'name': 'subject', 'age': 0, 'gender': 'm', 'email': 'sample@domain.com'}
    print subject, '\n'
    
    # add
    subId = db.subjects.add(subject)['subjectId']
    
    # getByName
    res = db.subjects.getByName(subject['name'])
    print res, '\n'
    
    # getById
    res = db.subjects.getById(subId)
    print res, '\n'
    
    # get
    res = db.subjects.get()
    print res, '\n'
    
    # update
    db.subjects.update(subId, {'new': 'field'})
    res = db.subjects.getById(subId)
    print res, '\n'
    
    # list
    res = db.subjects.list()
    print res, '\n'
    
    
    
    # experiments
    print "Testing experiments"
    experiment = {'name': 'experiment', 'description': 'This is a sample experiment.', 'goals': 'To illustrate an experiment document.'}
    print experiment, '\n'
    
    # add
    expId = db.experiments.add(experiment)['experimentId']
    
    # getByName
    res = db.experiments.getByName(experiment['name'])
    print res, '\n'
    
    # getById
    res = db.experiments.getById(expId)
    print res, '\n'
    
    # get
    res = db.experiments.get()
    print res, '\n'
    
    # update
    db.experiments.update(experiment['name'], {'new': 'field'})
    res = db.experiments.getById(expId)
    print res, '\n'
    
    # list
    res = db.experiments.list()
    print res, '\n'
    
    
    # records
    print "Testing records\n"
    record = {'name': 'record', 'date': datetime.datetime.utcnow().isoformat(), 'experiment': 'experiment', 'subject': subId, 'supervisor': 'supervisor'}
    print record, '\n'
    
    # add
    recId = db.records.add(record)['recordId']
    
    # addSignal
    signal = np.zeros((100, 5), dtype='float64')
    mdataS = {'comments': 'zeros'}
    db.records.addSignal(recId, '/test', signal, mdataS)
    
    # addEvent
    nts = 100
    timeStamps = []
    now = datetime.datetime.utcnow()
    for i in range(nts):
        instant = now + datetime.timedelta(seconds=i)
        timeStamps.append(time.mktime(instant.timetuple()))
    timeStamps = np.array(timeStamps, dtype='float64')
    values = np.zeros((nts, 1), dtype='float64')
    mdataE = {'comments': 'zeros'}
    db.records.addEvent(recId, '/test', timeStamps, values, mdataE)
    
    # getByName
    res = db.records.getByName(record['name'])
    print res, '\n'
    
    # getById
    res = db.records.getById(recId)
    print res, '\n'
    
    # get
    res = db.records.get()
    print res, '\n'
    
    # getSignal
    res = db.records.getSignal(recId, '/test', 0)
    print res, '\n'
    
    # getEvent
    res = db.records.getEvent(recId, '/test', 0)
    print res, '\n'
    
    # getAll
    res = db.records.getAll()
    print res, '\n'
    
    # listTypes
    res = db.records.listTypes(recId)
    print res, '\n'
    
    # update
    db.records.update(recId, {'new': 'field'})
    res = db.records.getById(recId)
    print res, '\n'
    
    # delSignal
    mdata = {'comments': 'zeros'}
    db.records.addSignal(recId, '/test/delete', signal, mdata)
    db.records.delSignal(recId, '/test/delete', 0)
    res = db.records.getById(recId)
    print res, '\n'
    
    # delSignalType
    db.records.delSignalType(recId, '/test/delete')
    res = db.records.getById(recId)
    print res, '\n'
    
    # delEvent
    mdata = {'comments': 'zeros'}
    db.records.addEvent(recId, '/test/delete', timeStamps, values, mdata)
    db.records.delEvent(recId, '/test/delete', 0)
    res = db.records.getById(recId)
    print res, '\n'
    
    # delEventType
    db.records.delEventType(recId, '/test/delete')
    res = db.records.getById(recId)
    print res, '\n'
    
    # dataMagic
    ty = db.records[recId]['signals'].list()
    print ty, '\n'
    ty = db.records[recId]['signals']['test'].list()
    print ty, '\n'
    data = db.records[recId]['signals']['test'][0]
    print data.metadata, '\n'
    print data.signal, '\n'
    
    ty = db.records[recId]['events'].list()
    print ty, '\n'
    ty = db.records[recId]['events']['test'].list()
    print ty, '\n'
    event = db.records[recId]['events']['test'][0]
    print event.metadata
    print event.timeStamps
    print event.values, '\n'
    
    # print db.drop(False)
    
    db.close()
    
    
