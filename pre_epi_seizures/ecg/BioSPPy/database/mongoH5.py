"""
	This module provides an interface to a biosognals database. See the specification at http://camoes.lx.it.pt/MediaWiki/index.php/Database_Specification#Data_Model.
	Requires the HDF5DB module.
	
	Functions:
		
		
"""


import pymongo as pmg
import HDF5DB as h5
import glob
import os

import sys
sys.path.append("../")


def bioDB(dbName=None, host='localhost', port=27017, path='~'):
	"""
	To establish a connection to a MongoDB server. If the database (DB) does not exist, one is created with the necessary basic structures.
	
	Parameters
	__________
	dbName : str
		Name of the DB to connect to.
	host : str, optional
		Network address of the MongoDB server (e.g. '127.0.0.1'  or 'localhost').
		Default: 'localhost'
	port : int, optional
		Port the MongoDB server is listening on.
		Default: 27017
	path : str, optional
		Path to store the HDF5 files.
		Default: '~'
	
	Returns
	_______
	kwrvals : dict
		A keyworded return values dict is returned with the following keys:
			'db' : pymongo.database.Database instance
				An instance representing the DB.
			'path' : str
				The path to store the HDF5 files.
	
	See Also
	________
	pymongo.database.Database
	
	Notes
	_____
	
	
	Example
	_______
	db = bioDB('test')
	
	References
	__________
	.. [1] 
	
	http://
	"""	
	
	# check inputs
	if dbName is None:
		raise TypeError, "A DB name must be specified."
	
	# establish connection
	connection = pmg.Connection(host, port)
	db = connection[dbName]
	
	# check basic structures
	if db.collection_names() == []:
		# DB is empty, create unknown experiment and subject
		db.experiments.insert({'_id':0, 'name': 'unknown', 'description': 'Generic experiment.', 'goals': 'To store records from unknown experiments.', 'records':[]})
		db.subjects.insert({'_id':0, 'name': 'unnamed', 'records': []})
	
	# ensure indexes
	db.experiments.ensure_index('name')
	db.subjects.ensure_index('name')
	db.records.ensure_index('date')
	db.records.ensure_index('experiment')
	db.records.ensure_index('subject')
	
	# path to write HDF5 files
	if path == '~':
		userPath = os.path.expanduser(path)
		path = os.path.join(userPath, dbName, 'hdf5')
	path = os.path.abspath(path)
	# make sure the path exists
	_hdf5Folder(path)
	
	# kwrvals
	kwrvals = {}
	kwrvals['db'] = db
	kwrvals['path'] = path
	
	return kwrvals


def dropDB(dbName=None, host='localhost', port=27017, flag=True):
	"""
	Function to remove a database from the server. The HDF5 files are preserved.
	
	Parameters
	__________
	dbName : str
		Name of the DB to drop.
	host : str
		Network address of the MongoDB server (e.g. '127.0.0.1' or 'localhost').
		Default: 'localhost'
	port : int
		Port the MongoDB server is listening on.
		Default: 27017
	flag : boolean
		Flag to override user confirmation about removal. Set to False to avoid user prompt.
	
	Returns
	_______
	
	
	See Also
	________
	
	Notes
	_____
	
	
	Example
	_______
	
	
	References
	__________
	.. [1] 
	
	http://
	"""
	
	# check inputs
	if dbName is None:
		raise TypeError, "A DB name must be specified."
	
	# connect to DB
	connection = pmg.Connection(host, port)
	db = connection[dbName]
	
	# make sure
	prompt = 'Are you sure you want to drop the databse "{0}" from the server at "{1}"? (y/n?)\nRemoved data cannot be restored!\n'.format(dbName, host)
	sure = True
	if flag:
		var = raw_input(prompt)
		if var != 'y' and var != 'Y':
			sure = False
	
	out = None
	if sure:
		# drop command
		out = db.command('dropDatabase')
	
	# close connection
	connection.close()
	
	return out


def listDB(host='localhost', port=27017):
	"""
	Lists the databases present in a given server.
	
	Parameters
	__________
	host : str
		Network address of the MongoDB server (e.g. '127.0.0.1'  or 'localhost').
		Default: 'localhost'
	port : int
		Port the MongoDB server is listening on.
		Default: 27017
	
	Returns
	_______
	kwrvals : dict
		A keyworded return values dict is returned with the following keys:
			'dbList' : list
				A list with DB names.
	
	See Also
	________
	
	Notes
	_____
	
	
	Example
	_______
	
	References
	__________
	.. [1] 
	
	http://
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


def closeDB(db=None, **kwarg):
	"""
	Close the connection to the database.
	
	Parameters
	__________
	db : pymongo.database.Database instance
		An instance representing the DB.
	
	Returns
	_______
	
	See Also
	________
	
	Notes
	_____
	
	
	Example
	_______
	
	
	References
	__________
	.. [1] 
	
	http://
	"""
	
	# check inputs
	if db is None:
		raise TypeError, "A DB instance must be provided."
	
	# close connection
	db.connection.close()


def _hdf5Folder(path=None):
	"""
	Check if the path for the HDF5 files exists, create if needed.
	
	Parameters
	__________
	path : str
		Path to store the HDF5 files.
	
	Returns
	_______
	
	See Also
	________
	
	Notes
	_____
	
	
	Example
	_______
	
	
	References
	__________
	.. [1] 
	
	http://
	"""
	
	# check inputs
	if path is None:
		raise TypeError, "A path must be provided."
	
	bits = []
	new = path
	while not os.path.exists(new):
		aux = os.path.split(new)
		new = aux[0]
		bits.append(aux[1])
	
	if not bits == []:
		partial = new
		for i in range(len(bits)):
			partial = os.path.join(partial, bits[-(i+1)])
			try:
				os.mkdir(partial)
			except OSError:
				print "Failed to create the directory for the HDF5 files. Check path spelling, or permissions."
				raise



class subjects:
	"""
	Class to operate on subjects via MongoDB.
	
	Parameters
	__________
	
	
	Returns
	_______
	
	See Also
	________
	
	Notes
	_____
	
	
	Example
	_______
	
	
	References
	__________
	.. [1] 
	
	http://
	"""
	
	def __init__(self, db=None, **kwarg):
		"""
		Initialize the subjects class.
		
		Parameters
		__________
		db : pymongo.database.Database instance
			An instance representing the DB.
		path : str
			The path to store the HDF5 files.
		
		Returns
		_______
		
		See Also
		________
		
		Notes
		_____
		
		
		Example
		_______
		
		
		References
		__________
		.. [1] 
		
		http://
		"""
		
		# check inputs
		if db is None:
			raise TypeError, "A DB instance must be provided."
		
		self.collection = db['subjects']
	
	
	def add(self, subject=None):
		"""
        To add a subject to the DB's 'subjects' collection.
		
        Parameters
        __________
        subject : dict
			Subject (JSON) to add. 
        
        Returns
        _______
        kwrvals : dict
			A keyworded return values dict is returned with the following keys:
				'subjectId' : int
					ID of the subject in the DB.
        
        See Also
        ________
        experiments.add
		records.add
        
        Notes
        _____
        If the subject already exists (i.e. there is a subject with the same name) the ID of the subject in the DB is returned.
        
        Example
        _______
		db = bioDB('test')
		subs = subjects(**db)
        subject = {'name': 'Mr. Sample Subject', 'age': 0, 'sex': 'm', 'email': 'sample@domain.com'}
		out = subs.add(subject)
        
        References
        __________
        .. [1] 
        
        http://
        """	
		
		# check inputs
		if subject is None:
			raise TypeError, "A JSON subject must be provided."
		
		# check if subject already exists (by name)
		res = self.getByName(subject['name'], restrict={'_id':1})
		
		# kwrvals
		kwrvals = {}
		
		if not res['doc'] is None:
			# the subject already exists
			print 'Warning: subject already exists in DB; skipping new insertion.'
			kwrvals['subjectId'] = res['doc']['_id']
			return kwrvals
		
		if not subject.has_key('_id'):
			# get a new id (starts at zero)
			newId = self.collection.count()
			
			# add id to subject document
			subject.update({'_id':newId})
		
		if not subject.has_key('records'):
			# add the "records" field
			subject.update({'records':[]})
		
		# save to db
		self.collection.insert(subject)
		
		kwrvals['subjectId'] = subject['_id']
		return kwrvals
		
	
	def _addExperiment(self, subjectId, experimentName):
		"""
        To add an experiment to the given subject.
		
        Parameters
        __________
        subjectId : int
			ID of the subject.
		experimentName : str
			Name of the experiment.
        
        Returns
        _______
        
        
        See Also
        ________
        
        
        Notes
        _____
        
        
        Example
        _______
        
        
        References
        __________
        .. [1] 
        
        http://
        """	
		
		if not self._hasExperiment(subjectId, experimentName):
			self.collection.update({'_id':subjectId}, {'$push': {'records': {'experiment':experimentName, 'list':[]}}})
		
		
	def _hasExperiment(self, subjectId, experimentName):
		"""
        Function to check if a given experiment is already included in a given subject.
		
        Parameters
        __________
        subjectId : int
			ID of the subject.
		experimentName : str
			Name of the experiment.
        
        Returns
        _______
         : boolean
		 True if experiment is present, False otherwise.
		
        
        See Also
        ________
        
        
        Notes
        _____
        
        
        Example
        _______
        
        
        References
        __________
        .. [1] 
        
        http://
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
	
	def _addRecord(self, subjectId, experimentName, recordId):
		"""
        To add a record link to a given subject.
		
        Parameters
        __________
        subjectId : int
			ID of the subject.
		experimentName : str
			Name of the experiment.
		recordId : int
			ID of the record.
        
        Returns
        _______
        
        
        See Also
        ________
        
        
        Notes
        _____
        
        
        Example
        _______
        
        
        References
        __________
        .. [1] 
        
        http://
        """	
		# add a new record to subject
		
		self._addExperiment(subjectId, experimentName)
		
		if not self._hasRecord(subjectId, experimentName, recordId):
			self.collection.update({'_id':subjectId, 'records.experiment':experimentName}, {'$push':{'records.$.list':recordId}})
		
	
	def _hasRecord(self, subjectId, experimentName, recordId):
		"""
        Check if subject has a given record associated with a certain experiment.
		
        Parameters
        __________
        subjectId : int
			ID of the subject.
		experimentName : str
			Name of the experiment.
		recordId : int
			ID of the record.
        
        Returns
        _______
         : boolean
		True if the subject has the record under the given experiment, False otherwise.
        
        See Also
        ________
        
        
        Notes
        _____
        
        
        Example
        _______
        
        
        References
        __________
        .. [1] 
        
        http://
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
		
        Parameters
        __________
        subjectName : str
			Name of the subject to query.
		restrict : dict
			Dictionary to restrict the information sent by the DB.
			Default={}
        
        Returns
        _______
        kwrvals : dict
			A keyworded return values dict is returned with the following keys:
				'doc' : dict
					The document with the results of the query. None if no match is found.
        
        See Also
        ________
        
        
        Notes
        _____
        
        
        Example
        _______
        getByName('unnamed', restrict={'email':1})
        
        References
        __________
        .. [1] 
        
        http://
        """	
		
		# check inputs
		if subjectName is None:
			raise TypeError, "A subject name must be provided."
		
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
		
        Parameters
        __________
        subjectId : int
			ID of the subject to query.
			Default={}
		restrict : dict
			Dictionary to restrict the information sent by the DB.
			Default={}
        
        Returns
        _______
        kwrvals : dict
			A keyworded return values dict is returned with the following keys:
				'doc' : dict
					The document with the results of the query. None if no match is found.
        
        See Also
        ________
        
        
        Notes
        _____
        
        
        Example
        _______
        getById(0, restrict={'email':1})
        
        References
        __________
        .. [1] 
        
        http://
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
		
        Parameters
        __________
		refine : dict
			Dictionary to refine the search.
			Default={}
		restrict : dict
			Dictionary to restrict the information sent by the DB.
			Default={}
        
        Returns
        _______
        kwrvals : dict
			A keyworded return values dict is returned with the following keys:
				'docList' : list
					The list of dictionaries with the results of the query. Empty list if no match is found.
        
        See Also
        ________
        
        
        Notes
        _____
        
        
        Example
        _______
        get(refine={'age':100}, restrict={'email':1})
        
        References
        __________
        .. [1] 
        
        http://
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
		
	
	def update(self, subjectId=None, info={}):
		"""
        Update a subject with the given information. Fiels can be added, and its type changed, but not deleted.
		
        Parameters
        __________
        subjectId : int
			ID of the subject to update.
		info : dict
			Dictionary with the information to add.
			Default={}
        
        Returns
        _______
        
        
        See Also
        ________
        
        
        Notes
        _____
        
        
        Example
        _______
        
        
        References
        __________
        .. [1] 
        
        http://
        """	
		
		# check inputs
		if subjectId is None:
			raise TypeError, "A subject ID must be provided."
		
		# don't update the following fields
		fields = ['records']
		for item in fields:
			if info.has_key(item):
				info.pop(item)
		
		# update
		self.collection.update({'_id': subjectId}, {'$set': info})
	
	def _delExperiment(self, subjectId, experimentName):
		"""
        Remove an experiment from a subject.
		
        Parameters
        __________
        subjectId : int
			ID of the subject.
		experimentName : str
			Name of the experiment to remove.
        
        Returns
        _______
        
        
        See Also
        ________
        
        
        Notes
        _____
        
        
        Example
        _______
        
        
        References
        __________
        .. [1] 
        
        http://
        """	
		
		self.collection.update({'_id':subjectId}, {'$pull':{'records':{'experiment':experimentName}}})
	
	
	def _delRecord(self, subjectId, experimentName, recordId):
		"""
        Remove a record (belonging to a certain experiment) from a subject.
		
        Parameters
        __________
        subJectId : int
			ID of the subject.
		experimentName : str
			Name of the experiment the record belengs to.
		recordId : int
			ID of the record to remove.
        
        Returns
        _______
        
        
        See Also
        ________
        
        
        Notes
        _____
        
        
        Example
        _______
        
        
        References
        __________
        .. [1] 
        
        http://
        """	
		
		self.collection.update({'_id': subjectId, 'records.experiment': experimentName}, {'$pull': {'records.$.list': recordId}})



class experiments:
	"""
	Class to operate on experiments via MongoDB.
	
	Parameters
	__________
	
	
	Returns
	_______
	
	See Also
	________
	
	Notes
	_____
	
	
	Example
	_______
	
	
	References
	__________
	.. [1] 
	
	http://
	"""
	
	def __init__(self, db=None, **kwarg):
		"""
		Initialize the experiments class.
		
		Parameters
		__________
		db : pymongo.database.Database instance
			An instance representing the DB.
		
		Returns
		_______
		
		See Also
		________
		
		Notes
		_____
		
		
		Example
		_______
		
		
		References
		__________
		.. [1] 
		
		http://
		"""
		# initiate the collection
		
		# check inputs
		if db is None:
			raise TypeError, "A DB instance must be provided."
		
		self.collection = db['experiments']
	
	def add(self, experiment=None):
		"""
        To add an experiment to the DB's 'experiments' collection.
		
        Parameters
        __________
        experiment : dict
			Experiment (JSON) to add.
        
        Returns
        _______
        kwrvals : dict
			A keyworded return values dict is returned with the following keys:
				'experimentId': int
					ID of the experiment in the DB.
        
        See Also
        ________
        subjects.add
		records.add
        
        Notes
        _____
        If the experiment already exists (i.e. there is an experiment with the same name) the ID of the experiment in the DB is returned.
        
        Example
        _______
        db = bioDB('test')
		exps = experiments(**db)
        experiment = {'name': 'exp1', 'description': 'This is a sample experiment.', 'goals': 'To illustrate an experiment document.'}
		out = exps.add(experiment)
        
        References
        __________
        .. [1] 
        
        http://
        """	
		
		# check inputs
		if experiment is None:
			raise TypeError, "A JSON experiment must be provided."
		
		# check if experiment already exists (by name)
		res = self.getByName(experiment['name'], restrict={'_id':1})
		
		# kwrvals
		kwrvals = {}
		
		if not res['doc'] is None:
			# the experiment already exists
			print 'Warning: experiment already exists in DB; skipping new insertion.'
			kwrvals['experimentId'] = res['doc']['_id']
			return kwrvals
		
		if not experiment.has_key('_id'):
			# get a new id (starts at zero)
			newId = self.collection.count()
			
			# add id to subject document
			experiment.update({'_id':newId})
		
		if not experiment.has_key('records'):
			# add the "records" field
			experiment.update({'records':[]})
		
		# save to db
		self.collection.insert(experiment)
		
		kwrvals['experimentId'] = experiment['_id']
		return kwrvals

		
	def _addSubject(self, experimentName, subjectId):
		"""
        To add a subject to the given experiment.
		
        Parameters
        __________
        experimentName : str
			Name of the experiment.
		subjectId : int
			ID of the subject.
        
        Returns
        _______
        
        
        See Also
        ________
        
        
        Notes
        _____
        
        
        Example
        _______
        
        
        References
        __________
        .. [1] 
        
        http://
        """	
		# add a new subject to experiment
		
		if not self._hasSubject(experimentName, subjectId):
			self.collection.update({'name':experimentName}, {'$push': {'records': {'subject':subjectId, 'list':[]}}})
		
	def _hasSubject(self, experimentName, subjectId):
		"""
        Function to check if a given subject is already included in a given experiment.
		
        Parameters
        __________
        experimentName : str
			Name of the experiment.
		subjectId : int
			ID of the subject.
        
        Returns
        _______
         : boolean
		 True if subject is present, False otherwise.
        
        See Also
        ________
        
        
        Notes
        _____
        
        
        Example
        _______
        
        
        References
        __________
        .. [1] 
        
        http://
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
	
	
	def _addRecord(self, experimentName, subjectId, recordId):
		"""
        To add a record link to a given experiment.
		
        Parameters
        __________
        experimentName : str
			Name of the experiment.
		subjectId : int
			ID of the subject.
		recordId : int
			ID of the record.
        
        Returns
        _______
        
        
        See Also
        ________
        
        
        Notes
        _____
        
        
        Example
        _______
        
        
        References
        __________
        .. [1] 
        
        http://
        """	
		
		self._addSubject(experimentName, subjectId)
		
		if not self._hasRecord(experimentName, subjectId, recordId):
			self.collection.update({'name':experimentName, 'records.subject':subjectId}, {'$push':{'records.$.list':recordId}})
	
	
	def _hasRecord(self, experimentName, subjectId, recordId):
		"""
        Check if experiment has a given record associated with a certain subject.
		
        Parameters
        __________
        experimentName : str
			Name of the experiment.
		subjectId : int
			ID of the subject.
		recordId : int
			ID of the record.
        
        Returns
        _______
         : boolean
		True if the experiment has the record under the given experiment, False otherwise.
        
        See Also
        ________
        
        
        Notes
        _____
        
        
        Example
        _______
        
        
        References
        __________
        .. [1] 
        
        http://
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
		
        Parameters
        __________
        experimentName : str
			Name of the experiment to query.
		restrict : dict
			Dictionary to restrict the information sent by the DB.
			Default={}
        
        Returns
        _______
        kwrvals : dict
			A keyworded return values dict is returned with the following keys:
				'doc' : dict
					The document with the results of the query. None if no match in found.
        
        See Also
        ________
        
        
        Notes
        _____
        
        
        Example
        _______
        getByName('unknown', restrict={'goals':1})
        
        References
        __________
        .. [1] 
        
        http://
        """	
		
		# check inputs
		if experimentName is None:
			raise TypeError, "An experiment name must be specified."
		
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
		
        Parameters
        __________
        experimentId : int
			ID of the experiment to query.
		restrict : dict
			Dictionary to restrict the information sent by the DB.
			Default={}
        
        Returns
        _______
        kwrvals : dict
			A keyworded return values dict is returned with the following keys:
				'doc' : dict
					The document with the results of the query. None if no match is found.
        
        See Also
        ________
        
        
        Notes
        _____
        
        
        Example
        _______
        getById(0, restrict={'email':1})
        
        References
        __________
        .. [1] 
        
        http://
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

		Parameters
		__________
		refine : dict
			Dictionary to refine the search.
			Default={}
		restrict : dict
			Dictionary to restrict the information sent by the DB.
			Default={}

		Returns
		_______
		kwrvals : dict
			A keyworded return values dict is returned with the following keys:
				'docList' : list
					The list of documents with the results of the query. Empty list if no match is found.

		See Also
		________


		Notes
		_____


		Example
		_______
		get(refine={'age':100}, restrict={'email':1})

		References
		__________
		.. [1] 

		http://
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
	
	
	def update(self, experimentName=None, info={}):
		"""
        Update an experiment with the given information. Fiels can be added, and its type changed, but not deleted.
		
        Parameters
        __________
        experimentName : str
			Name of the experiment to update.
		info : dict
			Dictionary with the information to add.
			Default={}
        
        Returns
        _______
        
        
        See Also
        ________
        
        
        Notes
        _____
        
        
        Example
        _______
        
        
        References
        __________
        .. [1] 
        
        http://
        """
		
		# check inputs
		if experimentName is None:
			raise TypeError, "An experiment name must be provided."
		
		# don't update the following fields
		fields = ['records']
		for item in fields:
			if info.has_key(item):
				info.pop(item)
		
		
		# update
		self.collection.update({'name': experimentName}, {'$set': info})
	
	
	def _delSubject(self, experimentId, subjectId):
		"""
        Remove a subject from an experiment.
		
        Parameters
        __________
        experimentId : int
			ID of the experiment.
		subjectId : int
			ID of the subject.
        
        Returns
        _______
        
        
        See Also
        ________
        
        
        Notes
        _____
        
        
        Example
        _______
        
        
        References
        __________
        .. [1] 
        
        http://
        """	
		# remove a subject from an experiment
		
		self.collection.update({'_id':experimentId}, {'$pull':{'records':{'subject': subjectId}}})
	
	
	def _delRecord(self, experimentId, subjectId, recordId):
		"""
        Remove a record (belonging to a certain subject) from an experiment.
		
        Parameters
        __________
        experimentId : int
			ID of the experiment.
		subjectId : int
			ID of the subject.
		recordId : int
			ID of the record to remove.
        
        Returns
        _______
        
        
        See Also
        ________
        
        
        Notes
        _____
        
        
        Example
        _______
        
        
        References
        __________
        .. [1] 
        
        http://
        """	
		# remove a record from an experiment
		
		self.collection.update({'_id': experimentId, 'records.subject': subjectId}, {'$pull': {'records.$.list': recordId}})
	


class records:
	"""
	Class to operate on records via MongoDB.
	
	Parameters
	__________
	
	
	Returns
	_______
	
	See Also
	________
	
	Notes
	_____
	
	
	Example
	_______
	
	
	References
	__________
	.. [1] 
	
	http://
	"""
	
	def __init__(self, db=None, path=None):
		"""
		Initialize the records class.
		
		Parameters
		__________
		db : pymongo.database.Database instance
			An instance representing the DB.
		path : str
			The path to store the HDF5 files.
		
		Returns
		_______
		
		See Also
		________
		
		Notes
		_____
		
		
		Example
		_______
		
		
		References
		__________
		.. [1] 
		
		http://
		"""
		# initiate the collection
		
		# check inputs
		if db is None:
			raise TypeError, "A DB instance must be provided."
		if path is None:
			raise TypeError, "A path to the HDF5 files must be privided."
		
		self.collection = db['records']
		self.experiments = experiments(db)
		self.subjects = subjects(db)
		self.name = 'rec_%d.hdf5'
		self.path = path

	def add(self, record=None, flag=True):
		"""
        To add a record to the DB's 'records' collection.
		
        Parameters
        __________
        record : dict
			Record (JSON) to add.
        
        Returns
        _______
        kwrvals : dict
			A keyworded return values dict is returned with the following keys:
				'recordId' : int
					ID of the record in the DB.
        
        See Also
        ________
        subjects.add
		experiments.add
        
        Notes
        _____
        A new record is always added, regardless of the existence of records with the same name. If the record to add has no 'experiment' and/or 'subject' fields, it will be linked to a generic experiment ('unknown') and/or to a generic subject ('unnamed').
        
        Example
        _______
        db = bioDB('test')
		recs = records(**db)
        record = {'name': 'sig0', 'session': 0, 'date': 'today', 'experiment': 'exp1', 'subject': 'Mr. Sample Subject'}
		out = recs.add(record)
        
        
        References
        __________
        .. [1] 
        
        http://
        """
		
		# check inputs
		if record is None:
			raise TypeError, "A JSON record must be provided."
		
		# get experiment name/ID
		if not record.has_key('experiment'):
			print("record not assigned to an experiment.\nStoring in 'unknown' experiments.")
			record.update({'experiment':'unknown'})
		experiment = record['experiment']
		if type(experiment) is not str:
			experiment = self.experiments.getById(experiment, restrict={'name':1})['doc']['name']
		
		# get subject ID/name
		if not record.has_key('subject'):
			print("Record has no subject.\nStoring in 'unnamed' subject.")
			record.update({'subject':'unnamed'})
		subject = record['subject']
		if type(subject) is str:
			subject = self.subjects.getByName(subject, restrict={'_id':1})['doc']['_id']
		
		# generate session number
		if not record.has_key('session'):
			session = self._counter(subject, experiment)
			record.update({'session':session})
		session = record['session']
		
		# get or generate ID (maybe change this to always generate???)
		if not record.has_key('_id'):
			# get a new id (starts at zero)
			newId = self.collection.count()
			# add id to subject document
			record.update({'_id':newId})
		recordId = record['_id']
		
		if flag:
			# save to new HDF5
			name = self.name % recordId
			filePath = os.path.join(self.path, name)
			f = h5.hdf(filePath, mode='w')
			f.addInfo(record)
			f.close()
		
		# other things for the DB
		if not record.has_key('audit'):
			# add the "audit" field
			record.update({'audit':[]})
			
		if not record.has_key('data'):
			# add the "data" field
			record.update({'data':{}})
			
		if not record.has_key('events'):
			# add the "events" field
			record.update({'events':{}})
		
		# # update record name
		# record.update({'name':(record['name']+'_'+str(session))})
		
		# set counts to zero
		record.update({'nbData':0, 'nbEvents':0})
		
		# save to db
		self.collection.insert(record)
		self.subjects._addRecord(subject, experiment, recordId)
		self.experiments._addRecord(experiment, subject, recordId)
		
		# kwrvals
		kwrvals = {}
		kwrvals['recordId'] = recordId
		
		return kwrvals
	
	
	def _counter(self, subjectId, experimentName, debug=False):
		"""
        Count the number of sessions a subject atended a given experiment.
		
        Parameters
        __________
        subjectId : int
			ID of the subject.
		experimentName : str
			Name of the experiment.
        
        Returns
        _______
        session : int
		Number of sessions.
        
        See Also
        ________
        
        
        Notes
        _____
        
        
        Example
        _______
        
        
        References
        __________
        .. [1] 
        
        http://
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
		
		
	
	
	def addAudit(self, recordId=None, audit={}):
		"""
        To add information to 'audit' field.
		
        Parameters
        __________
        recordId : int
			ID of the record.
		audit : dict
			Information to add.
			Default={}
        
        Returns
        _______
        
        
        See Also
        ________
        
        
        Notes
        _____
        
        
        Example
        _______
        
        
        References
        __________
        .. [1] 
        
        http://
        """	
		
		# check inputs
		if recordId is None:
			raise TypeError, "A record ID must be provided."
		
		self.collection.update({'_id':recordId}, {'$push':{'audit':audit}})
	
	
	def addData(self, recordId=None, signal=[], mdata={}, flag=True):
		"""
        To add synchronous data to a record.
		
        Parameters
        __________
        recordId : int
			ID of the record.
		signal : array
			Data to add.
			Default=[]
		mdata : dict
			JSON with metadata about the data.
			Default={}
        
        Returns
        _______
        kwrvals : dict
			A keyworded return values dict is returned with the following keys:
				'recordId' : int
					ID of the record.
				'dataName' : str
					Storge name of the data.
				'signalType' : str
					Type of the inserted data.
        
        See Also
        ________
        
        
        Notes
        _____
        
        
        Example
        _______
        
        
        References
        __________
        .. [1] 
        
        http://
        """	
		
		# check inputs
		if recordId is None:
			raise TypeError, "A record ID must be provided."
		
		# get record to add data to
		res = self.getById(recordId, restrict={'data':1, 'nbData':1, 'dataList':1})
		
		if res['doc'] is None:
			raise ValueError("No record found with given ID ({0}).".format(recordId))
		
		nb = res['doc']['nbData']
		
		# generate name
		try:
			dataName = mdata['name']
			# check if it is a protected name or if it exists
			check = dataName[:4] == 'data' and dataName[4:].isdigit()
			if check or self.nameChecker('data', res['doc'], recordId, dataName):
				dataName = 'data' + str(nb)
				mdata.update({'name': dataName})
		except KeyError:
			dataName = 'data' + str(nb)
			mdata.update({'name': dataName})
		
		# correctly format the type field in mdata
		try:
			weg = mdata['type']
			if weg is '':
				weg = '/'
			elif not weg[0] == '/':
				weg = '/' + weg
		except KeyError:
			weg = '/'
		
		mdata.update({'type': weg})
		
		# kwrvals
		kwrvals = {}
		kwrvals['recordId'] = recordId
		kwrvals['dataRef'] = dataName
		kwrvals['signalType'] = weg
		
		if flag:
			# add to HDF5 file
			name = self.name % recordId
			filePath = os.path.join(self.path, name)
			f = h5.hdf(filePath, mode='a')
			f.addData(signal, mdata, dataName)
			f.close()
		
		# update DB
		mdata.pop('type')
		if weg == '/':
			weg = 'data' + '.loc'
		else:
			weg = 'data' + weg.replace('/', '.') + '.loc'
		self.collection.update({'_id': recordId}, {'$push': {weg: mdata, 'dataList': dataName}, '$inc': {'nbData': 1}})
		
		return kwrvals
	
	
	def addEvent(self, recordId=None, timeStamps=[], values=[], mdata={}, flag=True):
		"""
        To add events (i.e. asynchronous data) to a record.
		
        Parameters
        __________
        recordId : int
			ID of the record.
		timeStamps : array
			Array of time stamps.
			Default=[]
		values : array
			Array with data for each time stamp.
			Default=[]
		mdata : dict
			JSON with metadata about the events.
			Default={}
        
        Returns
        _______
        kwrvals : dict
			A keyworded return values dict is returned with the following keys:
				'recordId' : int
					ID of the record.
				'eventName' : str
					Storge name of the events.
				'eventType' : str
					Type of the events.
        
        See Also
        ________
        
        
        Notes
        _____
        
        
        Example
        _______
        
        
        References
        __________
        .. [1] 
        
        http://
        """	
		
		# check inputs
		if recordId is None:
			raise TypeError, "A record ID must be provided."
		
		# get filename
		res = self.getById(recordId, restrict={'events':1, 'nbEvents':1, 'eventList':1})
		
		if res['doc'] is None:
			raise ValueError("No record found with given ID ({0}).".format(recordId))
		
		nb = res['doc']['nbEvents']
		
		# generate name
		try:
			eventName = mdata['name']
			# check if it is a protected name or if it exists
			check = eventName[:5] == 'event' and eventName[5:].isdigit()
			if check or self.nameChecker('event', res['doc'], recordId, eventName):
				eventName = 'event' + str(nb)
				mdata.update({'name': eventName})
		except KeyError:
			eventName = 'event' + str(nb)
			mdata.update({'name': eventName})
		
		# correctly format the type field in mdata
		try:
			weg = mdata['type']
			if weg is '':
				weg = '/'
			elif not weg[0] == '/':
				weg = '/' + weg
		except KeyError:
			weg = '/'
		
		mdata.update({'type': weg})
		
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
			f.addEvent(timeStamps, values, mdata, eventName)
			f.close()
		
		# update DB
		mdata.pop('type')
		if weg == '/':
			weg = 'events' + '.loc'
		else:
			weg = 'events' + weg.replace('/', '.') + '.loc'
		self.collection.update({'_id': recordId}, {'$push': {weg: mdata, 'eventList': eventName}, '$inc': {'nbEvents': 1}})
		
		return kwrvals
	
	
	def getByName(self, recordName=None, restrict={}):
		"""
        Query the 'records' collection by name.
		
        Parameters
        __________
        recordName : str
			Name of the record.
		restrict : dict
			Dictionary to restrict the information sent by the DB.
			Default={}
        
        Returns
        _______
        kwrvals : dict
			A keyworded return values dict is returned with the following keys:
				'doc' : dict
					The document with the results of the query. None if no match in found.
        
        See Also
        ________
        
        
        Notes
        _____
        
        
        Example
        _______
        
        
        References
        __________
        .. [1] 
        
        http://
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
		
        Parameters
        __________
        recordId : int
			ID of the record.
		restrict : dict
			Dictionary to restrict the information sent by the DB.
			Default={}
        
        Returns
        _______
        kwrvals : dict
			A keyworded return values dict is returned with the following keys:
				'doc' : dict
					The document with the results of the query. None if no match in found.
				
        See Also
        ________
        
        
        Notes
        _____
        
        
        Example
        _______
        
        
        References
        __________
        .. [1] 
        
        http://
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
		
        Parameters
        __________
		refine : dict
			Dictionary to refine the search.
			Default={}
		restrict : dict
			Dictionary to restrict the information sent by the DB.
			Default={}
        
        Returns
        _______
        kwrvals : dict
			A keyworded return values dict is returned with the following keys:
				'docList' : list
					The list of documents with the results of the query. Empty list if no match is found.
        
        See Also
        ________
        
        
        Notes
        _____
        
        
        Example
        _______
        get(refine={'age':100}, restrict={'email':1})
        
        References
        __________
        .. [1] 
        
        http://
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
	
	
	def getData(self, recordId=None, signalType='/', dataRef=None):
		"""
        Retrive synchronous data from a record.
		
        Parameters
        __________
        recordId : int
			ID of the record.
		signalType : str
			Type of the desired data.
			Default='/'
		dataRef : str, int
			Storge name or index of the desired data.
        
        Returns
        _______
        kwrvals : dict
			A keyworded return values dict is returned with the following keys:
				'signal' : array
					Array with the data.
				'mdata' : dict
					JSON with the data's accompanying metadata. 
        
        See Also
        ________
        
        
        Notes
        _____
        
        
        Example
        _______
        
        
        References
        __________
        .. [1] 
        
        http://
        """	
		
		# check inputs
		if recordId is None:
			raise TypeError, "A record ID must be provided."
		if dataRef is None:
			raise TypeError, "A data reference must be provided."
		
		# correctly format the type
		weg = signalType
		if weg == '':
			weg = '/'
		elif not weg[0] == '/':
			weg = '/' + weg
		
		if type(dataRef) is str:
			# dataRef is a name
			dataName = dataRef
		elif type(dataRef) is int:
			# dataRef is an index
			if weg == '/':
				wegM = 'data' + '.loc'
			else:
				wegM = 'data' + weg.replace('/', '.') + '.loc'
			res = self.getById(recordId, restrict={wegM:1})
			aux = res['doc']
			wegits = wegM.split('.')
			for item in wegits: aux = aux[item]
			dataName = aux[dataRef]['name']
		else:
			raise TypeError, "Input argument 'dataRef' must be of type str or int."
		
		# access the HDF5
		name = self.name % recordId
		filePath = os.path.join(self.path, name)
		f = h5.hdf(filePath, mode='r')
		kwrvals = f.getData(weg, dataName)
		f.close()
		
		return kwrvals
	
	
	def getEvent(self, recordId=None, eventType='/', eventRef=None):
		"""
        Retrieve events (i.e. asynchronous data) from a record.
		
        Parameters
        __________
        recordId : int
			ID of the record.
		eventType : str
			Type of the desired event.
			Default='/'
		eventRef : str, int
			Storage name or index of the desired events.
        
        Returns
        _______
        kwrvals : dict
			A keyworded return values dict is returned with the following keys:
				'timeStamps' : array
					Array of time stamps.
				'values' : array
					Array with data for each time stamp.
				'mdata' : dict
					JSON with metadata about the events.
        
        See Also
        ________
        
        
        Notes
        _____
        
        
        Example
        _______
        
        
        References
        __________
        .. [1] 
        
        http://
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
		
		if type(eventRef) is str:
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
		f = h5.hdf(filePath, mode='r')
		kwrvals = f.getEvent(weg, eventName)
		f.close()
		
		return kwrvals
	
	
	def getAll(self, restrict={}, count=-1, randomFlag=False):
		"""
        Generate a list of records present in the DB. The list may include all records, records pertaining to an experiment (or list of experiments), record pertaining to a subject (or list of subjects) or a combination of both experiments and subjects.
		
        Parameters
        __________
        restrict : dict
			To restrict the results. Can have the following keys:
				'experiment' : int, str, list
					Experiment ID, experiment name, list of experiment IDs, list of experiment names, or list of experiment names and IDs.
				'subject' : int, str, list
					Subject ID, subject name, list of subject IDs, list of subject names, or list of experiment names and IDs.
			Default={}
		count : int
			The resulting list has, at most, 'count' items. Set to -1 to output all records found.
			Default=-1
		randomFlag : boolean
			Set this flag to True to randomize the output list.
			Default=False
        
        Returns
        _______
        kwrvals : dict
			A keyworded return values dict is returned with the following keys:
				'idList' : list
					List with the records' IDs that match the search.
		
        See Also
        ________
        
        
        Notes
        _____
        
        
        Example
        _______
        
        
        References
        __________
        .. [1] 
        
        http://
        """	
		# get all the records' ID
		
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
			
			if type(exp) is str or type(exp) is int:
				exp = [exp]
			if type(sub) is str or type(sub) is int:
				sub = [sub]
				
			return exp, sub
		
		def worker1(self, tpe, thing):
			res = []
			for item in thing:
				if tpe == 'experiment':
					if type(item) is str:
						doc = self.experiments.getByName(item, restrict={'records': 1})['doc']
					else:
						doc = self.experiments.getById(item, restrict={'records': 1})['doc']
					if doc == None:
						continue
			
				elif tpe == 'subject':
					if type(item) is str:
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
	
	
	def getTypes(self, recordId=None, case=None):
		# to list the types of data/events
		
		# check inputs
		if recordId is None:
			raise TypeError, "A recordId must be provided."
		if case is None:
			raise TypeError, "A case ('data' or 'events') must be provided."
		
		# get the record
		res = self.getById(recordId, restrict={case: 1})
		if res['doc'] is None:
			raise ValueError, "No record found with given ID ({0}).".format(recordId)
		doc = res['doc']
		
		# helper
		def walkStruct(st, new):
			for key in st.iterkeys():
				if key == 'loc':
					pass
				else:
					new[key] = {}
					walkStruct(st[key], new[key])
		
		# walk the structure
		cSt = {}
		walkStruct({case: doc[case].copy()}, cSt)
		
		# kwrvals
		kwrvals = {}
		kwrvals['typeDict'] = cSt
		
		return kwrvals
	
	
	def typeCounter(self, recordId=None, case=None, sType=''):
		# to count the elements of data/events in given type
		
		# check inputs
		if recordId is None:
			raise TypeError, "A recordId must be provided."
		if case is None:
			raise TypeError, "A case ('data' or 'events') must be provided."
		
		# correctly format the type
		weg = sType
		if weg == '' or weg == '/':
			weg = case + '/loc'
		elif not weg[0] == '/':
			weg = case + '/' + weg + '/loc'
		else:
			weg = case + weg + '/loc'
		wegits = weg.split('/')
		
		# get the record
		res = self.getById(recordId, restrict={case: 1})
		if res['doc'] is None:
			raise ValueError, "No record found with given ID ({0}).".format(recordId)
		aux = res['doc']
		try:
			for item in wegits: aux = aux[item]
			count = len(aux)
		except KeyError:
			count = 0
		
		# kwrvals
		kwrvals = {}
		kwrvals['count'] = count
		
		return kwrvals
	
	
	def listTypes(self, recordId=None):
		"""
		
		To list the types of data and events.
		
		Kwargs:
			recordId (int): ID of the record.
		
		Kwrvals:
			signalTypes (list): List of data types.
			
			eventTypes (list): List of event types.
		
		See Also:
			
		
		Notes:
			
		
		Example:
			res = records.listTypes(0)
		
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
		res = self.getById(recordId, restrict={'data': 1, 'events': 1})
		if res['doc'] is None:
			raise ValueError, "No record found with given ID ({0}).".format(recordId)
		data = res['doc']['data']
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
		kwrvals['dataTypes'] = dataL
		kwrvals['eventTypes'] = eventsL
		
		return kwrvals
	
	
	def update(self, recordId=None, info={}):
		"""
        Update a record with the given information. Fiels can be added, and its type changed, but not deleted.
		
        Parameters
        __________
        recordId : int
			ID of the record to update.
		info : dict
			Dictionary with the information to add.
			Default={}
        
        Returns
        _______
        
        
        See Also
        ________
        
        
        Notes
        _____
        
        
        Example
        _______
        
        
        References
        __________
        .. [1] 
        
        http://
        """	
		
		# check inputs
		if recordId is None:
			raise TypeError, "A record ID must be provided."
		
		# don't update the following fields
		fields = ['experiment', 'subject', 'data', 'nbData', 'dataList', 'events', 'nbEvents', 'eventList']
		for item in fields:
			if info.has_key(item):
				info.pop(item)
		
		# update DB
		self.collection.update({'_id': recordId}, {'$set': info})
		
		# update HDF5
		name = self.name % recordId
		filePath = os.path.join(self.path, name)
		f = h5.hdf(filePath, mode='a')
		aux = f.getInfo()
		aux['header'].update(info)
		f.addInfo(**aux)
		f.close()
	
	
	def nameChecker(self, case=None, doc=None, recordId=None, name=None):
		# to check if name exists
		
		# check inputs
		if case is None:
			raise TypeError, "A case ('data' or 'event') must be provided."
		if recordId is None:
			raise TypeError, "A record ID must be provided."
		if name is None:
			raise TypeError, "A data or event name must be provided."
		
		# data or events?
		if case == 'data':
			nbStr = 'nbData'
			structStr = 'data'
			listStr = 'dataList'
		elif case == 'event' or case == 'events':
			nbStr = 'nbEvents'
			structStr = 'events'
			listStr = 'eventList'
		else:
			raise ValueError, "The case must either be 'data' or 'event'."
		
		if doc is None:
			# go to the DB again
			res = self.getById(recordId, restrict={nbStr:1, structStr:1, listStr:1})
			if res['doc'] is None:
				raise ValueError, "No record found with given ID ({0}).".format(recordId)
			doc = res['doc']
		
		# helper
		def walkStruct(st):
			for key in st.iterkeys():
				if key == 'loc':
					for item in st[key]: nameList.append(item['name'])
				else:
					walkStruct(st[key])
		
		# is list of names already created?
		try:
			nameList = doc[listStr]
		except KeyError:
			# we have to build it
			struct = doc[structStr]
			nameList = []
			walkStruct(struct)
			nameList.sort()
			
			# update DB
			self.collection.update({'_id': recordId}, {'$set': {listStr: nameList}})
		
		return (name in nameList)
	
	
	def h5Add(self, filePath=None):
		"""
		Procedure to add already created HDF5 files to mongoDB.
		
		Parameters
		__________
		filePath : str
			Location of the file to add.
		
		Returns
		_______
		
		
		See Also
		________
		
		Notes
		_____
		
		
		Example
		_______
		
		
		References
		__________
		.. [1] 
		
		http://
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
		f.data.visititems(filldict)
		for key in D.iterkeys():
			if isinstance(D[key], h5.h5py.Dataset):
				out = f.getDataInfo(key)
				# update the DB
				self.addData(recId, [], out['mdata'], flag=False)
		
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
			self.addEvent(recId, [], [], out['mdata'], flag=False)
		
		# close the file
		f.close()
	
	
	
	
	# def getTypeData(self, recordId=None, signalType=None):
	# """
	# Retrive synchronous data from a record by type.
	
	# Parameters
	# __________
	# recordId : int
		# ID of the record.
	# signalType : str
		# Type of the desired data.
	
	# Returns
	# _______
	# kwrvals : dict
		# A keyworded return values dict is returned with the following keys:
			# 'signal' : array
				# Array with the data.
			# 'mdata' : dict
				# JSON with the data's accompanying metadata.
		# If there is no signal with the desired type, kwrvals is an empty dictionary.
	
	# See Also
	# ________
	
	
	# Notes
	# _____
	
	
	# Example
	# _______
	
	
	# References
	# __________
	# .. [1] 
	
	# http://
	# """	
	
	# # check inputs
	# if recordId is None:
		# raise TypeError, "A record ID must be provided."
	# if signalType is None:
		# raise TypeError, "A signal type must be provided."
	
	# # get the document
	# res = self.getById(recordId, restrict={'data':1})
	
	# if res['doc'] is None:
		# raise ValueError("No record found with given ID.")
	
	# aux = res['doc']['data']
	
	# # parse the keys
	# flag = False
	# for key in aux.iterkeys():
		# if aux[key]['type'] == signalType:
			# flag = True
			# break
	
	# kwrvals = {}
	# if flag:
		# kwrvals = self.getData(recordId, key)
	
	# return kwrvals
	
	
	# def getTypeEvent(self, recordId=None, eventType=None):
	# """
	# Retrive asynchronous data (events) from a record by type.
	
	# Parameters
	# __________
	# recordId : int
		# ID of the record.
	# eventType : str
		# Type of the desired event.
	
	# Returns
	# _______
	# kwrvals : dict
		# A keyworded return values dict is returned with the following keys:
			# 'timeStamps' : array
				# Array of time stamps.
			# 'values' : array
				# Array with data for each time stamp.
			# 'mdata' : dict
				# JSON with metadata about the events.
		# If there is no event with the desired type, kwrvals is an empty dictionary.
	
	# See Also
	# ________
	
	
	# Notes
	# _____
	
	
	# Example
	# _______
	
	
	# References
	# __________
	# .. [1] 
	
	# http://
	# """	
	
	# # check inputs
	# if recordId is None:
		# raise TypeError, "A record ID must be provided."
	# if eventType is None:
		# raise TypeError, "An event type must be provided."
	
	# # get the document
	# res = self.getById(recordId, restrict={'events':1})
	
	# if res['doc'] is None:
		# raise ValueError("No record found with given ID.")
	
	# aux = res['doc']['events']
	
	# # parse the keys
	# flag = False
	# for key in aux.iterkeys():
		# if aux[key]['type'] == eventType:
			# flag = True
			# break
	
	# kwrvals = {}
	# if flag:
		# kwrvals = self.getEvent(recordId, key)
	
	# return kwrvals
	
	
	
	# def remove(self, recordId, flag=True):
		# """
		# Removes a record from the DB. It removes all data and events pertaining to the record and updates the DB structures accordingly.
		
		# Parameters
		# __________
		# recordId : ID of the record to remove.
		# flag : boolean
		# Flag to override user confirmation about removal. Set to False to avoid user prompt.
		# Default=True
		
		# Returns
		# _______
		
		
		# See Also
		# ________
		
		
		# Notes
		# _____
		
		
		# Example
		# _______
		
		
		# References
		# __________
		# .. [1] 
		
		# http://
		# """	
		
		# # get the doc to remove
		# doc = self.getById(recordId, restrict={'subject':1, 'experiment':1, 'nbData':1, 'nbEvents':1})
		
		# if doc != None:
			# experiment = doc['experiment']
			# if type(experiment) is unicode:
				# experimentId = self.experiments.getByName(experiment, restrict={'_id':1})['_id']
			# else:
				# experimentId = experiment
				# experiment = self.experiments.getById(experiment, restrict={'_name':1})['name']
			# subject = doc['subject']
			# if type(subject) is unicode:
				# sname = subject
				# subject = self.subjects.getByName(subject, restrict={'_id':1})['_id']
			
			# nbData = doc['nbData']
			# nbEvents = doc['nbEvents']
			
			# # make sure
			# sure = True
			# prompt = 'Are you sure you want to delete the Record %s from Subject %s, and belonging to Experiment %s?\nRemoved data cannot be restored!\n(y/n?)\n' % (str(recordId), str(sname) , str(experiment))
			
			# if flag:
				# var = raw_input(prompt)
				# if var != 'y' and var != 'Y':
					# sure = False
				
			# if sure:
				# # remove from experiments and subjects
				# self.experiments.delRecord(experimentId, subject, recordId)
				# self.subjects.delRecord(subject, experiment, recordId)
				
				# # remove all data and events
				# for i in range(nbData):
					# self.delData(recordId, i, False)
				# for i in range(nbEvents):
					# self.delEvent(recordId, i, False)
				
				# # remove document
				# self.collection.remove(recordId)


# def delData(self, recordId, index, flag=True):
	# """
	# Remove data from a record.
	
	# Parameters
	# __________
	# recordId : int
	# ID of the record.
	# index : int
	# Storage index of the desired data.
	# flag : boolean
	# Flag to override user confirmation about removal. Set to False to avoid user prompt.
	# Default=True
	
	# Returns
	# _______
	
	
	# See Also
	# ________
	
	
	# Notes
	# _____
	
	
	# Example
	# _______
	
	
	# References
	# __________
	# .. [1] 
	
	# http://
	# """	
	
	# # make sure
	# prompt = 'Are you sure you want to delete the Data with index %d from Record %s?\nRemoved data cannot be restored!\n(y/n?)\n' % (index, str(recordId))
	# sure = True
	# if flag:
		# var = raw_input(prompt)
		# if var != 'y' and var != 'Y':
			# sure = False
	
	# if sure:
		# # get tge doc
		# doc = self.getById(recordId, restrict={'data':1})
		
		# if doc == None:
			# raise ValueError("No record found with given ID.")
		
		# doc = doc['data']
		
		# # find desired data
		# mdata = None
		# for item in doc:
			# if item['index'] == index:
				# mdata = item
				# break
		
		# if mdata == None:
			# raise ValueError("Record does not have data with the given index.")
		
		# # remove from GridFS
		# stream = mdata.pop('stream')
		# self.fs.delete(stream['fid'])
		
		# # remove from data array and update count field
		# self.collection.update({'_id': recordId}, {'$pull': {'data': {'index': index}}, '$inc': {'nbData': -1}})
	


# def delEvent(self, recordId, index, flag=True):
	# """
	# Remove events from a record.
	
	# Parameters
	# __________
	# recordId : int
	# ID of the record.
	# index : int
	# Storage index of the desired event.
	# flag : boolean
	# Flag to override user confirmation about removal. Set to False to avoid user prompt.
	# Default=True
	
	# Returns
	# _______
	
	
	# See Also
	# ________
	
	
	# Notes
	# _____
	
	
	# Example
	# _______
	
	
	# References
	# __________
	# .. [1] 
	
	# http://
	# """	
	
	# # make sure
	# prompt = 'Are you sure you want to delete the Event with index %d from Record %s?\nRemoved data cannot be restored!\n(y/n?)\n' % (index, str(recordId))
	# sure = True
	# if flag:
		# var = raw_input(prompt)
		# if var != 'y' and var != 'Y':
			# sure = False
	
	# if sure:
		# # get the doc
		# doc = self.getById(recordId, restrict={'events':1})
		
		# if doc == None:
			# raise ValueError("No record found with given ID.")
		
		# doc = doc['events']
		
		# # find the desired data
		# mdata = None
		# for item in doc:
			# if item['index'] == index:
				# mdata = item
				# break
		
		# if mdata == None:
			# raise ValueError("Record does not have events with the given index.")
		
		# # remove from GridFS
		# ts = mdata.pop('timeStamps')
		# self.fs.delete(ts['fid'])
		# val = mdata.pop('values')
		# self.fs.delete(val['fid'])
		
		# # remove from data array and update count field
		# self.collection.update({'_id': recordId}, {'$pull': {'events': {'index': index}}, '$inc': {'nbData': -1}})

		
	


# def str2tuple(tupleString=None):
	# """
	# Convert tuple-like strings to real tuples, e.g. '(1,2,3,4)' -> (1, 2, 3, 4).
	
	# Parameters
	# __________
	# s : str
	# String with the tuple.
	
	# Returns
	# _______
	# : tuple
	# Resulting tuple from the conversion.
	
	# See Also
	# ________
	
	
	# Notes
	# _____
	
	
	# Example
	# _______
	
	
	# References
	# __________
	# .. [1] 
	
	# http://
	# """	
	
	# # check inputs
	# if tupleString is None:
		# raise TypeError, "A string must be provided."
	
	# if s[0] + s[-1] != "()":
		# raise ValueError("Badly formatted string (missing brackets).")
	# items = s[1:-1] # removes the leading and trailing brackets
	# items = items.split(',')
	# try:
		# L = [int(x.strip()) for x in items] # clean up spaces, convert to ints
	# except ValueError:
		# # for 1-sized tuples, e.g. (1,)
		# L = [int(items[0]), ]
	
	# # kwrvals
	# kwrvals = {}
	# kwrvals['tuple'] = tuple(L)
	
	# return kwrvals
	
if __name__ == '__main__':
	# Example
	import numpy as np
	import pylab as pl
	import datetime
	import time

	exp0 = {'name': 'exp1', 'description': 'This is a sample experiment.', 'goals': 'To illustrate an experiment document.'}

	pp0 = {'name': 'Mr. Sample Subject', 'age': 0, 'sex': 'm', 'email': 'sample@domain.com'}

	rec0 = {'name': 'rec0', 'date': datetime.datetime.utcnow().isoformat(), 'experiment': 'exp1', 'subject': 'Mr. Sample Subject', 'supervisor': 'Mr. Super'}

	# open connection to db
	config = {'dbName': 'biodb_test', 'host': '193.136.222.234', 'port': 27017}
	db = bioDB(**config)

	# collections
	exps = experiments(**db)
	subs = subjects(**db)
	recs = records(**db)

	# add experiment
	expId = exps.add(exp0)['experimentId']

	# add subject
	subId = subs.add(pp0)['subjectId']

	# add record
	recId = recs.add(rec0)['recordId']

	# add data to record
	Fs = 1024
	duration = 1
	t = np.arange(0, duration, 1./Fs)
	x = pl.cos(2*pl.pi*10*t)

	data = np.transpose(np.array([x, 2*x, 0.1*x]))
	mdata = {'type': '/EEG', 'labels': ['Ch.1', 'Ch.2', 'Ch.3'], 'device': {'channels': [0, 1, 2], 'name': 'Sample Device'}, 'transducer': 'Ag/AgCl electrodes', 'units': {'time': 'second', 'sensor': 'microVolt'}, 'sampleRate': Fs, 'resolution': 12, 'duration': duration}
	dataIn = recs.addData(recId, data, mdata)
	
	# add events to record
	step = 0.1
	dt = 'int16'
	nts = int(duration/step)
	timeStamps = []
	now = datetime.datetime.utcnow()
	for i in range(nts):
		instant = now + datetime.timedelta(seconds=i*step)
		timeStamps.append(time.mktime(instant.timetuple()))
	timeStamps = np.array(timeStamps, dtype='float')

	values = np.zeros([nts, 2], dt)
	mdata = {'source': 'eventGenerator', 'type': '/2D array', 'dictionary':{'0':0, '1':1, '2':2, '3':3}, 'eventSync': now.isoformat()}
	
	eventIn = recs.addEvent(recId, timeStamps, values, mdata)
	
	# get data
	dataOut = recs.getData(**dataIn)
	
	# close connection
	closeDB(**db)
	
	pl.plot(t, dataOut['signal'])
	pl.show()
	
	
