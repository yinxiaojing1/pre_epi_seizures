"""
.. module:: example
   :platform: Unix, Windows
   :synopsis: Example/tutorial of the BioMESH API.
   
.. modeuleauthor:: Carlos Carreiras
"""


import biomesh

if __name__ == '__main__':
	
	# Connect to the database
	# The directory dstPath/Databases/dbName is created if it does not exist
	# The 'sync' flag can be set to False if the intent is only to access the database (no insertions or updates)
	db = biomesh.biomesh(dbName='biomesh_tst', host='193.136.222.234', dstPath='~/tmp/biomesh', sync=True)
	
	
	
	# Adding a subject (fields according to specification)
	subject = {'name': 'Subject', 'birthdate': '2012-01-01T00:00:00.0', 'gender': 'm', 'email': 'subject@domain.net'}
	subId = db.subjects.add(subject)['subjectId']
	
	# Adding an experiment (fields according to specification)
	experiment = {'name': 'Experiment', 'description': 'Description of the experiment.', 'goals': 'Goals of the experiment.'}
	expId = db.experiments.add(experiment)['experimentId']
	
	# Adding a record (fields according to specification)
	record = {'name': 'Record', 'date': '2012-01-01T00:00:00.0', 'experiment': 'Experiment', 'subject': subId, 'supervisor': 'Supervisor'}
	recId = db.records.add(record)['recordId']
	
	
	
	# Querying subjects, experiments or records (exemplified for subjects)
	# Single document
	doc = db.subjects.getByName('Name')['doc']
	doc = db.subjects.getById(subId)['doc']
	# It is possible to limit the returned information by setting the 'restrict' keyword input
	doc = db.subjects.getById(subId, restrict={'_id': 1})['doc'] # only return the ID
	doc = db.subjects.getById(subId, restrict={'name': 1})['doc'] # only return the name (ID is also returned)
	doc = db.subjects.getById(subId, restrict={'_id': 0, 'name': 1})['doc'] # only return the name and don't return the ID
	
	# Multiple documents
	docList = db.subjects.get()['docList'] # returns all subjects
	# Specific queries can be made be setting the 'refine' keyword input
	docList = db.subjects.get(refine={'field': 'val', 'flag': False})['docList'] # returns all subjects with 'field' set to 'val' AND 'flag' set to False
	# The 'restrict' keyword can also be set
	docList = db.subjects.get(refine={'field': 'val'}, restrict={'flag': 1})['docList']
	
	# Special query for records (returns a list of IDs)
	# get all records
	idList = db.records.getAll()['idList']
	# get at most 5 records
	idList = db.records.getAll(count=5)['idList']
	# get all records from experiment 'Experiment'
	idList = db.records.getAll(restrict={'experiment': 'Experiment'})['idList']
	# get all records from experiment 'Experiment' and 'exp'
	idList = db.records.getAll(restrict={'experiment': ['Experiment', 'exp']})['idList']
	# get all records from subject 0
	idList = db.records.getAll(restrict={'subject': 0})['idList']
	# get all records from subject 0 and 1
	idList = db.records.getAll(restrict={'subject': [0, 1]})['idList']
	# get all records from subject 0 and 1 and 'Record'
	idList = db.records.getAll(restrict={'subject': [0, 1, 'Record']})['idList']
	# get all records from experiment 'Experiment', BUT only from subjects 0 and 1
	idList = db.records.getAll(restrict={'experiment': 'exp', 'subject': [0, 1]})['idList']
	
	
	
	# Update a subject (by ID)
	db.subjects.update(subId, {'new': 'field'})
	
	# Update an experiment (by Name)
	db.experiments.update('Experiment', {'new': 'field'})
	
	# Update a record (by ID)
	db.records.update(recId, {'new': 'field'})
	
	
	
	# Adding a signal to a record
	signalType = '/test'
	signal = [0, 1, 2, 3]
	metadata = {'name': 'testSignal',
                'labels': ['signal'],
                'device': {'name': 'Device', 'channels': [0]},
                'transducer': 'electrodes',
                'units': {'time': 'seconds', 'sensor': 'volt'},
                'sampleRate': 1000.,
                'duration': 1.,
                'resolution': 12}
	db.records.addSignal(recId, signalType, signal, metadata)
	
	# Retrieving a signal from a record
	res = db.records.getSignal(recId, '/test', 0) # by index
	res = db.records.getSignal(recId, '/test', 'testSignal') # by name
	signal = res['signal']
	metadata = res['mdata']
	
	
	
	# Adding an event sequence to a record
	eventType = '/test'
	timeStamps = [0, 1, 2, 3]
	values = ['a', 'b', 'c', 'a']
	metadata = {'name': 'testEvent',
                'source': 'eventsGenerator',
                'eventSync': 0.,
                'dictionary': {'a': 'Event a', 'b': 'Event b', 'c': 'Event c'}}
	db.records.addEvent(recId, eventType, timeStamps, values, metadata)
	
	# Retrieving an event sequence from a record
	res = db.records.getEvent(recId, '/test', 0) # by index
	res = db.records.getEvent(recId, '/test', 'testEvent') # by name
	timeStamps = res['timeStamps']
	values = res['values']
	metadata = res['mdata']
	
	
	
	# Operators
	# __getitem__
	rec = db.records[0] # get record 0
	recs = db.records[:] # get a slice
	recs = db.records[[0, 4, 5]] # get records 0, 4 and 5
	
	# __len__
	nbRecs = len(db.records)
	
	# __iter__
	for item in db.records:
		print item
	
	# dataSelector and dataContainer classes
	# Signals
	sel = db.records[recId]['signals']['test']
	sel = db.records[recId]['signals/test']
	sel.list() # lists the contents of the sub-type
	nb = len(sel) # number of datasets on the sub-type
	
	data = sel[0] # can also be a slice, or a list of indices
	signal = data.signal
	metadata = data.metadata
	
	# __iter__
	for item in sel:
		print item.metadata['name']
	
	# Events
	sel = db.records[recId]['events']['test']
	sel = db.records[recId]['events/test']
	sel.list() # lists the contents of the sub-type
	nb = len(sel) # number of datasets on the sub-type
	
	data = sel[0] # can also be a slice, or a list of indices
	timeStamps = data.timeStamps
	values = data.values
	metadata = data.metadata
	
	# __iter__
	for item in sel:
		print item.metadata['name']
	
	# Do not forget to close the connection when finished, specially when there where insertions or updates (and the 'sync' flag is True)
	db.close()
