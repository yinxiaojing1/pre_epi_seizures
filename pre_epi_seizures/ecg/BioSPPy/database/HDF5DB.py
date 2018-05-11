"""
.. module:: HDF5DB
   :platform: Unix, Windows
   :synopsis: This module provides a wrapper to the HDF5 file format, adapting it to store biosignals according to the BioMESH specification at http://camoes.lx.it.pt/MediaWiki/index.php/Database_Specification#Data_Model

.. moduleauthor:: Carlos Carreiras
"""


import h5py, json


class hdf:
	"""
	
	Wrapper class to operate on HDF5 records according to the BioMESH specification.
	
	Kwargs:
		
	
	Kwrvals:
		
	
	See Also:
		
	
	Notes:
		
	
	Example:
		
	
	References:
		.. [1]
		
	"""
	
	
	def __init__(self, filePath=None, mode='a'):
		"""
		
		Open the HDF5 record.
		
		Kwargs:
			filePath (str): Path to HDF5 file.
			
			mode (str): File access mode. Available modes:
				'r+': Read/write, file must exist
				'r': Read only, file must exist
				'w': Create file, truncate if exists
				'w-': Create file, fail if exists
				'a': Read/write if exists, create otherwise
				Default: 'a'.
		
		Kwrvals:
			
		
		See Also:
			
		
		Notes:
			
		
		Example:
			fid = hdf('record.hdf5', 'a')
		
		References:
			.. [1]
			
		"""
		
		# check inputs
		if filePath is None:
			raise TypeError, "A path to the HDF5 file is needed."
		
		# open the file
		self.file = h5py.File(filePath, mode)
		# check the basic structures
		try:
			self.data = self.file['data']
		except KeyError:
			if mode == 'r':
				raise IOError("File is in read only mode and doesn't have the required Group 'data'; change to another mode.")
			self.data = self.file.create_group('data')
		try:
			self.events = self.file['events']
		except KeyError:
			if mode == 'r':
				raise IOError("File is in read only mode and doesn't have the required Group 'events'; change to another mode.")
			self.events = self.file.create_group('events')
	
	
	def addInfo(self, header={}):
		"""
		
		Method to add or overwrite the basic information (header) of the HDF5 record.
		
		Kwargs:
			header (dict): Dictionary (JSON) object with the information. Default: {}.
		
		Kwrvals:
			
		
		See Also:
			
		
		Notes:
			
		
		Example:
			fid.addInfo({'name': 'record'})
		
		References:
			.. [1]
			
		"""
		
		# add the information
		self.file.attrs['json'] = json.dumps(header)
		
		
	def getInfo(self):
		"""
		
		Method to retrieve the basic information (header) of the HDF5 record.
		
		Kwargs:
			
		
		Kwrvals:
			header (dict): Dictionary object with the header information.
		
		See Also:
			
		
		Notes:
			
		
		Example:
			header = fid.get()['header']
		
		References:
			.. [1]
			
		"""
		
		# kwrvals
		kwrvals = {'header': {}}
		
		try:
			kwrvals['header'] = json.loads(self.file.attrs['json'])
		except KeyError:
			pass
		
		return kwrvals
	
	def addData(self, signal=[], mdata={}, dataName=None):
		"""
		
		Method to add synchronous data to the HDF5 record.
		
		Kwargs:
			signal (array): Array with the data to add. Default: [].
			
			mdata (dict): Dictionary object with metadata about the data. Default: {}.
			
			dataName (str): Name of the dataset to be created.
		
		Kwrvals:
			
		
		See Also:
			
		
		Notes:
			
		
		Example:
			fid.addData([0, 1, 2], {'type': '/test'}, 'data0')
		
		References:
			.. [1]
			
		"""
		
		# check inputs
		if dataName is None:
			raise TypeError, "A name for the dataset must be specified."
		
		# navigate to group
		try:
			weg = self.data.name + mdata['type']
		except KeyError:
			weg = self.data.name
		
		try:
			group = self.file[weg]
		except KeyError:
			# create group
			group = self.file.create_group(weg)
		
		# create new dataset in group
		dset = group.create_dataset(dataName, data=signal)
		
		# set the attributes
		dset.attrs['json'] = json.dumps(mdata)
		
		return None
		
	def getData(self, signalType='', dataName=None):
		"""
		
		Method to retrieve synchronous data from the HDF5 record.
		
		Kwargs:
			signalType (str): Type of the desired data. Default: ''.
			
			dataName (str): Name of the dataset to retrieve.
		
		Kwrvals:
			signal (array): Array with the data.
			
			mdata (dict): Dictionary object with metadata about the data.
		
		See Also:
			
		
		Notes:
			
		
		Example:
			out = fid.getData('/test', 'data0')
			signal = out['signal']
			metadata = out['mdata']
		
		References:
			.. [1]
			
		"""
		
		# check inputs
		if dataName is None:
			raise TypeError, "A data name must be specified."
		
		if signalType == '':
			group = self.data
		else:
			# navigate to group
			weg = self.data.name + signalType
			group = self.file[weg]
		
		# get the data and mdata
		dset = group[dataName]
		try:
			data = dset[...]
		except ValueError:
			data = []
		mdata = json.loads(dset.attrs['json'])
		
		# kwrvals
		kwrvals = {}
		kwrvals['signal'] = data
		kwrvals['mdata'] = mdata
		
		return kwrvals
	
	
	def delData(self, signalType='', dataName=None):
		"""
		
		Method to delete synchronous data from the HDF5 record. The record is marked for repackaging.
		
		Kwargs:
			signalType (str): Type of the desired data. Default: ''.
			
			dataName (str): Name of the dataset to retrieve.
		
		Kwrvals:
			
		
		See Also:
			
		
		Notes:
			
		
		Example:
			fid.delData('/test', 'data0')
		
		References:
			.. [1]
			
		"""
		
		# check inputs
		if dataName is None:
			raise TypeError, "A data name must be specified."
		
		if signalType == '':
			group = self.data
		else:
			# navigate to group
			weg = self.data.name + signalType
			group = self.file[weg]
		
		try:
			del group[dataName]
		except ValueError, KeyError:
			pass
		
		# set to repack
		self.setRepack()
	
	
	def delDataType(self, signalType=''):
		"""
		
		Method to delete a type of synchronous data from the HDF5 record. The record is marked for repackaging.
		
		Kwargs:
			signalType (str): Type of the desired data. Default: ''.
		
		Kwrvals:
			
		
		See Also:
			
		
		Notes:
			
		
		Example:
			fid.delDataType('/test')
		
		References:
			.. [1]
			
		"""
		
		try:
			try:
				del self.data['./' + signalType]
			except KeyError, e:
				print e
		except ValueError, e:
			print e
		
		# set to repack
		self.setRepack()
	
	
	def getDataInfo(self, dataWeg=None):
		"""
		
		Method to retrieve the metadata of synchronous.
		
		Kwargs:
			dataWeg (str): Path to the dataset.
		
		Kwrvals:
			mdata (dict): Dictionary with the desired information.
		
		See Also:
			
		
		Notes:
			
		
		Example:
			out = fid.getDataInfo('/test/data0')
			metadata = out['mdata']
		
		References:
			.. [1]
			
		"""
		
		# check inputs
		if dataWeg is None:
			raise TypeError, "A data path must be specified."
		
		# get the metadata
		dset = self.data[dataWeg]
		mdata = json.loads(dset.attrs['json'])
		
		# kwrvals
		kwrvals = {}
		kwrvals['mdata'] = mdata
		
		return kwrvals
		
		
	def addEvent(self, timeStamps=[], values=[], mdata={}, eventName=None):
		"""
		
		Method to add asynchronous data (events) to the HDF5 record.
		
		Kwargs:
			timeStamps (array): Array of time stamps. Default: [].
			
			values (array): Array with data for each time stamp. Default: [].
			
			mdata (dict): Dictionary object with metadata about the events. Default: {}.
			
			eventName (str): Name of the group to be created.
		
		Kwrvals:
			
		
		See Also:
			
		
		Notes:
			
		
		Example:
			fid.addEvent([0, 1, 2], [[1, 2], [3, 4], [5, 6]], {'type': '/test'}, 'event0')
		
		References:
			.. [1]
			
		"""
		
		# check inputs
		if eventName is None:
			raise TypeError, "A name for the event must be specified."
		
		# navigate to group
		try:
			weg = self.events.name + mdata['type']
		except KeyError:
			weg = self.events.name
		
		try:
			parentGr = self.file[weg]
		except KeyError:
			# create group
			parentGr = self.file.create_group(weg)
		
		# create new group in parentGr
		group = parentGr.create_group(eventName)
		
		# add the timeStamps and values
		ts = group.create_dataset('timeStamps', data=timeStamps)
		val = group.create_dataset('values', data=values)
		
		# set the attributes
		group.attrs['json'] = json.dumps(mdata)
		
		return None
	
	
	def getEvent(self, eventType='', eventName=None):
		"""
		
		Method to retrieve asynchronous data(events) from the HDF5 record.
		
		Kwargs:
			eventType (str): Type of the desired event. Default: ''.
			
			eventName (str): Name of the dataset to retrieve.
		
		Kwrvals:
			timeStamps (array): Array of time stamps.
			
			values (array): Array with data for each time stamp.
			
			mdata (dict): Dictionary object with metadata about the events.
		
		See Also:
			
		
		Notes:
			
		
		Example:
			out = fid.getEvent('/test', 'event0')
			timeStamps = out['timeStamps']
			values = out['values']
			metadata = out['mdata']
		
		References:
			.. [1]
			
		"""
		
		# check inputs
		if eventName is None:
			raise TypeError, "An event name must be specified."
		
		if eventType == '':
			parentGr = self.events
		else:
			# navigate to group
			weg = self.events.name + eventType
			parentGr = self.file[weg]
		
		# get the things
		group = parentGr[eventName]
		mdata = json.loads(group.attrs['json'])
		timeStamps = group['timeStamps']
		try:
			timeStamps = timeStamps[...]
		except ValueError:
			timeStamps = []
		values = group['values']
		try:
			values = values[...]
		except ValueError:
			values = []
		
		# kwrvals
		kwrvals = {}
		kwrvals['timeStamps'] = timeStamps
		kwrvals['values'] = values
		kwrvals['mdata'] = mdata
		
		return kwrvals
	
	
	def delEvent(self, eventType='', eventName=None):
		"""
		
		Method to delete asynchronous data (events) from the HDF5 record. The record is marked for repackaging.
		
		Kwargs:
			eventType (str): Type of the desired event. Default: ''.
			
			eventName (str): Name of the dataset to retrieve.
		
		Kwrvals:
			
		
		See Also:
			
		
		Notes:
			
		
		Example:
			fid.delEvent('/test', 'event0')
		
		References:
			.. [1]
			
		"""
		
		# check inputs
		if eventName is None:
			raise TypeError, "An event name must be specified."
		
		if eventType == '':
			parentGr = self.events
		else:
			# navigate to group
			weg = self.events.name + eventType
			parentGr = self.file[weg]
		
		try:
			del parentGr[eventName]
		except ValueError, KeyError:
			pass
		
		# set to repack
		self.setRepack()
	
	
	def delEventType(self, eventType=''):
		"""
		
		Method to delete a type of asynchronous data from the HDF5 record. The record is marked for repackaging.
		
		Kwargs:
			eventType (str): Type of the desired event. Default: ''.
		
		Kwrvals:
			
		
		See Also:
			
		
		Notes:
			
		
		Example:
			fid.delEventType('/test')
		
		References:
			.. [1]
			
		"""
		
		try:
			try:
				del self.events['./' + eventType]
			except KeyError, e:
				print e
		except ValueError, e:
			print e
		
		# set to repack
		self.setRepack()
	
	
	def getEventInfo(self, eventWeg=None):
		"""
		
		Method to retrieve the metadata of asynchronous.
		
		Kwargs:
			eventWeg (str): Path to the group.
		
		Kwrvals:
			mdata (dict): Dictionary with the desired information.
		
		See Also:
			
		
		Notes:
			
		
		Example:
			out = fid.getEventInfo('/test/event0')
			metadata = out['mdata']
		
		References:
			.. [1]
			
		"""
		
		# check inputs
		if eventWeg is None:
			raise TypeError, "An event path must be specified."
		
		group = self.events[eventWeg]
		mdata = json.loads(group.attrs['json'])
		
		# kwrvals
		kwrvals = {}
		kwrvals['mdata'] = mdata
		
		return kwrvals
	
	
	def setRepack(self):
		"""
		
		Set flag to repack.
		
		Kwargs:
			
		
		Kwrvals:
			
		
		See Also:
			
		
		Notes:
			
		
		Example:
			fid.setRepack()
		
		References:
			.. [1]
			
		"""
		
		# set the flag
		self.file.attrs['repack'] = True
	
	
	def unsetRepack(self):
		"""
		
		Unset flag to repack.
		
		Kwargs:
			
		
		Kwrvals:
			
		
		See Also:
			
		
		Notes:
			
		
		Example:
			fid.unsetRepack()
		
		References:
			.. [1]
			
		"""
		
		# set the flag
		self.file.attrs['repack'] = False
	
	
	def getRepack(self):
		"""
		
		Get the repack flag.
		
		Kwargs:
			
		
		Kwrvals:
			
		
		See Also:
			
		
		Notes:
			
		
		Example:
			repack = fid.getRepack()
		
		References:
			.. [1]
			
		"""
		
		return self.file.attrs['repack']
	
	
	def close(self):
		"""
		
		Method to close the HDF5 record.
		
		Kwargs:
			
		
		Kwrvals:
			
		
		See Also:
			
		
		Notes:
			
		
		Example:
			fid.close()
		
		References:
			.. [1]
			
		"""
		
		# close the file
		self.file.close()
	
	
class meta:
	"""
	
	Wrapper class to store experiments and subjects on HDF5.
	
	Kwargs:
		
	
	Kwrvals:
		
	
	See Also:
		
	
	Notes:
		
	
	Example:
		
	
	References:
		.. [1]
		
	"""
	
	
	def __init__(self, filePath=None, mode='a'):
		"""
		
		Open the HDF5 file.
		
		Kwargs:
			filePath (str): Path to HDF5 file.
			
			mode (str): File access mode. Available modes:
				'r+': Read/write, file must exist
				'r': Read only, file must exist
				'w': Create file, truncate if exists
				'w-': Create file, fail if exists
				'a': Read/write if exists, create otherwise
				Default: 'a'.
		
		Kwrvals:
			
		
		See Also:
			
		
		Notes:
			
		
		Example:
			fid = meta('expsub.hdf5', 'a')
		
		References:
			.. [1]
			
		"""
		
		# check inputs
		if filePath is None:
			raise TypeError, "A path to the HDF5 file is needed."
			
		# open the file
		self.file = h5py.File(filePath, mode)
		
		# check the basic structures
		try:
			self.experiments = self.file['experiments']
		except KeyError:
			if mode == 'r':
				raise IOError, "File is in read only mode and doesn't have the required Group 'experiments'; change to another mode."
			self.experiments = self.file.create_group('experiments')
		
		try:
			self.subjects = self.file['subjects']
		except KeyError:
			if mode == 'r':
				raise IOError("File is in read only mode and doesn't have the required Group 'subjects'; change to another mode.")
			self.subjects = self.file.create_group('subjects')
		try:
			self.file.attrs['repack']
		except KeyError:
			if mode == 'r':
				raise IOError("File is in read only mode and doesn't have the required flag 'repack'; change to another mode.")
			self.file.attrs['repack'] = False
	
	
	def setDB(self, dbName=None):
		"""
		
		Method to set the DB the file belongs to.
		
		Kwargs:
			dbName (str): Name of the database.
		
		Kwrvals:
			
		
		See Also:
			
		
		Notes:
			
		
		Example:
			fid.setDB('database')
		
		References:
			.. [1]
			
		"""
		
		# check inputs
		if dbName is None:
			raise TypeError, "A DB name must be provided."
		
		# set
		self.file.attrs['dbName'] = dbName
	
	
	def addSubject(self, subject={}):
		"""
		
		Method to add a subject to the file.
		
		Kwargs:
			subject (dict): Dictionary with the subject information. Default: {}.
		
		Kwrvals:
			
		
		See Also:
			
		
		Notes:
			The subject must have a '_id' key.
		
		Example:
			fid.addSubject({'_id': 0, 'name': 'subject'})
		
		References:
			.. [1]
			
		"""
		
		# get the ID
		subjectId = subject['_id']
		
		# add the information
		dset = self.subjects.create_dataset(str(subjectId), data=json.dumps(subject))
	
	
	def getSubject(self, subjectId=None):
		"""
		
		Method to get the information about a subject.
		
		Kwargs:
			subjectId (int): The ID of the subject.
		
		Kwrvals:
			subject (dict): Dictionary with the subject information.
		
		See Also:
			
		
		Notes:
			
		
		Example:
			subject = fid.getSubject(0)['subject']
		
		References:
			.. [1]
			
		"""
		
		# check inputs
		if subjectId is None:
			raise TypeError, "A subject ID must be provided."
		
		try:
			aux = json.loads(str(self.subjects[str(subjectId)][...]))
		except KeyError:
			aux = {}
		
		# kwrvals
		kwrvals = {}
		kwrvals['subject'] = aux
		
		return kwrvals
	
	
	def updateSubject(self, subjectId=None, info={}):
		"""
		
		Method to update a subject's information.
		
		Kwargs:
			sunjectId (int): The ID of the subject.
			
			info (dict): Dictionary with the information to update.
		
		Kwrvals:
			
		
		See Also:
			
		
		Notes:
			
		
		Example:
			fid.updateSubject(0, {'new': 'field'})
		
		References:
			.. [1]
			
		"""
		
		# check inputs
		if subjectId is None:
			raise TypeError, "A subject ID must be provided."
		
		# get old info
		sub = self.getSubject(subjectId)['subject']
		del self.subjects[str(subjectId)]
		
		# update with new info
		sub.update(info)
		
		# store
		self.addSubject(sub)
	
	
	def listSubjects(self):
		"""
		
		Method to list all the subjects in the file.
		
		Kwargs:
			
		
		Kwrvals:
			subList (list): List with the subjects.
		
		See Also:
			
		
		Notes:
			
		
		Example:
			subList = fid.listSubjects()['subList']
		
		References:
			.. [1]
			
		"""
		
		subList = []
		for item in self.subjects.iteritems():
			subList.append(self.getSubject(item[0])['subject'])
		
		return {'subList': subList}
	
	
	def addExperiment(self, experiment={}):
		"""
		
		Method to add an experiment to the file.
		
		Kwargs:
			experiment (dict): Dictionary with the experiment information. Default: {}.
		
		Kwrvals:
			
		
		See Also:
			
		
		Notes:
			The experiment must have a 'name' key.
		
		Example:
			fid.addExperiment({'name': 'experiment', 'comments': 'Hello world.'})
		
		References:
			.. [1]
			
		"""
		
		# get the ID
		experimentName = experiment['name']
		
		# add the information
		dset = self.experiments.create_dataset(str(experimentName), data=json.dumps(experiment))
	
	
	def getExperiment(self, experimentName=None):
		"""
		
		Method to get the information about an experiment.
		
		Kwargs:
			experimentName (str): The name of the experiment.
		
		Kwrvals:
			experiment (dict): Dictionary with the experiment information.
		
		See Also:
			
		
		Notes:
			
		
		Example:
			experiment = fid.getExperiment('experiment')['experiment']
		
		References:
			.. [1]
			
		"""
		
		# check inputs
		if experimentName is None:
			raise TypeError, "An experiment name must be provided."
		
		try:
			aux = json.loads(str(self.experiments[str(experimentName)][...]))
		except KeyError:
			aux = {}
		
		# kwrvals
		kwrvals = {}
		kwrvals['experiment'] = aux
		
		return kwrvals
	
	
	def updateExperiment(self, experimentName=None, info={}):
		"""
		
		Method to update an experiment's information.
		
		Kwargs:
			experimentName (str): Name of the experiment.
			
			info (dict): Dictionary with the information to update.
		
		Kwrvals:
			
		
		See Also:
			
		
		Notes:
			
		
		Example:
			fid.updateExperiment('experiment', {'new': 'field'})
		
		References:
			.. [1]
			
		"""
		
		# check inputs
		if experimentName is None:
			raise TypeError, "An experiment name must be provided."
		
		# get old info
		exp = self.getExperiment(experimentName)['experiment']
		del self.experiments[str(experimentName)]
		
		# update with new info
		exp.update(info)
		
		# store
		self.addExperiment(exp)
	
	
	def listExperiments(self):
		"""
		
		Method to list all the experiments in the file.
		
		Kwargs:
			
		
		Kwrvals:
			expList (list): List with the experiments.
		
		See Also:
			
		
		Notes:
			
		
		Example:
			expList = fid.listExperiments()['expList']
		
		References:
			.. [1]
			
		"""
		
		expList = []
		for item in self.experiments.iteritems():
			expList.append(self.getExperiment(item[0])['experiment'])
		
		return {'expList': expList}
	
	
	def close(self):
		"""
		
		Method to close the HDF5 file.
		
		Kwargs:
			
		
		Kwrvals:
			
		
		See Also:
			
		
		Notes:
			
		
		Example:
			fid.close()
		
		References:
			.. [1]
			
		"""
		
		# close the file
		self.file.close()



if __name__ == '__main__':
	# Example
	import numpy as np
	import datetime
	import time
	import os
	
	# create directory for tests
	path = os.path.abspath(os.path.expanduser('~/tmp/HDF5DB'))
	if not os.path.exists(path):
		os.makedirs(path)
	
	# open the test record
	fid = hdf(os.path.join(path, 'rec.hdf5'), 'w')
	
	# addInfo
	header = {'name': 'rec', 'date': datetime.datetime.utcnow().isoformat(), 'experiment': 'experiment', 'subject': 'subject'}
	fid.addInfo(header)
	
	# getInfo
	header_ = fid.getInfo()['header']
	
	print "Header OK?", header == header_
	
	# addData
	signal = np.zeros((100, 5), dtype='float64')
	mdata = {'type': '/test', 'comments': 'zeros', 'name': 'signal'}
	dataName = 'signal'
	fid.addData(signal, mdata, dataName)
	
	# getData
	res = fid.getData('/test', 'signal')
	signal_ = res['signal']
	mdata_ = res['mdata']
	
	print "Data OK?"
	aux = signal == signal_
	print "  Signal: ", bool(np.prod(aux.flatten()))
	print "  Metadata: ", mdata == mdata_
	
	# delData
	mdata = {'type': '/test', 'comments': 'zeros', 'name': 'signalD'}
	dataName = 'signalD'
	fid.addData(signal, mdata, dataName)
	fid.delData('/test', dataName)
	
	try:
		res = fid.getData('/test', 'signalD')
		print "Delete data OK?", False
	except Exception, e:
		print "Delete data OK?", e
	
	# addEvent
	nts = 100
	timeStamps = []
	now = datetime.datetime.utcnow()
	for i in range(nts):
		instant = now + datetime.timedelta(seconds=i)
		timeStamps.append(time.mktime(instant.timetuple()))
	timeStamps = np.array(timeStamps, dtype='float')
	values = np.zeros((nts, 1), dtype='float64')
	mdata = {'type': '/test', 'comments': 'zeros', 'name': 'event'}
	eventName = 'event'
	fid.addEvent(timeStamps, values, mdata, eventName)
	
	# getEvent
	res = fid.getEvent('/test', 'event')
	timeStamps_ = res['timeStamps']
	values_ = res['values']
	mdata_ = res['mdata']
	
	print "Events OK?"
	aux = timeStamps == timeStamps_
	print "  Timestamps: ", bool(np.prod(aux.flatten()))
	aux = values == values_
	print "  Values: ", bool(np.prod(aux.flatten()))
	print "  Metadata: ", mdata == mdata_
	
	# delEvent
	mdata = {'type': '/test', 'comments': 'zeros', 'name': 'eventD'}
	eventName = 'eventD'
	fid.addEvent(timeStamps, values, mdata, eventName)
	fid.delEvent('/test', eventName)
	
	try:
		res = fid.getEvent('/test', 'eventD')
		print "Delete event OK?", False
	except Exception, e:
		print "Delete event OK?", e
	
	# close
	fid.close()
	
	
	# open the test metafile
	fid = meta(os.path.join(path, 'ExpSub.hdf5'), 'w')
	
	# addSubject
	subject = {'name': 'subject', '_id': 0}
	fid.addSubject(subject)
	
	# getSubject
	subject_ = fid.getSubject(0)['subject']
	
	print "Subject OK?", subject == subject_
	
	# updateSubject
	fid.updateSubject(0, {'new': 'field'})
	subject_ = fid.getSubject(0)['subject']
	
	print "Updade subject OK?", subject_['new'] == 'field'
	
	# listSubjects
	res = fid.listSubjects()
	
	print "List subjects OK?", res['subList']
	
	# addExperiment
	experiment = {'name': 'experiment', '_id': 0}
	fid.addExperiment(experiment)
	
	# getExperiment
	experiment_ = fid.getExperiment('experiment')['experiment']
	
	print "Experiment OK?", experiment == experiment_
	
	# updateSubject
	fid.updateExperiment('experiment', {'new': 'field'})
	experiment_ = fid.getExperiment('experiment')['experiment']
	
	print "Updade experiment OK?", experiment_['new'] == 'field'
	
	# listSubjects
	res = fid.listExperiments()
	
	print "List experiments OK?", res['expList']
	
	# close
	fid.close()
	
