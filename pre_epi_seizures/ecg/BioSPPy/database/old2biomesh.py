"""
.. module:: old2biomesh
   :platform: Unix, Windows
   :synopsis: To convert a database from the old API to the new BioMESH DB.
   
.. modeuleauthor:: Carlos Carreiras
"""


import os
import string
import mongoH5 as mongo
import biomesh


def dry_run(dbName, path):
	"""
	
	Dry run of the conversion process; checks for errors on the source database.
	
	Kwargs:
		dbName (str): Name of the database.
		
		path (str): Path to the HDF5 files.
	
	Kwrvals:
		
	
	See Also:
		
	
	Notes:
		
	
	Example:
		
	
	References:
		.. [1]
		
	"""
	
	# connect to the old DB
	old_db = mongo.bioDB(dbName, host='193.136.222.234', path=path)
	subs = mongo.subjects(**old_db)
	exps = mongo.experiments(**old_db)
	recs = mongo.records(**old_db)
	
	# convert the subjects
	print "Converting Subjects - DRY RUN"
	res = subs.get()['docList']
	for item in res:
		print item['_id']
		
		item.pop('records')
	
	# convert the experiments
	print "Convert Experiments"
	res = exps.get()['docList']
	for item in res:
		print item['_id']
		
		item.pop('records')
	
	# convert the records
	print "Convert Records"
	res = recs.getAll()['idList']
	for item in res:
		print item
		
		# header
		rec = recs.getById(item, {'data': 0, 'ndData': 0, 'dataList': 0, 'events': 0, 'nbEvents': 0, 'eventList': 0})['doc']
		
		types = recs.listTypes(item)
		# data
		for ty in types['dataTypes']:
			print ty
			bits = ty.split('/')
			weg = string.join(bits[:-1], '/')
			name = str(bits[-1])
			
			dataOut = recs.getData(item, weg, name)
			mdata = dataOut['mdata']
			if 'data' in mdata['name']:
				mdata.pop('name')
		
		# events
		for ty in types['eventTypes']:
			print ty
			bits = ty.split('/')
			weg = string.join(bits[:-1], '/')
			name = str(bits[-1])
			
			dataOut = recs.getEvent(item, weg, name)
			mdata = dataOut['mdata']
			if 'event' in mdata['name']:
				mdata.pop('name')
	
	print "Conversion Complete - DRY RUN"


def run(dbName, path):
	"""
	
	Runs the conversion process.
	
	Kwargs:
		dbName (str): Name of the database.
		
		path (str): Path to the HDF5 files.
	
	Kwrvals:
		
	
	See Also:
		
	
	Notes:
		
	
	Example:
		
	
	References:
		.. [1]
		
	"""
	
	# connect to the old DB
	old_db = mongo.bioDB(dbName, host='193.136.222.234', path=path)
	subs = mongo.subjects(**old_db)
	exps = mongo.experiments(**old_db)
	recs = mongo.records(**old_db)
	
	# connect to the new DB
	db = biomesh.biomesh(dbName=dbName+'_new', host='193.136.222.234', port=27017, dstPath=os.path.join(old_db['path'], 'new'), srvPath='/BioMESH', sync=True)
	
	# convert the subjects
	print "Converting Subjects"
	res = subs.get()['docList']
	for item in res:
		print item['_id']
		
		item.pop('records')
		db.subjects.add(item)
	
	# convert the experiments
	print "Convert Experiments"
	res = exps.get()['docList']
	for item in res:
		print item['_id']
		
		item.pop('records')
		db.experiments.add(item)
	
	# convert the records
	print "Convert Records"
	res = recs.getAll()['idList']
	for item in res:
		print item
		
		# header
		rec = recs.getById(item, {'data': 0, 'ndData': 0, 'dataList': 0, 'events': 0, 'nbEvents': 0, 'eventList': 0})['doc']
		recId = db.records.add(rec)['recordId']
		# update the dbName
		db.records.update(recId, {'dbName': dbName})
		
		types = recs.listTypes(item)
		# data
		for ty in types['dataTypes']:
			print ty
			bits = ty.split('/')
			weg = string.join(bits[:-1], '/')
			name = str(bits[-1])
			
			dataOut = recs.getData(item, weg, name)
			mdata = dataOut['mdata']
			if 'data' in mdata['name']:
				mdata.pop('name')
			db.records.addSignal(recId, weg, dataOut['signal'], mdata)
		
		# events
		for ty in types['eventTypes']:
			print ty
			bits = ty.split('/')
			weg = string.join(bits[:-1], '/')
			name = str(bits[-1])
			
			dataOut = recs.getEvent(item, weg, name)
			mdata = dataOut['mdata']
			if 'event' in mdata['name']:
				mdata.pop('name')
			db.records.addEvent(recId, weg, dataOut['timeStamps'], dataOut['values'], mdata)
	
	# close the DB
	db.close()
	
	print "Conversion Complete"


if __name__ == '__main__':
	dbName = 'HiMotion' # name of the database
	path = 'D:/BioMESH/Databases/HiMotion' # path to the HDF5 files
	
	# dry run - check for errors
	dry_run(dbName, path)
	
	# real run
	run(dbName, path)
