import sys
sys.path.append('database/')
import mongoH5 as mg
from mg import *

if __name__ == '__main__':

	# Example
	import numpy as np
	import pylab as pl
	import datetime
	import time

	exp0 = {'name': 'exp1', 'description': 'This is a sample experiment.', 'goals': 'To illustrate an experiment document.'}

	pp0 = {'name': 'Mr. Sample Subject', 'age': 0, 'sex': 'm', 'email': 'sample@domain.com'}

	rec0 = {'name': 'rec0', 'date': datetime.datetime.utcnow().isoformat(), 'experiment': 'exp1', 'subject': 'Mr. Sample Subject', 'supervisor': 'Mr. Super'}

	# open connection to db, create one if there is none
	config = {'dbName': 'biodb_test', 'host': '193.136.222.235', 'port': 27017}
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
	mdata = {'type': 'EEG', 'labels': ['Ch.1', 'Ch.2', 'Ch.3'], 'device': {'channels': [0, 1, 2], 'name': 'Sample Device'}, 'tranducer': 'Ag/AgCl electrodes', 'units': {'time': 'second', 'sensor': 'microVolt'}, 'sampleRate': Fs, 'resolution': 12, 'duration': duration}
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
	mdata = {'source': 'eventGenerator', 'type': '2D array', 'dictionary':{'0':0, '1':1, '2':2, '3':3}, 'eventSync': now.isoformat()}
	
	eventIn = recs.addEvent(recId, timeStamps, values, mdata)
	
	# get data
	dataOut = recs.getData(**dataIn)
	
	pl.plot(t, dataOut['signal'])
	pl.show()