'''
Created on 31 de Out de 2012

@author: Carlos
'''

import datamanager as dm
import pylab

from database import biomesh
from database import h5db
import os
import glob

import h5py
from multiprocessing import Process, Queue
from Queue import Empty



def unwrap_updateBuffer(queue, ref, start, stop):
    queue.put({'data': [0, 1, 2, 3]})


def haha(queue, txt):
    queue.put({'text': txt})


def do():
    txt = 'haha'
    queue = Queue()
    p = Process(target=haha, args=(queue, txt))
    p.start()
    
    out = queue.get()
    
    print out

class H5Surf:
    def __init__(self, path, winLength):
        # open file
        self.fid = h5py.File(path, 'r')
        
        # self things
        self.buffer = {}
        self.winLength = winLength
        self.queue = Queue()
        self.process = None
        
        # load initial buffer
        # helper
        def filldict(x, y): D[x] = y
        D = dict()
        
        self.fid['signals'].visititems(filldict)
        
        for key in D.iterkeys():
            if isinstance(D[key], h5py.Dataset):
                dset = D[key]
                self.buffer[dset.name] = {'data': dset[:2*winLength], 'length': dset.shape[0], 'window': (0, 2 * winLength)}
    
    def updateBuffer(self):
        try:
            update = self.queue.get(timeout=5)
        except Empty:
            pass
        else:
            for key in update:
                self.buffer[key]['data'] = update[key]['data']
                self.buffer[key]['window'] = update[key]['window']
    
    def getFrame(self, ref, start, stop):
        try:
            update = self.queue.get(timeout=5)
        except Empty:
            pass
        else:
            for key in update:
                self.buffer[key] = update[key]
        
        try:
            data = self.buffer[ref]
        except KeyError:
            data = {'data': []}
        
        out = data['data']
        
        # update data in buffer
        self.queue = Queue()
        self.process = Process(target=unwrap_updateBuffer, args=(self.queue, ref, start, stop,))
        self.process.start()
        self.busy = True
        
        return out
    
    
    def moveLeft(self, ref, pan):
        self.updateBuffer()
        
        try:
            data = self.buffer[ref]
        except KeyError:
            data = {'data': [], 'length': 0, 'window': (0, 0)}
        
        start = self.winLength - pan
        if start < 0:
            start = 0
        stop = start + self.winLength
        
        out = data['data'][start:stop]
        
        return out
    
    def moveRight(self, ref, pan):
        self.updateBuffer()
        
        try:
            data = self.buffer[ref]
        except KeyError:
            data = {'data': [], 'length': 0, 'window': (0, 0)}
        
        stop = 2 * self.winLength + pan
        if stop > (3 * self.winLength):
            stop = 3 * self.winLength
        start = stop - self.winLength
        
        out = data['data'][start:stop]
        
        return out
    
    def close(self):
        self.fid.close()


if __name__ == '__main__':
    path = 'C:\\Users\\Carlos\\tmp\\datamanager\\rec_0.hdf5'
    
#    do()
    
    tst = H5Surf(path, 1000)
    
    
    
    out = tst.moveRight('signals/test/a/signal0', 10)
    
    
    
    
    # add some tags
#    
#    # connect to db
#    db = biomesh.biomesh('CVP', host='193.136.222.234', dstPath='D:\\BioMESH', sync=True)
#    records = db.records.getAll()['idList']
#    
#    haha = {'T1-Sitting': ['T1', 'Sitting'],
#            'T1-Recumbent': ['T1', 'Recumbent'],
#            'T2-Sitting': ['T2', 'Sitting'],
#            'T2-Recumbent': ['T2', 'Recumbent']}
#    
#    for item in records:
#        print item
#        exp = db.records.getById(item, restrict={'experiment': 1})['doc']['experiment']
#        db.records.addTags(item, haha[exp])
#    
#    # close db
#    db.close()
#
#
#
#    # convert CVP
#    
#    # path to old version
#    path = 'D:\\BioMESH\\Databases\\CVP_old'
#    
#    # connect to database
#    db = biomesh.biomesh('CVP', host='193.136.222.234', dstPath='D:\\BioMESH', sync=True)
#    
#    # add the experiments
#    db.experiments.add({'name': 'T1-Recumbent', 'goals': 'ECG acquisition.', 'description': 'First session; ECG acquired from hands, in recumbent position.'})
#    db.experiments.add({'name': 'T1-Sitting', 'goals': 'ECG acquisition.', 'description': 'First session; ECG acquired from hands, in sitting position.'})
#    db.experiments.add({'name': 'T2-Recumbent', 'goals': 'ECG acquisition.', 'description': 'Second session; ECG acquired from hands, in recumbent position.'})
#    db.experiments.add({'name': 'T2-Sitting', 'goals': 'ECG acquisition.', 'description': 'Second session; ECG acquired from hands, in sitting position.'})
#    
#    # add the subjects
#    with h5db.meta(os.path.join(path, 'ExpSub.hdf5')) as fidM:
#        subList = fidM.listSubjects()['subList']
#    
#    for item in subList:
#        if item['_id'] == 0:
#            continue
#        item.pop('records')
#        item.pop('source')
#        item['gender'] = item.pop('sex')
#        db.subjects.add(item)
#    
#    # add the records
#    cases = ['Sitting', 'Recumbent']
#    wavelets = ['coif5', 'db3', 'db8', 'rbio3-3', 'rbio3-5', 'rbio3-9', 'rbio5-5']
#    recFiles = glob.glob(os.path.join(path, 'rec_*.hdf5'))
#    for fname in recFiles:
#        with h5db.hdf(fname) as fid:
#            header = fid.getInfo()['header']
#            exp = header.pop('experiment')
#            header.pop('_id')
#            header.pop('dbName')
#            header.pop('session')
#            source = header['source']
#            
#            for case in cases:
#                # raw signal
#                weg = '/ECG/hand/' + case + '/raw'
#                weg_new = '/ECG/hand/raw'
#                out = fid.getSignal(weg, 'signal0')
#                Fs = float(out['mdata']['sampleRate'])
#                out['mdata'].pop('type')
#                duration = len(out['signal']) / Fs
#                
#                # add the record
#                record = header.copy()
#                record.update({'experiment': exp + '-' + case,
#                               'tags': [source],
#                               'duration': duration,
#                               'durationReference': '/ECG/hand/raw/signal0'})
#                recId = db.records.add(record)['recordId']
#                
#                # add raw signal
#                db.records.addSignal(recId, weg_new, out['signal'], out['mdata'])
#                
#                # add wavelets
#                for wave in wavelets:
#                    # coefficients
#                    weg = '/ECG/hand/' + case + '/Wavelets/' + wave
#                    weg_new = '/ECG/hand/Wavelets/' + wave
#                    out = fid.getSignal(weg, 'signal0')
#                    out['mdata'].pop('type')
#                    db.records.addSignal(recId, weg_new, out['signal'], out['mdata'])
#                    
#                    # segments
#                    weg = '/ECG/hand/' + case + '/Wavelets/' + wave + '/Segments'
#                    weg_new = '/ECG/hand/Wavelets/' + wave + '/Segments'
#                    out = fid.getSignal(weg, 'signal0')
#                    out['mdata'].pop('type')
#                    db.records.addSignal(recId, weg_new, out['signal'], out['mdata'])
#                    
#                    # R events
#                    weg = '/ECG/hand/' + case + '/Wavelets/' + wave + '/R'
#                    weg_new = '/ECG/hand/Wavelets/' + wave + '/R'
#                    out = fid.getEvent(weg, 'event0')
#                    out['mdata'].pop('type')
#                    db.records.addEvent(recId, weg_new, out['timeStamps'], out['values'], out['mdata'])
#    
#    # close db
#    db.close()
                    
    
    
#    config = {'source': 'HDF5',
#                  'path': 'D:\\BioMESH\\Databases\\CVP',
#                  #'experiments': ['T1-Deit','T2-Deit'],
#                  'experiments': ['T1-Deit','T2-Deit'],
#                  'source':'Enfermagem',
#                  'mapper': {'var1': 'signals/ECG/hand/Recumbent/raw',
#                             'var2': 'signals/ECG/hand/Sitting/raw'}
#                  }
#    
#    st = dm.Store(config)
#    
#    # db2data
#    out = st.db2data('var1', refine={'_id': [0]})
#    
#    #print out
#    
#    pylab.plot(out[0][0]['signal'])
#    pylab.show()
