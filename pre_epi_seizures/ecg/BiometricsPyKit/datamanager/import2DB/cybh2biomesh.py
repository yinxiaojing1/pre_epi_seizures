"""
.. module:: cybh2biomesh
   :platform: Unix, Windows
   :synopsis: To convert the CYBH database from the old API to the new BioMESH DB.
   
.. modeuleauthor:: Carlos Carreiras
"""

from database import biomesh



def run(path):
    # connect to the old DB
    db_old = biomesh.biomesh(dbName='CYBH_new', host='193.136.222.234', port=27017, dstPath=path, srvPath='/BioMESH', sync=False)
    
    # connect to the new DB
    db_new = biomesh.biomesh(dbName='CYBH2', host='193.136.222.234', port=27017, dstPath=path, srvPath='/BioMESH', sync=True)
    
    import time
    time.sleep(60)
    
    # convert the subjects
    print "Converting Subjects"
    res = db_old.subjects.get()['docList']
    for item in res:
        print item['_id']
        
        item.pop('records')
        item.pop('_id')
        db_new.subjects.add(item)
    
    
    # add the experiments
    print "Add Experiments"
    db_new.experiments.add({'name': 'A1', 'goals': 'To stimulate low amplitude SCR events.', 'description': 'Amusing video; [0, 55]s Sony Bravia paint ad; [55, 60]s Monkey falls off tree.'})
    db_new.experiments.add({'name': 'A2', 'goals': 'To stimulate high amplitude SCR events.', 'description': 'Intense video; [0, 55]s Sony Bravia bouncy balls ad; [55, 60]s Final part of the REC Movie trailer.'})
    db_new.experiments.add({'name': 'CI', 'goals': '', 'description': 'Acquisition during reading of informed consent.'})
    db_new.experiments.add({'name': 'BT', 'goals': 'Increase the CYBH database; compare the ECG signal from the capacitive sensor to the signal from the hands.', 'description': 'Acquisition during Bio@IT 2012.'})
    db_new.experiments.add({'name': 'RWS', 'goals': 'To see how the ECG changes with the heart rate.', 'description': 'ECG Signals acquired during rest, mild exercise (walk) and exercise (stairs).'})
    
    
    # convert the records
    print "Convert Records"
    # A1, A2, CI experiment
    res = db_old.records.getAll({'experiment': ['A1', 'A2', 'CI']})['idList']
    for item in res:
        print item
        
        # header
        rec = db_old.records.getById(item, {'signals': 0, 'events': 0, 'audit': 0, 'nbData': 0, '_id': 0, 'dbName': 0})['doc']
        recId = db_new.records.add(rec)['recordId']
        
        # get the raw signals
        types = ['/ECG/fingers/raw', '/ECG/hand/raw', '/EDA/left_hand/raw',
                 '/EDA/left_hand/Glove/raw', '/EDA/right_hand/raw', '/Sync',
                 '/Sync/LDR/LED', '/Sync/LDR/iPad']
        # data
        for ty in types:
            print ty
            try:
                data = db_old.records[item]['signals'][ty][:]
            except KeyError:
                print "skipped ", ty
                continue
            else:
                for obj in data:
                    obj.metadata.pop('name')
                    obj.metadata.pop('type')
                    db_new.records.addSignal(recId, ty, obj.signal, obj.metadata)
    
    # BT experiment
    res = db_old.records.getAll({'experiment': 'BT'})['idList']
    for item in res:
        print item
        
        # header
        rec = db_old.records.getById(item, {'signals': 0, 'events': 0, 'audit': 0, 'nbData': 0, '_id': 0, 'dbName': 0, 'experiment': 0})['doc']
        rec.update({'experiment': 'BT'})
        recId = db_new.records.add(rec)['recordId']
        
        # get the raw signals
        types = ['/ECG/back/raw', '/ECG/fingers/raw']
        # data
        for ty in types:
            print ty
            try:
                data = db_old.records[item]['signals'][ty][:]
            except KeyError:
                print "skipped ", ty
                continue
            else:
                for obj in data:
                    obj.metadata.pop('name')
                    obj.metadata.pop('type')
                    db_new.records.addSignal(recId, ty, obj.signal, obj.metadata)
    
    # RWS experiment
    res = db_old.records.getAll({'experiment': 'ReWast'})['idList']
    for item in res:
        print item
        
        # header
        rec = db_old.records.getById(item, {'signals': 0, 'events': 0, 'audit': 0, 'nbData': 0, '_id': 0, 'dbName': 0, 'experiment': 0})['doc']
        rec.update({'experiment': 'RWS'})
        recId = db_new.records.add(rec)['recordId']
        
        # get the raw signals
        types = ['/ECG/Rest/raw', '/ECG/Walk/raw', '/ECG/Stairs/raw']
        # data
        for ty in types:
            print ty
            try:
                data = db_old.records[item]['signals'][ty][:]
            except KeyError:
                print "skipped ", ty
                continue
            else:
                for obj in data:
                    obj.metadata.pop('name')
                    obj.metadata.pop('type')
                    db_new.records.addSignal(recId, ty, obj.signal, obj.metadata)
    
    # close the DB
    db_old.close()
    db_new.close()
    
    print "Conversion Complete."


if __name__ == '__main__':
    path = 'D:\BioMESH' # path to the HDF5 files
    
    # run
    run(path)
    
    