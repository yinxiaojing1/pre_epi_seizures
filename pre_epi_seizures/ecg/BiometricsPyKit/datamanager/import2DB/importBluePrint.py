"""
.. module:: importBluePrint
   :platform: Unix, Windows
   :synopsis: Blue print to import data to StorageBIT.

.. moduleauthor:: Carlos Carreiras


"""

# imports
import os
import glob
from database import biomesh
import plux


if __name__ == '__main__':
    # connect to DB
    db = biomesh.biomesh(dbName='', dstPath='', host='193.136.222.234')
    
    # add experiments
    db.experiments.add({'name': '', 'goals': '', 'description': ''})
    
    # add subjects
    db.subjects.add({'name': '', 'birthdate': '', 'gender': '',
                     'email': '', 'notes': ''})
    
    # add records (from Plux files)
    rawDataPath = ""
    
    for fpath in glob.glob(os.path.join(rawDataPath, '*.txt')):
        print fpath
        
        # file name
        fname = os.path.split(fpath)[1]
        
        # map file name to a subject
        subId = db.subjects.get({'field': 'value'}, {'_id': 1})['docList'][0]['_id']
        
        # create record
        recId = db.records.add({'experiment': '', 'subject': subId, 'supervisor': '', 'date': ''})['recordId']
        
        # load data
        data = plux.loadbpf(fpath)
        
        # add data
        dataType = ''
        metadata = {'device': {'name': '',
                               'channels': [0, 1],
                               'Vcc': data.header['Vcc'],
                               'version': data.header['Version']
                               },
                    'transducer': '', 'placement': '',
                    'units': {'time': 'second', 'sensor': data.header['Units']},
                    'sampleRate': data.header['SamplingFrequency'],
                    'resolution': data.header['SamplingResolution']
                    }
        db.records.addSignal(recId, dataType, data, metadata)
        
        
    