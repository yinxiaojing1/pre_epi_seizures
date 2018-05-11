'''
Created on 27 de Nov de 2012

@author: Carlos
'''

# imports
import sys
sys.path.append('/home/ccarreiras/work/BiometricsPyKit')
sys.path.append('/home/ccarreiras/work/BioSPPy')

import wavelets
from database import biomesh

if __name__ == '__main__':
    # connect to db
    db = biomesh.biomesh('CVP', host='193.136.222.234', dstPath='/home/ccarreiras/work/BioMESH', sync=True)
    
    records = db.records.getAll()['idList']
    waves = ['coif5', 'db3', 'db8', 'rbio3-3', 'rbio3-5', 'rbio3-7', 'rbio3-9', 'rbio5-5']
    
    for rec in records:
        print rec
        for wave in waves:
            print wave
            
            # get wavelet coeffs
            data = db.records.getSignal(rec, '/ECG/hand/Wavelets/' + wave, 0)
            matrix = data['signal']
            sampleRate = data['mdata']['sampleRate']
            
            # segment
            out = wavelets.waveletSegments(matrix, sampleRate)
            
            # save R
            mdata = {'source': '/ECG/hand/Wavelets/' + wave + '/signal0',
                     'algorithm': 'wavelet2',
                     }
            res = db.records.addEvent(rec, '/ECG/hand/Wavelets/' + wave + '/R/waveletSegmentation2', out['R'], mdata=mdata)
            
            # save segs
            mdata = {'source': {'signal': '/ECG/hand/Wavelets/' + wave + '/signal0', 'R': res['eventType'] + '/' + res['eventRef']},
                     'units': data['mdata']['units'],
                     'sampleRate': sampleRate,
                     'labels': ['Segments']
                     }
            res = db.records.addSignal(rec, '/ECG/hand/Wavelets/' + wave + '/Segments/waveletSegmentation2', out['segments'], mdata)
    
    # close connection to db
    db.close()
            
            
            