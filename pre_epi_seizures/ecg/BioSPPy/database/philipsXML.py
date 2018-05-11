"""
.. module:: philipsXML
   :platform: Windows
   :synopsis: This module provides tools to read ECG records stored using the Philips SierraECG standard.

.. moduleauthor:: Carlos Carreiras
"""

# Imports
# built-in
import os

# 3rd party
import numpy as np
from lxml import etree
import xmltodict

# local
import h5db



def readXML(fpath=None, validate=True):
    """
    
    Read a decompressed Philips XML file.
    
    Kwargs:
        fpath (str): XML file to read.
        
        validate (bool): If True, validates the XML schema.
    
    Kwrvals:
        date (str): ISO 8601 acquisition date and time.
        
        labels (list): Label for each acquired ECG channel.
        
        resolution (int): Signal resolution.
        
        sampleRate (float): Acquisition sample rate.
        
        signal (array): Signal array, where each line is an ECG channel.
        
        subjectID (str): The ID of the subject.
        
        subjectName (str): The name of the subject.
        
        xmlFileName (str): Name of the source XML file.
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1]
        
    """
    
    # open file and read XML
    with open(fpath, 'r') as fid:
        # load to string
        doc_str = fid.read()
        # load to xml parser
        fid.seek(0)
        doc_parser = etree.parse(fid)
    
    if validate:
        # load schema file
        with open(os.path.join(os.path.split(__file__)[0], 'SierraECG.xsd'), 'r') as fid:
            schema_doc = etree.parse(fid)
        schema = etree.XMLSchema(schema_doc)
        
        # validate schema
        if not schema.validate(doc_parser):
            raise IOError, "XML file failed schema validation."
    
    # convert to dict
    doc = xmltodict.parse(doc_str)
    
    # return what we need
    output = {}
    
    # filename for reference
    output['xmlFileName'] = os.path.split(fpath)[1]
    
    # sample frequency
    output['sampleRate'] = float(doc['restingecgdata']['dataacquisition']['signalcharacteristics']['samplingrate'])
    
    # resolution
    output['resolution'] = int(doc['restingecgdata']['dataacquisition']['signalcharacteristics']['bitspersample'])
    
    # number of channels
    #nb = int(doc['restingecgdata']['dataacquisition']['signalcharacteristics']['numberchannelsallocated'])
    nb = int(doc['restingecgdata']['dataacquisition']['signalcharacteristics']['numberchannelsvalid'])
    
    # channel labels
    labels = doc['restingecgdata']['reportinfo']['reportformat']['waveformformat']['mainwaveformformat']['#text']
    output['labels'] = labels.split(' ')
    
    # date
    date = doc['restingecgdata']['dataacquisition']['@date']
    time = doc['restingecgdata']['dataacquisition']['@time']
    output['date'] = date + 'T' + time
    
    # patient ID
    output['subjectID'] = doc['restingecgdata']['patient']['generalpatientdata']['patientid']
    
    # patient name
    subjectName = ''
    firstName = doc['restingecgdata']['patient']['generalpatientdata']['name']['firstname']
    if firstName is not None:
        firstName = ' '.join([h5db.latin1ToAscii(item).capitalize() for item in firstName.split()])
        subjectName = subjectName + firstName
    middleName = doc['restingecgdata']['patient']['generalpatientdata']['name']['middlename']
    if middleName is not None:
        middleName = ' '.join([h5db.latin1ToAscii(item).capitalize() for item in middleName.split()])
        subjectName = subjectName + ' ' + middleName
    lastName = doc['restingecgdata']['patient']['generalpatientdata']['name']['lastname']
    if lastName is not None:
        lastName = ' '.join([h5db.latin1ToAscii(item).capitalize() for item in lastName.split()])
        subjectName = subjectName + ' ' + lastName
    output['subjectName'] = subjectName
    
    # waveform data
    compress = doc['restingecgdata']['waveforms']['parsedwaveforms']['@compressflag']
    if compress == 'True':
        raise IOError, "Compressed waveforms; please decompress first."
    
    data = np.fromstring(doc['restingecgdata']['waveforms']['parsedwaveforms']['#text'], sep=' ')
    data = data.reshape((nb, len(data) / nb))
    output['signal'] = data
    
    return output

