"""
.. module:: biomesh
   :platform: Unix, Windows
   :synopsis: Convert Physionet database files to csv (requires wfdb).

.. moduleauthor:: Carlos Carreiras
"""

# Imports
# built-in
import csv
import fnmatch
import os
import subprocess

# 3rd party
import numpy as np



def convert2csv(basePath, record, dirPath):
    # convert a Physionet data file to CSV format
    # basePath - path to database record
    # record - record name without extension
    # dirPath - destination path to save csv file (don't forget to put filename with csv extension)
    
    # make sure there are no trailing spaces
    record = record.strip()
    
    # read file descriptor
    command = 'wfdbdesc %s' % record
    p = subprocess.Popen(unicode(command).encode('utf-8'),
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         shell=True, cwd=basePath)
    stdout, stderr = p.communicate()
    ret = p.returncode
    
    if ret < 0:
        print stderr
        raise OSError, "Error reading file descriptor."
    
    # extract signal length
    desc = stdout.split('\r\n')
    desc = filter(lambda line: 'Length' in line, desc)
    
    if len(desc) != 1:
        raise ValueError, "Unexpected descriptor format (length hits)."
    
    desc = desc[0]
    try:
        length = int(desc.split('(')[1].split(' ')[0]) - 1
    except IndexError:
        raise ValueError, "Unexpected descriptor format (length structure)."
    
    # read data file
    command = 'rdsamp -r %s -c -p -t s%d -v > %s' % (record, length, dirPath)
    p = subprocess.Popen(unicode(command).encode('utf-8'),
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         shell=True, cwd=basePath)
    _, stderr = p.communicate()
    ret = p.returncode
    
    if ret < 0:
        print stderr
        raise OSError, "Error reading data file."


def convertAnnotation2CSV(basePath, record, dirPath):
    # convert Physionet annotations files .st to csv
    # basePath - path to database record
    # record - record name without extension
    # dirPath - destination path to save csv file (don't forget to put filename with csv extension)

    # make sure there are no trailing spaces
    record = record.strip()

    # read file descriptor
    command = 'rdann -v -r %s -a st > %s' % (record, dirPath)
    p = subprocess.Popen(unicode(command).encode('utf-8'),
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         shell=True, cwd=basePath)
    stdout, stderr = p.communicate()
    ret = p.returncode

    if ret < 0:
        print stderr
        raise OSError, "Error reading st file."


def listFiles(basePath, db):
    # list all the files belonging to db
    
    path = os.path.join(basePath, db)
    nb = len(path.split(os.path.sep)) - 1
    for root, _, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*.dat'):
            aux = os.path.splitext(os.path.join(root, filename))[0]
            aux = aux.split(os.path.sep)[nb:]
            yield os.path.join(*aux)


def batchConvert(basePath, db, dirPath=None):
    # convert all files of db into csv files
    
    # check inputs
    if dirPath is None:
        dirPath = basePath
    
    for record in listFiles(basePath, db):
        # print record.split("\\")[1]
        try:
            convert2csv(basePath, record, os.path.join(dirPath, record + '.csv'))
            convertAnnotation2CSV(basePath, record, os.path.join(dirPath, record + '_ST.csv') )
        except (OSError, ValueError), e:
            print "Error:", e


def readCSV(path):
    # read a Physionet CSV file to numpy
    
    # read from file
    data = []
    with open(path, 'rb') as fid:
        reader = csv.reader(fid)
        
        # read header
        labels = reader.next()
        units = reader.next()
        
        # read values
        for row in reader:
            data.append(row)
    
    # convert to numpy
    data = np.array(data, dtype='float')
    
    return {'labels': labels, 'units': units, 'data': data}

