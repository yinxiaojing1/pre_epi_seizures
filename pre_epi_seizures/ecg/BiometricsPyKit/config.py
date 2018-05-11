"""
.. module:: config
   :platform: Unix, Windows
   :synopsis: This module provides various configuration parameters to be shared across all modules of BiometricsPyKit.

.. moduleauthor:: Carlos Carreiras


"""
# imports
# buily-in
import multiprocessing
import os

# 3rd party
import numpy as np
import pylab

# parameters for multiprocessing
try:
    nP = multiprocessing.cpu_count()
except NotImplementedError:
    numberProcesses = 4
else:
    numberProcesses =  nP / 2 if nP > 1 else 1

queueTimeOut = 2

# multiprocessing manager
manager = None

# folder for temp files
baseFolder = os.path.normpath(os.path.expanduser('~/BiometricsPyKitRun'))
folder = os.path.normpath(os.path.expanduser('~/BiometricsPyKitRun'))

# plotting parameters
# font parameters
font = {'family': 'Bitstream Vera Sans',
        'weight': 'normal',
        'size': 16}
pylab.rc('font', **font)

# pseudorandom number generator
randomSeed = None
random = np.random.RandomState()



def deploy(folders=None):
    # create the temp folders
    if folders is None:
        # default folders
        folders = [
                   'results',
                   'log',
                   ]
    
    global folder
    
    # expand the user and properly format the path
    if '~' in folder:
        folder = os.path.abspath(os.path.expanduser(folder))
    else:
        folder = os.path.abspath(folder)
    
    for f in folders:
        path = os.path.join(folder, os.path.normpath(f))
        
        # make sure the path exists
        if not os.path.exists(path):
            os.makedirs(path)


def getPath(sequence, verify=True):
    # build path
    
    global folder
    
    aux = [folder]
    aux.extend([os.path.normpath(item) for item in sequence])
    path = os.path.join(*aux)
    
    if verify:
        # make sure the path exists
        if not os.path.exists(path):
            os.makedirs(path)
    
    return path


def getFilePath(sequence):
    # build a file path
    
    global folder
    
    aux = [folder]
    aux.extend([os.path.normpath(item) for item in sequence])
    fpath = os.path.join(*aux)
    
    # make sure the path exists
    path = os.path.split(fpath)[0]
    if not os.path.exists(path):
        os.makedirs(path)
    
    return fpath


def setRandomSeed(seed=None):
    # set the seed for the random generator
    
    global random, randomSeed
    
    randomSeed = seed
    random.seed(seed)


def getRandomSeed():
    # get the current seed
    
    global randomSeed
    
    return randomSeed

