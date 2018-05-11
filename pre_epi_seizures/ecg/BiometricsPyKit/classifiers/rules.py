"""
.. module:: rules
   :platform: Unix, Windows
   :synopsis: This module provides various functions to...

.. moduleauthor:: Carlos Carreiras


"""

# Imports
# built-in

# 3rd party
import numpy as np

# BiometricsPyKit
import config



class EmptyCombination(Exception):
    # Exception raised when the the combination method is called on an empty array
    
    def __str__(self):
        return str("Combination of empty array.")


def majorityRule(labels):
    # majority
    pass


def pluralityRule(labels):
    # determine the most frequent class label
    
    counts = np.bincount(labels)
    predMax = counts.argmax()
    
    # check for repeats
    ind = np.nonzero(counts == counts[predMax])[0]
    length = len(ind)
    if length > 1:
        # choose randomly
        predMax = ind[config.random.randint(0, length)]
    
    return predMax, counts[predMax]


def combination(labels):
    # combine labels
    
    # ensure numpy
    res = np.array(labels, copy=False)
    
    # unique labels
    unq = np.unique(labels)
    
    nb = len(unq)
    if nb == 0:
        # empty array
        raise EmptyCombination
    elif nb == 1:
        # unanimous result
        return unq[0], np.sum(res == unq[0])
    else:
        counts = np.zeros(nb, dtype='float')
        
        # get count for each unique class
        for i in xrange(nb):
            counts[i] = np.sum(res == unq[i])
        
        # most frequent class
        predMax = counts.argmax()
        
        ### check for repeats?
        
        return unq[predMax], counts[predMax]

