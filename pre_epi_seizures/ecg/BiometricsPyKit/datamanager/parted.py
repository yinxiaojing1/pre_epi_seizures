"""
.. module:: parted
   :platform: Unix, Windows
   :synopsis: This module provides various functions to partition data into train and test sets.

.. moduleauthor:: Carlos Carreiras


"""


# imports
# built-in
import copy

# 3rd party
import numpy as np

# BiometricsPyKit
import config



def selector(method):
    """
    Selector for the parted functions and methods.
    
    Input:
        method (str): The desired function or method.
    
    Output:
        fcn (function): The function pointer.
    
    Configurable fields:
    
    See Also:
    
    
    Example:
    
    
    References:
        .. [1]
    """
    
    if method == 'random':
        fcn = randomFraction
    elif method == 'equal':
        fcn = equalFraction
    elif method == 'LKO':
        fcn = leaveKOut
    elif method == 'select':
        fcn = randomSelection
    else:
        raise TypeError, "Method %s not implemented." % method
    
    return fcn


def randomFraction(indx, run, fraction=None):
    """
    Randomly select a fraction of the elements for the training set.
    
    Input:
        indx (list, array): Indices to partition.
        
        run (int): Number of the run.
        
        fraction (int, float): The fraction to select.
    
    Output:
        use (list, array): The selected indices.
        
        unuse (list, array): The exluded indices.
    
    Configurable fields:
    
    See Also:
    
    
    Example:
    
    
    References:
        .. [1]
    """
    
    # check inputs
    if fraction is None:
        raise TypeError, "Please specify the selection fraction."
    
    # number of elements to use
    fraction = float(fraction)
    nb = int(fraction * len(indx))
    
    # copy indx because shuffle works in place
    aux = copy.deepcopy(indx)
    
    # shuffle
    config.random.shuffle(aux)
    
    # select
    use = aux[:nb]
    use.sort()
    unuse = aux[nb:]
    unuse.sort()
    
    return use, unuse


def deterministicFraction(indx, run, fraction=None, order='start'):
    """
    Deterministically select a fraction of the elements for the training set.
    
    Input:
        indx (list, array): Indices to partition.
        
        run (int): Number of the run.
        
        fraction (int, float): The fraction to select.
        
        order (str): Determines whether the training data is selected from the start ('start', the default), or from the end ('end') of the set.
    
    Output:
        use (list, array): The selected indices.
        
        unuse (list, array): The exluded indices.
    
    Configurable fields:
    
    See Also:
    
    
    Example:
    
    
    References:
        .. [1]
    """
    
    # check inputs
    if fraction is None:
        raise TypeError, "Please specify the selection fraction."
    
    # number of elements to use
    fraction = float(fraction)
    nb = int(fraction * len(indx))
    
    # select
    if order == 'start':
        use = indx[:nb]
        unuse = indx[nb:]
    elif order == 'end':
        use = indx[-nb:]
        unuse = indx[:-nb]
    else:
        raise ValueError, "Unknown order method."
    
    return use, unuse


def randomSelection(indx, run, k=None):
    """
    Randomly select a k elements for the training set.
    
    Input:
        indx (list, array): Indices to partition.
        
        run (int): Number of the run.
        
        k (int): The number of elements to select.
    
    Output:
        use (list, array): The selected indices.
        
        unuse (list, array): The exluded indices.
    
    Configurable fields:
    
    See Also:
    
    
    Example:
    
    
    References:
        .. [1]
    """
    
    # check inputs
    if k is None:
        raise TypeError, "Please specify the number of elements."
    
    # copy indx because shuffle works in place
    aux = copy.deepcopy(indx)
    
    # shuffle
    config.random.shuffle(aux)
    
    # select
    use = aux[:k]
    use.sort()
    unuse = aux[k:]
    unuse.sort()
    
    return use, unuse


def equalFraction(indx, run):
    """
    Equal sets (no separation).
    
    Input:
        indx (list, array): Indices to partition.
        
        run (int): Number of the run.
    
    Output:
        use (list, array): The selected indices.
        
        unuse (list, array): The exluded indices.
    
    Configurable fields:
    
    See Also:
    
    
    Example:
    
    
    References:
        .. [1]
    """
        
    # select
    use = copy.deepcopy(indx)
    unuse = copy.deepcopy(indx)
    
    return use, unuse


def leaveKOut(indx, run, k=1, random=True):
    """
    Leave k items out of the training set.
    
    Input:
        indx (list, array): Indices to partition.
        
        run (int): Number of the run.
        
        k (int): Number of elements to leave out. Default=1
        
        random (bool): If True, select randomly the items. Default=True
    
    Output:
        use (list, array): The selected indices.
        
        unuse (list, array): The exluded indices.
    
    Configurable fields:
    
    See Also:
    
    
    Example:
    
    
    References:
        .. [1]
    """
    
    # check inputs
    if k < 1:
        raise ValueError, "The parameter k is smaller than 1."
    if k > 1 and not random:
        raise ValueError, "Incompatible parameters: sequential selection only implemented for k=1."
    
    # helper function
    if random:
        def helper(indx):
            # shufle
            config.random.shuffle(indx)
            
            # select
            use = indx[:-k]
            use.sort()
            unuse = indx[-k:]
            unuse.sort()
            
            return use, unuse
    else:
        nb = len(indx)
        def helper(indx):
            # position to leave out
            i = run % nb
            
            try:
                u = indx.pop(i)
            except AttributeError:
                # maybe numpy?
                u = indx[i]
                indx = np.delete(indx, i)
            
            use = indx
            unuse = [u]
            
            return use, unuse
    
    # copy indx
    aux = copy.deepcopy(indx)
    
    use, unuse = helper(aux)
    
    return use, unuse

