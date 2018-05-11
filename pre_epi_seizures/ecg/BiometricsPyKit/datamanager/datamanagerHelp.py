"""
.. module:: datamanagerHelp
   :platform: Unix, Windows
   :synopsis: Script to exemplify datamanager usage.

.. moduleauthor:: Carlos Carreiras


"""

import numpy as np
import datamanager as dm

if __name__ == '__main__':
    print "Data Manager Help Script"
    
    """
    Configuration Dictionary
    
    Field description:
        source (string): Type of the underlying storage ('StorageBIT' or 'HDF5')
        experiments (string, or list of strings): The experiments to load.
        mapper (dict): To map user variables to the database representation.
        DBconnection (dict): Connection parameters for StorageBIT (only needed in 'StorageBIT')
        path (string): Path to the HDF5 files (only needed in 'HDF5').
    
    The online version ('StorageBIT') requires Unison.
    """
    # online database ('StorageBIT')
    config = {'source': 'StorageBIT',
              'DBConnection': {'dbName': 'storageTest',
                               'host': '193.136.222.234',
                               'port': 27017,
                               'dstPath': 'D:/BioMESH'},
              'experiments': ['T1', 'T2', 'T3'],
              'mapper': {'var1': 'signals/random/raw',
                         'var2': 'signals/random/a',
                         'var3': 'events/random/annotation',
                         'var4': 'signals/random/b'}
              }
    
    # offline database ('HDF5')
    config = {'source': 'HDF5',
              'path': '~/storageTest',
              'experiments': ['T1', 'T2', 'T3'],
              'mapper': {'var1': 'signals/test',
                         'var2': 'signals/test/a',
                         'var3': 'events/test/annotation',
                         'var4': 'signals/test/c'}
              }
    
    #----------------------------------------------------
    """
    Launch the manager - Store
    """
    
    store = dm.Store(config)
    
    #----------------------------------------------------
    """
    Check what's inside - dbmetada
    
    The information (metadata) relating to the loaded records (belonging to the experiments
    specified in the configuration dictionary) is accessible with the method 'dbmetada'.
    This produces a dictionary where the keys are record IDs, and the values are dictionaries
    with the metadata.
    """
    
    output = store.dbmetada()
    # output is {0: record 0's metadata, 1: record 1's metadata, ...}
    
    # get metadata only for specific records (e.g. only records 0 and 1)
    output = store.dbmetada(refine={'_id': [0, 1]})
    
    #----------------------------------------------------
    """
    Check what's inside - getSubjectInfo
    
    The information (metadata) relating to the loaded subjects (those that participated
    in the experiments specified in the configuration dictionary) is accessible with the
    method 'getSubjectInfo'. This produces a dictionary where the keys are subject IDs,
    and the values are dictionaries with the metadata.
    """
    
    output = store.getSubjectInfo()
    # output is {0: subject 0's metadata, 1: subject 1's metadata, ...}
    
    # get metadata only for specific subjects (e.g. only subjects 1 and 2)
    # don't forget that subject 0 is the generic (unnamed) subject
    output = store.getSubjectInfo(refine={'_id': [1, 2]})
    
    #----------------------------------------------------
    """
    Add a new experiment - addExperiment
    
    To add a new experiment to the database use the addExperiment method. The
    experiment needs to specify a name. If an experiment with the same name
    already exists in the database, the new insertion is skipped and a warning
    is printed. The ID of the new experiment is automatically generated.
    
    Non-ASCII characters in experiment names are automatically converted to an
    ASCII equivalent.
    """
    
    experiment = {'name': 'T1',
                  'goals': 'Test the insertion of experiments.',
                  'description': 'Test experiment 1.'
    }
    experimentName = store.addExperiment(experiment)
    
    #----------------------------------------------------
    """
    Add a new subject - addSubject
    
    To add a new subject to the database use the addSubject method. The
    subject needs to specify a name. If a subject with the same name
    already exists in the database, the new insertion is skipped and a warning
    is printed. The ID of the new subject is automatically generated.
    
    Non-ASCII characters in subject names are automatically converted to an
    ASCII equivalent.
    """
    
    subject = {'name': 'John Smith',
               'birthdate': 'yyyy-mm-ddThh:mm:ss',
               'gender': 'M',
               'email': 'john.smith@domain.com'
    }
    subjectId = store.addSubject(subject)
    
    #----------------------------------------------------
    """
    Add a new record - addRecord
    
    To add a new record to the database (e.g. as a result of a new acquisition) use
    the addRecord method. The header needs to specify an experiment and a subject.
    If these are not present, the generic experiment and/or subject will be used. The
    specified experiment and/or subject must exist in the database; otherwise a
    ValueError exception is raised. The session number and duration fields are
    generated automatically. The experiment name is always added to the tags field.
    """
    
    header = {'experiment': 'T1',
              'subject': subjectId,
              'date': 'yyyy-mm-ddThh:mm:ss',
              'tags': ['test'],
              'comments': 'New test record.',
              'supervisor': 'John Smith'
    }
    recordId = store.addRecord(header)
    
    #----------------------------------------------------
    
    """
    Add data to the database - data2db
    
    To add data to the database (signals or events) use the method data2db.
    To do this, pass a dictionary to the method containing the data to add for
    each record, and the list of user variables (where the data will be stored).
    The keys of the data dictionary are record IDs, and the corresponding values
    are dictionaries with the data to add indexed by user variables. The data
    dictionary also has the 'info' key, where the metadata is included (common
    to all records). The distinction between signals or events is implicitly
    defined by the user variable. The method also takes an optional argument
    'updateDuration' (boolean) to update the duration field of the record. This
    should be used when adding the raw signals to an empty record. The default
    behavior is to not update the duration.
    """
    
    # add signals
    signal = np.zeros(100)
    mdata = {'comments': 'zeros',
             'sampleRate': 100.,
             'resolution': 10,
             'units': {'sensor': 'ADC', 'time': 'second'},
             'labels': ['Z'],
             'device': {'name': 'MAC add.', 'channels': ['channels used']}
             }
    data = {recordId: {'var1': signal}, 'info': mdata}
    store.data2db(data, ['var1'], updateDuration=True)
    
    # add events
    timeStamps = range(0, 100, 10)
    mdata = {'comments': 'indexes',
             'source': config['mapper']['var3'] # this is ugly, I know...
             }
    data = {recordId: {'var3': timeStamps}, 'info': mdata}
    store.data2db(data, ['var3'])
    
    #----------------------------------------------------
    """
    Get data from the database - db2data
    
    To get data from the database (signals or events) use the method db2data.
    The method produces a dictionary where the keys are record IDs and the
    corresponding values are lists (there may be more than one dataset for the
    specified user variable) with the retrieved data ('signal' or 'timeStamps')
    from the specified user variable, and the metadata ('mdata').
    """
    
    # get signals
    out = store.db2data('var1')
    # output is {0: [{'signal': [...], 'mdata': {...}}, ...], 1: ...}
    
    # get data only for specific records (e.g. only record 0)
    out = store.db2data('var1', refine={'_id': [0]})
    # output is {0: [{'signal': [...], 'mdata': {...}}, ...]}
    
    # get events
    out = store.db2data('var3')
    # output is {0: [{'timeStamps': [...], 'mdata': {...}}, ...], 1: ...}
    
    # get data only for specific records (e.g. only record 0)
    out = store.db2data('var3', refine={'_id': [0]})
    # output is {0: [{'timeStamps': [...], 'mdata': {...}}, ...]}
    
    #----------------------------------------------------
    """
    Produce Train and Test Sets - subjectTTSets
    
    In order to split the available data into Train and Test sets, use the
    method subjectTTSets. This is where tags become useful.
    The method takes as input a dictionary with the following keys:
        train_set (list): train tags.
        test_set (list): test tags.
        train_time (tuple of ints): train record time in seconds.
        test_time (tuple of ints): test record time in seconds.
    """
    
    info = {'train_set': ['T1'],
            'test_set': ['T2', 'T3'],
            'train_time': (0, 60),
            'test_time': (0, 60)
            }
    out = store.subjectTTSets(info)
    # output is [(subject 0, [train record id(s)], [test record id(s))], ...]
    
    #----------------------------------------------------
    """
    Close the store - close
    
    Do not forget to close the manager instance once all the work is done. This
    is especially important when using the online database (to allow the
    synchronization with Unison).
    """
    
    store.close()
