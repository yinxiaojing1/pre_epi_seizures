"""
.. module:: client
   :platform: Unix, Windows
   :synopsis: This module implements the BITCloud client.

.. moduleauthor:: Carlos Carreiras


"""

# Notes


# Imports
# built in
import copy
import os

# 3rd party
from kombu import Connection
import pymongo as pmg

# BiometricsPyKit
from datamanager import datamanager
from misc import misc


class connection:
    """
    
    Class to connect to a BITCloud server.
    
    Kwargs:
        
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1]
        
    """
    TRANSPORT = 'amqp'
    VHOST = '/'
    RABBIT_PORT = 5672
    MONGO_HOST = '193.136.222.234'
    MONGO_PORT = 27017
    
    def __init__(self, host='localhost', user=None, passwd=None, queue='BiometricsQ', dbName='BiometricsExperiments'):
        """
        
        Establish the connection to the server.
        
        Kwargs:
            
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        # connect to RabbitMQ server
        self.conn = Connection(
                               hostname=host,
                               port=self.RABBIT_PORT,
                               transport=self.TRANSPORT,
                               userid=user,
                               password=passwd,
                               virtual_host=self.VHOST
                               )
        
        # declare queue (persistent queue)
        self.queue = self.conn.SimpleQueue(queue)
        
        # connect to MongoDB
        self.mongo_conn = pmg.Connection(self.MONGO_HOST, self.MONGO_PORT)
        self.mongo_db = self.mongo_conn[dbName]
        self.mongo_collection = self.mongo_db['experiments']
        self.mongo_id = self.mongo_db['IDTracker']
        
        # check if ID tracker exists
        res = self.mongo_id.find_one({'_id': 0}, {'_id': 1})
        if res is None:
            self.mongo_id.insert({'_id': 0, 'nextID': 0})
        
        # check if user exists in MongoDB
        collection = self.mongo_db['users']
        doc = collection.find_one({'username': user}, {'_id': 1})
        if doc is None:
            s = raw_input("Please register your E-Mail to be notified of task completion:\n")
            while True:
                if s is '':
                    collection.insert({'username': user})
                    print "User %s registered without E-Mail.\nYou will not be able to receive notifications." % user
                    break
                elif '@' not in s:
                    s = raw_input("Badly formated E-Mail address - missing '@' character.\nYou provided %s.\nPlease retype the E-Mail address:\n" % s)
                elif '.' not in s.split('@')[1]:
                    s = raw_input("Badly formated E-Mail address - missing domain name.\nYou provided %s.\nPlease retype the E-Mail address:\n" % s)
                else:
                    collection.insert({'username': user, 'email': s})
                    print "User %s registered with E-Mail %s." % (user, s)
                    break
        
        # extra self things
        self.user = user
    
    def addTask(self, task=None):
        
        # check inputs
        if task is None:
            raise TypeError, "Please specify a task."
        
        # get a new experiment ID from MongoDB (atomic operation)
        expID = self.mongo_id.find_and_modify(query={'_id': 0}, update={'$inc': {'nextID': 1}})['nextID']
        
        task.update({'_id': expID,
                     'user': self.user,
                     'status': 'waiting',
                     'seen': False})
        
        # save to MongoDB
        self.mongo_collection.insert(task)
        
        # confirm insert
        _ = self.mongo_collection.find_one({'_id': expID}, {'status': 1})
        
        # publish to RabbitMQ
        self.queue.put({'_id': expID})
        
        return expID
    
    def cancelTask(self, expID):
        # cancel a task (will be ignored by worker when delivered)
        
        ### update status checks on worker
        
        # update and get task
        qry = {'_id': expID}
        upd = {'$set': {'user': self.user,
                        'status': 'finished',
                        'seen': True,
                        'results': None,
                        'worker': None,
                        },
               }
        self.mongo_collection.update(qry, upd)
    
    def requeue(self, expID):
        # requeue a task
        
        # update and get task
        qry = {'_id': expID}
        upd = {'$set': {'user': self.user,
                        'status': 'waiting',
                        'seen': False}}
        self.mongo_collection.update(qry, upd)
        
        # publish to RabbitMQ
        self.queue.put(qry)
    
    def listTasks(self, status='all', seen=None):
        # list tasks belonging to user
        # status = 'waiting' | 'running' | 'finished' | 'error' | 'all'
        
        search = {'user': self.user}
        
        # update the search
        if status is not 'all':
            search.update({'status': status})
        
        if seen is not None:
            search.update({'seen': seen})
        
        # perform the search
        doc = self.mongo_collection.find(search, {'_id': 1})
        res = [item['_id'] for item in doc]
        
        return res
    
    def getTaskInfo(self, expID):
        # retrieve a task
        
        doc = self.mongo_collection.find_one({'_id': expID})
        
        return doc
    
    def search(self, spec=None, fields=None):
        # search task DB
        
        return self.mongo_collection.find(spec, fields)
    
    def collectResults(self, expIDList, mapper, basePath, headerOrder=None):
        # collect the results of a list of experiments, selecting the items in mapper
        
        if headerOrder is None:
            # use default order
            headerOrder = ['starting_data', 'train', 'test', 'dimreduction']
        
        # determine number of columns per row
        keyOrder = []
        for item in headerOrder:
            try:
                aux = len(mapper[item]['items'])
            except KeyError:
                pass
            else:
                if aux > 0:
                    keyOrder.append(item)
        
        rows = []
        for item in expIDList:
            task = self.getTaskInfo(item)
            mrow = []
            
            # task ID
            mrow.append(task['_id'])
            
            # general parameters
            for c in keyOrder:
                for pair in mapper[c]['items']:
                    mrow.append(misc.slasherDict(task[c], pair[1]))
            
            # classifiers
            for name in mapper['classifier']['instances'].iterkeys():
                for clf in mapper['classifier']['instances'][name]:
                    row = copy.deepcopy(mrow)
                    
                    # append name
                    row.append(name)
                    
                    # append parameters
                    try:
                        for pair in mapper['classifier']['items']:
                            row.append(misc.slasherDict(task['classifier'][clf], pair[1]))
                    except KeyError:
                        # clf not present
                        continue
                    
                    # load global results
                    res = datamanager.skLoad(os.path.join(basePath,
                                                          'Exp-%d' % item,
                                                          'results', clf,
                                                          'results-global.dict'))
                    EER = 100. * res['global']['authentication']['rates']['EER'][0, 1]
                    EID = 100. * res['global']['identification']['rates']['Err']
                    row.append(EER)
                    row.append(EID)
                    
                    # append row
                    rows.append(row)
        
        # build header
        header = [[], []]
        
        # task ID
        header[0].append('Task ID')
        header[1].append('')
        
        # general parameters
        for c in keyOrder:
            header[0].append(mapper[c]['name'])
            nb = len(mapper[c]['items'])
            header[0].extend((nb - 1) * [''])
            for i in xrange(nb):
                header[1].append(mapper[c]['items'][i][0])
        
        # classifier
        header[0].append(mapper['classifier']['name'])
        nb = len(mapper['classifier']['items'])
        header[0].extend((nb + 2) * ['']) # do not forget name, EER, EID
        header[1].append('Name')
        for i in xrange(nb):
            header[1].append(mapper['classifier']['items'][i][0])
        header[1].append('EER (%)')
        header[1].append('EID (%)')
        
        return {'rows': rows, 'header': header}
    
    def markSeen(self, expID):
        # mark a task (or list of tasks) as seen
        
        if type(expID) is int:
            expID = [expID]
        elif type(expID) is list:
            pass
        else:
            raise TypeError, "The input must be of type int or list"
        
        for item in expID:
            self.mongo_collection.update({'_id': item}, {'$set': {'seen': True}})
    
    def close(self):
        # close connection to RabbitMQ
        self.conn.release()
        # close connection to MongoDB
        self.mongo_conn.close()



if __name__ == '__main__':
    pass
