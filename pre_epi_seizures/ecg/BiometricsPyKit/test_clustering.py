'''
Created on Feb 4, 2013

@author: andre
'''

# how to read files

import gzip, cPickle
import os
import cluster
import pylab as pl
from itertools import cycle
from time import time

from multiprocessing import Process, Queue, Manager
from Queue import Empty
global NUMBER_PROCESSES
NUMBER_PROCESSES = 4

def recordRetrieval(FileUsers):
    listRecords=[]
    f = open(FileUsers)
    for s in f:
        listRecords.append(s.strip("\n"))
    f.close()

    listUsers=[]
    recordsPerUser={}
    for files in listRecords:
        t = files.split('-')
        print 'user:',t[0]
        print 'record:',t[1]
        print 'time:',t[2]
        if recordsPerUser.has_key(t[0]):
            a=recordsPerUser[t[0]]
            a.append(files)
        else: #new user
            listUsers.append(t[0])
            recordsPerUser[t[0]]=[files]
           
    return (listUsers, recordsPerUser)

def readSegments(record):
    #if isinstance(record,basestring):
    filename = record
    #else:
    #    print "more than one record, using the first"
    #    filename = recordrUser[0]

    dp = 'D:/data/mimic2wdb/segmentation/essf/'
    outp = 'D:/data/mimic2wdb/outlier/rcosdmean-mimicdb/output-%s'

    print filename
    fd = gzip.open(dp + filename, 'rb')
    data = cPickle.load(fd)
    fd.close()
    #data is dict with keys: 'segments', 'R', 'duration'
    segments=data['segments']

    fd = gzip.open(outp%filename, 'rb')
    outliersDict = cPickle.load(fd)
    #outliersDict['0']#outliersDict['-1']
    fd.close()
    #segments[outliersDict['0'],:]
    #plot(mean(segments[outliersDict['0'],:],0)-std(segments[outliersDict['0'],:],0),'--',color=c, linewidth=l)
    return (segments, outliersDict)

def do_work(work_queue, output):
    while 1:
        try:
            q = work_queue.get(block=False)
            # data
            print q['user']
            t0 = time()
            (segments, outlierDict) = readSegments(q['record'])

            #classifier_results = classifier.classify(data, **q['classifier_parameters']['parameters'])
            #clustering
            res = cluster.dbscan(segments)

            #prototypeCreation
            resProto=cluster.prototypeSelection(segments, res)
            
            # save
            #templates={}
            out = {'centroids': resProto["centroids"],
                   'nsamples_in_cluster':resProto["nsamples_in_cluster"], 
                   'clusters': res["clusters"]}
            
            filename = q['out_dir']+ q['record'] + "-prototype-temp.pkl"
            
            fd = gzip.open(filename, 'wb')
            cPickle.dump(out, fd)
            fd.close()
            print "done in %0.3fs" % (time() - t0)
            
        except Empty:
            break
            
def pkl2hdf5(listUsers,recordsPerUser,outfile):
    import h5py
    fhdf5 = h5py.File(outfile)
    for user in listUsers:
        for record in recordsPerUser[user]:
            subgroup = fhdf5.create_group(record)
            fd = gzip.open('prototypes2/'+ record + "-prototype-temp.pkl", 'rb')
            d = cPickle.load(fd)
            fd.close()
            dset2 = subgroup.create_dataset('centroids', data=d["centroids"])
            dset2 = subgroup.create_dataset('nsamples_in_cluster', data=d["nsamples_in_cluster"])
            #dset2 = subgroup.create_dataset('clusters', Dict=d["clusters"])
    fhdf5.close()        
    return

if __name__=='__main__':


    import test_clustering2FC
    from test_clustering2FC import *

    record_file = '1017T1.txt'
    outfile = 'prototypes2/mimic2data1017users-prototypes.hdf5'

    t0 = time()
    (listUsers, recordsPerUser) = recordRetrieval(record_file)
    sublist=listUsers#[0:10]
    #run_parallel(sublist)
    manager = Manager()
    output = manager.dict()

    cluster_parameters=[]
    out_dir='prototypes2/'

    work_queue = Queue()
    print "\n\n Computing clusters\n" 

    acc=0
    for user in sublist:
        for record in recordsPerUser[user]:
            acc=acc+1
            print "recordPerUser:" + record
            work_queue.put({
                        'user': user,
                        'record': record,
                        'cluster_parameters': cluster_parameters,
                        'out_dir': out_dir,
                        })

    print "Processing " + str(acc) + " Records."

    # create N processes and associate them with the work_queue and do_work function
    processes = [Process(target=do_work, args=(work_queue,output,)) for _ in range(NUMBER_PROCESSES)]
    # lauch processes
    print "working ...\n"
    for p in processes: p.start()
    # wait for processes to finish
    for p in processes: p.join()
    for p in processes: p.terminate()  

    t_batch = time() - t0
    print "total time of %0.3fs" % (time() - t0)
    pkl2hdf5(sublist)