'''
Created on Dec 3, 2012

@author: Filipe
'''
global NUMBER_PROCESSES
NUMBER_PROCESSES = 6

import os, scipy, numpy, pylab, glob, gzip, cPickle
from multiprocessing import Process, Queue, Manager
from Queue import Empty

def do_work(work_queue):
    while 1:
        try:
            q = work_queue.get(block=False)
            
            print "working on %s"%q['user']
            
            data_path =  'D:/data/mimic2wdb/distances/' 
            list_files = glob.glob(os.path.join(data_path, '0-0-%s-*'%q['user']))            
            
            os.chdir(data_path)
            
            res = {'genuine': [], 'imposter': []}
            
            for fn in list_files:
                user2 = fn.split('-')[-1].replace('.dict', '')
            
                fd = gzip.open(fn, 'rb')
                d = cPickle.load(fd)
                fd.close()
                            
                if q['user'] == user2: label = 'genuine'
                else: label = 'imposter'
                    
                for dis in d['distances']:
                    res[label] = scipy.hstack((res[label], dis))
            fd = gzip.open('hists/hist-%s.dict'%q['user'], 'wb')
            cPickle.dump(res, fd)
            fd.close()

#            bins = scipy.arange(0, 1.0, 0.005)
#            fig = pylab.figure()                
#            try:
#                pdfg, _, _ = pylab.hist(res['genuine'], bins, normed=True, color='g', alpha=0.5, label='genuine')
#                pdfi, _, _ = pylab.hist(res['imposter'], bins, normed=True, color='r', alpha=0.5, label='imposter')
#                pylab.grid()
#                pylab.axis('tight')
#                M = max(max(pdfg), max(pdfi))
#                aux = []
#                for i in xrange(len(bins)-1):
#                    aux.append([sum(pdfg[:i]*numpy.diff(bins[:i+1])), sum(pdfi[:i]*numpy.diff(bins[:i+1]))])
#                aux = scipy.array(aux)
#                i = scipy.argmax(aux[:,0] - aux[:,1])
#                pylab.vlines(bins[i], 0, M, 'k', '--', lw=3)
#                pylab.text(bins[i+1], M*0.5, '%0.3f genuine\n%0.3f imposter'%(aux[i][0], aux[i][1]))
#                fig.savefig('hists/hist-%s.png'%q['user'])
#                pylab.close()
#                
#                res['results'] = {'EER': bins[i], 'FAR': aux[i][1], 'FRR': 1.-aux[i][0]}
#                fd = gzip.open('hists/hist-%s.dict'%q['user'], 'wb')
#                cPickle.dump(res, fd)
#                fd.close()              
#            except Exception as e:
#                print q['user'], e
#                break
            
        except Empty:
            break

if __name__ == '__main__':
    
    data_path =  'D:/data/mimic2wdb/distances/'

    manager = Manager()
    output = manager.dict()
    work_queue = Queue() 
    
    list_files = glob.glob(os.path.join(data_path, '0-0-*'))
    os.chdir(data_path)
    
    users = []
    
    # fill queue
    print "filling queue"
    for fn in list_files:
        user = fn.split('-')[-2]
        if user in users: continue
        users.append(user)
        work_queue.put({'user': user})
    
    print "\nworking"    
    # create N processes and associate them with the work_queue and do_work function
    processes = [Process(target=do_work, args=(work_queue,)) for _ in range(NUMBER_PROCESSES)]
    for p in processes: p.start()
    for p in processes: p.join()
    for p in processes: p.terminate()
    