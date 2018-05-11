'''
Created on Nov 13, 2012

@author: Filipe
'''

import scipy, pylab, gzip, cPickle, numpy, os, glob
import peakd
from datamanager import datamanager 
from misc import misc
from itertools import izip

if __name__ == '__main__':
              
    if False: # PLOT SEGMENTS, OUTLIERS, Q & S position
        
        config = {'source': 'HDF5',
                  'path': r'\\193.136.222.220\cybh\data\CVP\hdf5',
                  'experiments': ['T1-Sitting', 'T2-Sitting', 'T1-Recumbent', 'T2-Recumbent'],
                  'mapper': {'raw': 'signals/ECG/hand/raw',
                             'filtered': 'signals/ECG/hand/filtered/fir5to20',
                             'segments': 'signals/ECG/hand/segments/engzee',
                             'R': 'events/ECG/hand/R',
                             'outlier': 'events/ECG/hand/outlier/dbscan'}
                  }
        st = datamanager.Store(config)
        
        outlier_parameters = {'method': 'dbscan', 
                              'min_samples': 10, 
                              'eps': 0.95, 
                              'metric': 'euclidean'}
        outlier_results = {'info': outlier_parameters}
        for i in xrange(252):
            fd = gzip.open('falc_temp/outlier/dbscan/output-%d'%i, 'rb')
            outlier_results[i] = cPickle.load(fd)
            fd.close()
        outlier_method = 'DBSCAN'
            
        mdata = st.dbmetada()        
        
        data_path = 'falc_temp/segmentation/engzee/output-%d'
        
        output_folder = 'falc_temp/%s/%d-%d-%s.%s'
        
        
        pylab.close('all')
        pylab.ioff()
        
        fig1 = pylab.figure(1, (10,9))
        ax11 = fig1.add_subplot(311)
        ax21 = fig1.add_subplot(312)
        ax31 = fig1.add_subplot(313)
        
        for rid in mdata:
            
            print rid,
            # load
            fd = gzip.open(data_path%rid, 'rb')
            data = scipy.array(cPickle.load(fd)['segments'])
            fd.close()
            # outliers
            idxs = map(lambda i: int(i), misc.merge_clusters(outlier_results[rid])) 
            outliers = outlier_results[rid]
            
            if idxs != []:        
                q = map(lambda i: peakd.sgndiff(-i[:200])['Peak'][-1], data[idxs])
                s = map(lambda i: 200+peakd.sgndiff(-i[200:])['Peak'][0], data[idxs])
                
                fd = gzip.open('falc_temp/qs/%d-qs.dict'%rid, 'wb')
                cPickle.dump({'q': q, 's': s}, fd)
                fd.close()
                
                mn, mx = numpy.min(data[idxs]), numpy.max(data[idxs])
                
                mq, sdq = scipy.mean(q), scipy.std(q)
                ms, sds = scipy.mean(s), scipy.std(s)
                
                
                misc.plot_data(data[idxs], ax21, 'selected')
                ax21.vlines([mq-sdq], mn, mx, 'k', '--')
                ax21.vlines([mq], mn, mx, 'k')
                ax21.vlines([mq+sdq], mn, mx, 'k', '--')
                ax21.vlines([ms-sds], mn, mx, 'k', '--')
                ax21.vlines([ms], mn, mx, 'k')
                ax21.vlines([ms+sds], mn, mx, 'k', '--')
            
            misc.plot_data(data, ax11, 'all')
            ax31.cla()
            ax31.plot(data[outliers['-1']].T, 'r')
            ax31.set_title('outliers')
            ax31.grid()
            ax31.autoscale(enable=True, axis='both', tight=True)
            tlt = '%s-%s-%s'%(mdata[rid]['source'], mdata[rid]['experiment'], outlier_method)
            fig1.suptitle(tlt)
            fig1.savefig(output_folder%('segments', mdata[rid]['subject'], rid, tlt, 'png'))
            
            fig1.clear()
            ax11 = fig1.add_subplot(311)
            ax21 = fig1.add_subplot(312)
            ax31 = fig1.add_subplot(313)
            
    if False: # Q & S position HISTOGRAMS
        
        data_path = 'falc_temp/qs'
        
        
        pylab.ioff()
        fig = pylab.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        for fn in glob.glob(os.path.join(data_path, '*.dict')):
            fd = gzip.open(fn, 'rb')
            data = cPickle.load(fd)
            fd.close()
            try:
                bns_q = range(int(min(data['q'])), int(max(data['q'])))
                bns_s = range(int(min(data['s'])), int(max(data['s'])))    
                fig.suptitle('Histograms')
                ax1.hist(data['q'], bns_q, normed=True, color='k', alpha=0.5)
                ax1.grid()
                ax1.set_title('Q position')
                ax2.hist(data['s'], bns_s, normed=True, color='b', alpha=0.5)
                ax2.set_title('S position', color='b')
                ax2.grid()
            except:
                pass
                
            pylab.savefig(fn.replace('.dict','-histogram.png'))
            
            fig.clear()
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)        
            
    if False: # EXAMPLE OF HOW TO GET SUBGROUP    
        
        config = {'source': 'HDF5',
                  'path': r'\\193.136.222.220\cybh\data\CVP\hdf5',
                  'experiments': ['T1-Sitting', 'T2-Sitting', 'T1-Recumbent', 'T2-Recumbent'],
                  'mapper': {'raw': 'signals/ECG/hand/raw',
                             'filtered': 'signals/ECG/hand/filtered/fir5to20',
                             'segments': 'signals/ECG/hand/segments/engzee',
                             'R': 'events/ECG/hand/R',
                             'outlier': 'events/ECG/hand/outlier/dbscan'}
                  }
        st = datamanager.Store(config)
        
        
        res = misc.get_subgroup_by_tags(st, ['Enfermagem', 'T1', 'Sitting']) # res is  dict  with record id: user id  
        
        
    if False: # Q & S position statistics for all
        
        config = {'source': 'HDF5',
                  'path': r'\\193.136.222.220\cybh\data\CVP\hdf5',
                  'experiments': ['T1-Sitting', 'T2-Sitting', 'T1-Recumbent', 'T2-Recumbent'],
                  'mapper': {'raw': 'signals/ECG/hand/raw',
                             'filtered': 'signals/ECG/hand/filtered/fir5to20',
                             'segments': 'signals/ECG/hand/segments/engzee',
                             'R': 'events/ECG/hand/R',
                             'outlier': 'events/ECG/hand/outlier/dbscan'}
                  }
        st = datamanager.Store(config)
        
        data_path = 'falc_temp/qs/%s-qs.dict'
        
        # all
        mdata = st.dbmetada()
        
        q, s = [], []
        for rid in mdata:
            print rid,
            try:
                fd = gzip.open(data_path%rid, 'rb')
                data = cPickle.load(fd)
                fd.close()
            except:
                pass
            
            q = scipy.hstack((q, data['q']))
            s = scipy.hstack((s, data['s']))
        
        bns_q = range(int(min(q)), int(max(q)))
        bns_s = range(int(min(s)), int(max(s)))
        pylab.ioff()
        fig = pylab.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        fig.suptitle('Histograms')
        ax1.hist(q, bns_q, normed=True, color='k', alpha=0.5, label='u: %0.3f, sd: %0.3f'%(scipy.mean(q), scipy.std(q)))
        ax1.grid()
        ax1.axis('tight')
        ax1.legend(loc=0)
        ax1.set_title('Q position')
        ax2.hist(s, bns_s, normed=True, color='b', alpha=0.5, label='u: %0.3f, sd: %0.3f'%(scipy.mean(s), scipy.std(s)))
        ax2.set_title('S position', color='b')
        ax2.grid()
        ax2.axis('tight')
        ax2.legend(loc=0)
        pylab.savefig(data_path.replace('.dict','.png')%'ALL')

    if False: # Q & S position statistics for subgroup
        
        config = {'source': 'HDF5',
                  'path': r'\\193.136.222.220\cybh\data\CVP\hdf5',
                  'experiments': ['T1-Sitting', 'T2-Sitting', 'T1-Recumbent', 'T2-Recumbent'],
                  'mapper': {'raw': 'signals/ECG/hand/raw',
                             'filtered': 'signals/ECG/hand/filtered/fir5to20',
                             'segments': 'signals/ECG/hand/segments/engzee',
                             'R': 'events/ECG/hand/R',
                             'outlier': 'events/ECG/hand/outlier/dbscan'}
                  }
        st = datamanager.Store(config)
        
        data_path = 'falc_temp/qs/%s-qs.dict'
        
        for exp in ['Enfermagem', 'Fisioterapia', 'Cardiopneumologia']:
            for times in ['T1', 'T2']:
                for post in ['Sitting', 'Recumbent']:
                    
                    tags = [exp, times, post]
                    
                    # all
                    mdata = misc.get_subgroup_by_tags(st, tags)
                    
                    label = '-'.join(i for i in tags)
                    
                    q, s = [], []
                    for rid in mdata:
                        print rid,
                        try:
                            fd = gzip.open(data_path%rid, 'rb')
                            data = cPickle.load(fd)
                            fd.close()
                        except:
                            pass
                        
                        q = scipy.hstack((q, data['q']))
                        s = scipy.hstack((s, data['s']))
                    
                    bns_q = range(int(min(q)), int(max(q)))
                    bns_s = range(int(min(s)), int(max(s)))
                    pylab.ioff()
                    fig = pylab.figure()
                    ax1 = fig.add_subplot(211)
                    ax2 = fig.add_subplot(212)
                    fig.suptitle('Histograms')
                    ax1.hist(q, bns_q, normed=True, color='k', alpha=0.5, label='u: %0.3f, sd: %0.3f'%(scipy.mean(q), scipy.std(q)))
                    ax1.grid()
                    ax1.axis('tight')
                    ax1.set_title('Q position')
                    ax1.legend(loc=0)
                    ax2.hist(s, bns_s, normed=True, color='b', alpha=0.5, label='u: %0.3f, sd: %0.3f'%(scipy.mean(s), scipy.std(s)))
                    ax2.set_title('S position', color='b')
                    ax2.grid()
                    ax2.axis('tight')
                    ax2.legend(loc=0)
                    pylab.savefig(data_path.replace('.dict','.png')%label)    
                    
    if False: # PLOT SEGMENTS, and Q & S histograms
        
        config = {'source': 'HDF5',
                  'path': r'\\193.136.222.220\cybh\data\CVP\hdf5',
                  'experiments': ['T1-Sitting', 'T2-Sitting', 'T1-Recumbent', 'T2-Recumbent'],
                  'mapper': {'raw': 'signals/ECG/hand/raw',
                             'filtered': 'signals/ECG/hand/filtered/fir5to20',
                             'segments': 'signals/ECG/hand/segments/engzee',
                             'R': 'events/ECG/hand/R',
                             'outlier': 'events/ECG/hand/outlier/dbscan'}
                  }
        st = datamanager.Store(config)
        
        outlier_parameters = {'method': 'dbscan', 
                              'min_samples': 10, 
                              'eps': 0.95, 
                              'metric': 'euclidean'}
        outlier_results = {'info': outlier_parameters}
        for i in xrange(252):
            fd = gzip.open('falc_temp/outlier/dbscan/output-%d'%i, 'rb')
            outlier_results[i] = cPickle.load(fd)
            fd.close()
        outlier_method = 'DBSCAN'
            
        mdata = st.dbmetada()        
        
        data_path = 'falc_temp/segmentation/engzee/output-%d'
        
        output_folder = 'falc_temp/%s/%d-%s-%s.%s'
        
        for exp in ['Enfermagem', 'Fisioterapia', 'Cardiopneumologia']:
            for post in ['Sitting', 'Recumbent']:
                info = {
                        'train_set': [exp, 'T1-%s'%post],                              # train set
                        'test_set': [exp, 'T2-%s'%post],                               # test set
                        'train_time': (0, 10),                                   # train time window in seconds
                        'test_time': (0, 10),                                    # test time window in seconds
                        } 
                # get subject info
                subjects = st.subjectTTSets(info) # name, train record ids, test record ids
                
                pylab.close('all')
                pylab.ioff()
                
                fig1 = pylab.figure(1, (16,9))
                ax11 = fig1.add_subplot(221)
                ax12 = fig1.add_subplot(222)
                ax21 = fig1.add_subplot(223)
                ax22 = fig1.add_subplot(224)        
                
                for subject in subjects:
                    for rid, ax, lbl in izip([subject[1][0], subject[2][0]], [(ax11, ax12), (ax21, ax22)], [info['train_set'][1], info['test_set'][1]]):
                    
                        fd = gzip.open(data_path%rid, 'rb')
                        data = scipy.array(cPickle.load(fd)['segments'])
                        fd.close()
                        
                        # outliers
                        idxs = map(lambda i: int(i), misc.merge_clusters(outlier_results[rid])) 
                        outliers = outlier_results[rid]
                        
                        if idxs != []:
                            q = map(lambda i: peakd.sgndiff(-i[:200])['Peak'][-1], data[idxs])
                            s = map(lambda i: 200+peakd.sgndiff(-i[200:])['Peak'][0], data[idxs])
                            
                            mn, mx = numpy.min(data[idxs]), numpy.max(data[idxs])
                            
                            mq, sdq = scipy.mean(q), scipy.std(q)
                            ms, sds = scipy.mean(s), scipy.std(s)
                            
                            misc.plot_data(data[idxs], ax[0], lbl)
                            ax[0].vlines([mq-sdq], mn, mx, 'k', '--')
                            ax[0].vlines([mq], mn, mx, 'k')
                            ax[0].vlines([mq+sdq], mn, mx, 'k', '--')
                            ax[0].vlines([ms-sds], mn, mx, 'k', '--')
                            ax[0].vlines([ms], mn, mx, 'k')
                            ax[0].vlines([ms+sds], mn, mx, 'k', '--')
                        
                            bns_q = range(int(min(q)), int(max(q)))
                            bns_s = range(int(min(s)), int(max(s)))
                            
                            ax[1].hist(q, bns_q, normed=True, color='k', alpha=0.5, label='Q - u: %0.3f, sd: %0.3f'%(scipy.mean(q), scipy.std(q)))
                            ax[1].hist(s, bns_s, normed=True, color='b', alpha=0.5, label='S - u: %0.3f, sd: %0.3f'%(scipy.mean(s), scipy.std(s)))
                            ax[1].grid()
                            ax[1].legend(loc=0)
                            xys = ax[1].axis()
                            ax[1].axis([155, 241, 0, xys[3]])              
                            
                    
                    tlt = '%s-%s'%(mdata[rid]['source'], outlier_method)
                    fig1.savefig(output_folder%('temp', mdata[rid]['subject'], str(subject[1][0])+'-'+str(subject[2][0]), tlt, 'png'))
                    
                    fig1.clear()
                    ax11 = fig1.add_subplot(221)
                    ax12 = fig1.add_subplot(222)
                    ax21 = fig1.add_subplot(223)
                    ax22 = fig1.add_subplot(224)
     
     
    if False: # PLOT mean curve T1 Vs T2
            
        outlier_method = 'rcosdmean'
        
        config = {'source': 'HDF5',
                  'path': r'\\193.136.222.220\cybh\data\CVP\hdf5',
                  'experiments': ['T1-Sitting', 'T2-Sitting', 'T1-Recumbent', 'T2-Recumbent'],
                  'mapper': {'raw': 'signals/ECG/hand/raw',
                             'filtered': 'signals/ECG/hand/filtered/fir5to20',
                             'segments': 'signals/ECG/hand/segments/engzee',
                             'R': 'events/ECG/hand/R',
                             'outlier': 'events/ECG/hand/outlier/dbscan'}
                  }
        st = datamanager.Store(config)
        
        outlier_results = {}
        for i in xrange(252):
            fd = gzip.open('falc_temp/outlier/%s/output-%d'%(outlier_method, i), 'rb')
            outlier_results[i] = cPickle.load(fd)
            fd.close()
            
        mdata = st.dbmetada()        
        
        data_path = 'falc_temp/segmentation/engzee/output-%d'
        
        output_folder = 'falc_temp/%s/%d-%s-%s.%s'                    
        
        for exp in ['Enfermagem', 'Fisioterapia', 'Cardiopneumologia']:
            for post in ['Sitting', 'Recumbent']:
                info = {
                        'train_set': [exp, 'T1-%s'%post],                              # train set
                        'test_set': [exp, 'T2-%s'%post],                               # test set
                        'train_time': (0, 10),                                   # train time window in seconds
                        'test_time': (0, 10),                                    # test time window in seconds
                        } 
                # get subject info
                subjects = st.subjectTTSets(info) # name, train record ids, test record ids
                
                pylab.close('all')
                pylab.ioff()
                
                fig1 = pylab.figure(1, (16,9))
                ax11 = fig1.add_subplot(111)      
                
                for subject in subjects:
                    for rid, color, lbl in izip([subject[1][0], subject[2][0]], ['b', 'g'], [info['train_set'][1], info['test_set'][1]]):
                    
                        fd = gzip.open(data_path%rid, 'rb')
                        data = scipy.array(cPickle.load(fd)['segments'])
                        fd.close()
                        
                        # outliers
                        idxs = map(lambda i: int(i), misc.merge_clusters(outlier_results[rid])) 
                        outliers = outlier_results[rid]
                        
                        if idxs != []: data = data[idxs]
                        
                        #data -= scipy.mean(data)
                        #data /= scipy.median(numpy.max(data, 1))
                        data /= scipy.mean(numpy.max(data, 1))                        
                        
                        if idxs != []:
                            mean_curve = scipy.mean(data,0)
                            ax11.plot(mean_curve, color, linewidth=5., label=lbl)
                    ax11.grid()
                    ax11.autoscale(enable=True, axis='both', tight=True)
                    ax11.legend(loc=0)  
                    
                    tlt = '%s-%s'%(mdata[rid]['source'], outlier_method)
                    fig1.savefig(output_folder%('temp', mdata[rid]['subject'], str(subject[1][0])+'-'+str(subject[2][0]), tlt, 'png'))
                    
                    fig1.clear()
                    ax11 = fig1.add_subplot(111)              
                    
                    
    if False: # determine mean curve of entire population
                
        data_path = r'C:\work\python\Cardioprint\BiometricsPyKit\falc_temp\segmentation\engzee'
        list_files = glob.glob(os.path.join(data_path, 'output-*')) 
        
        population_mean_curve = []
        
        for fn in list_files:
            print fn
            
            fd = gzip.open(fn, 'rb')
            d = cPickle.load(fd)
            fd.close()
            
            segs = scipy.array(d['segments'])
            segs /= scipy.median(numpy.max(segs, 1))
            
            try: population_mean_curve = scipy.vstack((population_mean_curve, segs))
            except: population_mean_curve = segs
            
        mean_curve = scipy.mean(population_mean_curve, 0)
        sd_curve = scipy.std(population_mean_curve, 0)
        
        fd = gzip.open('population_stats', 'wb')
        cPickle.dump({'mean': mean_curve, 'sd': sd_curve}, fd)
        fd.close()
        
        
    if False: # determine mean curve of entire population
                
        data_path = r'C:\work\python\Cardioprint\BiometricsPyKit\falc_temp\segmentation\engzee'
        list_files = glob.glob(os.path.join(data_path, 'output-*'))
        
        res = {}
        
        fig = pylab.figure(1)
        
        for fn in list_files:
            rid = int(fn.split('-')[-1])
            
            print fn
            
            fd = gzip.open(fn, 'rb')
            d = cPickle.load(fd)
            fd.close()
            
            segs = scipy.array(d['segments'])
            
            m = scipy.mean(segs, 0)
            sd = scipy.std(segs, 0)
            
            ssd = sum(sd)
            
            
            res[rid] = ssd
            
            pylab.plot(m, 'k')
            pylab.plot(m+sd, 'k--')
            pylab.plot(m-sd, 'k--')
            pylab.grid()
            pylab.axis('tight')
            pylab.title('%0.3f'%ssd)
            fig.savefig('falc_temp/temp/output-%d.png'%rid)
            pylab.cla()
      
        fd = gzip.open('falc_temp/temp/outliers_cosd_sd', 'wb')
        cPickle.dump(res, fd)
        fd.close()
        
    if False: # QRS M m COSD OUTLIER
                
        data_path = r'C:\work\python\Cardioprint\BiometricsPyKit\falc_temp\segmentation\engzee'
        list_files = glob.glob(os.path.join(data_path, 'output-*'))
        
        fig = pylab.figure(1)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        for fn in list_files:
            
            
            rid = int(fn.split('-')[-1])
            
            print '\n', fn
            
            fd = gzip.open(fn, 'rb')
            d = cPickle.load(fd)
            fd.close()
            
            segs = scipy.array(d['segments'])
            
            lsegs = len(segs)
            
            outliers = []
            alls = set(range(lsegs))
            
            M = numpy.median(numpy.max(segs, 1))*1.5
            m = numpy.median(numpy.min(segs, 1))*1.5
            mean_curve = scipy.mean(segs, 0)
            sd_segs = scipy.std(segs, 0)
            
            cosds = misc.wavedistance(mean_curve, segs, misc.cosdistance)
            
            for i in xrange(lsegs):
                
                R = scipy.argmax(segs[i])
                Q = scipy.argmin(segs[i][:200])
                S = scipy.argmin(segs[i][200:]) + 200
                
                if R != 200:
                    outliers.append(i)
#                elif Q > 170 or Q < 160:
#                    outliers.append(i)
#                elif S > 240 or S < 230:
#                    outliers.append(i)
                elif max(segs[i]) > M:
                    outliers.append(i)
                elif min(segs[i]) < m:
                    outliers.append(i)
                else:
                    #cosd = misc.wavedistance(segs[i], [mean_curve], misc.cosdistance)[0]
                    cosd = cosds[i]
                    th = scipy.mean(cosds)#0.05
                    if cosd > th:
                        outliers.append(i)
                
            goods = list(alls - set(outliers))
            
            if goods != []:
                ax1.plot(segs[goods].T, 'k')
                ax1.axis('tight')
                ax1.grid()
            if outliers != []:
                ax2.plot(segs[outliers].T, 'r')
                ax2.axis('tight')
                ax2.grid()
            fig.savefig('falc_temp/temp/output-%d.png'%rid)
            ax1.cla() 
            ax2.cla()
            
            fd = gzip.open('falc_temp/temp/output-%d'%rid, 'wb')
            cPickle.dump({'0': goods, '-1': outliers}, fd)
            fd.close()        
            
    if False: # R M m mean SD OUTLIER
                
        data_path = r'C:\work\python\Cardioprint\BiometricsPyKit\falc_temp\segmentation\engzee'
        list_files = glob.glob(os.path.join(data_path, 'output-*'))
        
        fig = pylab.figure(1)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        for fn in list_files:

            rid = int(fn.split('-')[-1])
            
            print '\n', fn
            
            fd = gzip.open(fn, 'rb')
            d = cPickle.load(fd)
            fd.close()
            
            segs = scipy.array(d['segments'])
            
            lsegs = len(segs)
            
            outliers = []
            alls = set(range(lsegs))
            
            M = numpy.median(numpy.max(segs, 1))*1.5
            m = numpy.median(numpy.min(segs, 1))*1.5
            mean_curve = scipy.mean(segs, 0)
            sd_segs = scipy.std(segs, 0)
            
            
            for i in xrange(lsegs):
                
                R = scipy.argmax(segs[i])
                
                if R != 200:
                    outliers.append(i)
                elif max(segs[i]) > M:
                    outliers.append(i)
                elif min(segs[i]) < m:
                    outliers.append(i)
                else:
                    up = pylab.find(segs[i] > mean_curve-sd_segs)
                    down = pylab.find(segs[i] < mean_curve+sd_segs)
                    ins = len(scipy.intersect1d(up, down))
                    
                    if ins < 540:
                        outliers.append(i)
                    
            goods = list(alls - set(outliers))
            
            if goods != []:
                ax1.plot(segs[goods].T, 'k')
                ax1.axis('tight')
                ax1.grid()
            if outliers != []:
                ax2.plot(segs[outliers].T, 'r')
                ax2.axis('tight')
                ax2.grid()
            fig.savefig('falc_temp/temp/output-%d.png'%rid)
            ax1.cla() 
            ax2.cla()
            
            fd = gzip.open('falc_temp/temp/output-%d'%rid, 'wb')
            cPickle.dump({'0': goods, '-1': outliers}, fd)
            fd.close()          
            
    if True: # histograms
                
        data_path = r'C:\work\python\Cardioprint\BiometricsPyKit\falc_temp\tests\allT1T2-rmcosd-bigmedianwave\all-5-knn3\cosine'
        
        data_path = r'C:\work\python\Cardioprint\BiometricsPyKit\falc_temp\distances'
        
        
        list_files = glob.glob(os.path.join(data_path, '0-0-*'))
        
        res = {}
        
        for fn in list_files:
            
            print fn
            
            rid1 = fn.split('-')
            rid2 = rid1[-2]
            rid1 = rid1[-1].split('.')[0]
        
            fd = gzip.open(fn, 'rb')
            d = cPickle.load(fd)
            fd.close()
            
            if not res.has_key(rid1): res[rid1] = {'genuine': [], 'imposter': []}
                        
            if rid1 == rid2: label = 'genuine'
            else: label = 'imposter'
                
            for dis in d['distances']:
                res[rid1][label] = scipy.hstack((res[rid1][label], dis))

        fd = gzip.open('falc_temp/temp/hists', 'wb')
        cPickle.dump(res, fd)
        fd.close()
        
        fd = gzip.open('falc_temp/temp/hists', 'rb')
        res = cPickle.load(fd)
        fd.close()    

        fig = pylab.figure(1)
        
        bins = scipy.arange(0, 0.2, 0.001)
        
        zooplot = []
        
        for rid in res:
            try:
                pdfg, _, _ = pylab.hist(res[rid]['genuine'], bins, normed=True, color='g', label='genuine')
                pdfi, _, _ = pylab.hist(res[rid]['imposter'], bins, normed=True, color='r', label='imposter')
                pylab.grid()
                pylab.axis('tight')
                
                M = max(max(pdfg), max(pdfi))
                
                aux = []
                for i in xrange(len(bins)-1):
                    aux.append([sum(pdfg[:i]*numpy.diff(bins[:i+1])), sum(pdfi[:i]*numpy.diff(bins[:i+1]))])
                aux = scipy.array(aux)
                i = scipy.argmax(aux[:,0] - aux[:,1])
                pylab.vlines(bins[i], 0, M, 'k', '--', lw=3)
                pylab.text(bins[i+1], M*0.5, '%0.3f genuine\n%0.3f imposter'%(aux[i][0], aux[i][1]))
                fig.savefig('falc_temp/temp/hist-%s.png'%rid)
                pylab.cla()
                
                zooplot.append([rid, aux[i][0], aux[i][1]])
            except Exception as e:
                print rid, e
                break
        
        zoo = []
        for z in zooplot:
            pylab.plot(z[1], z[2], 'o', label=z[0])
            if z[1] <= 0.05: # low genuine
                if z[2] <= 0.05: # low imposter
                    label = 'phantom'
                elif z[2] >= 0.95: # high imposter
                    label = 'worm'
                else: # middle imposter
                    label = 'worm-phantom'
            elif z[1] >= 0.95: # high genuine
                if z[2] <= 0.05: # low imposter
                    label = 'dove'
                elif z[2] >= 0.95: # high imposter
                    label = 'chameleon'
                else: # middle imposter
                    label = 'dove-chameleon'
            else:   # middle genuine
                if z[2] <= 0.05: # low imposter
                    label = 'dove-phantom'
                elif z[2] >= 0.95: # high imposter
                    label = 'worm-chameleon'
                else: # middle imposter
                    label = 'bit of all'
            zoo.append([label, z[0]])
        
#        pylab.grid()
#        pylab.legend(loc=0)
#        pylab.show()
        zoo = scipy.array(zoo)
        
        file = open('falc_temp/temp/zoo.txt', 'wb')
        for label in ['phantoms', 'doves', 'chameleons', 'worms', 'worm-phantoms', 'dove-chameleons', 'dove-phantoms', 'worm-chameleons', 'bit of alls']:
            n = zoo[:,1][pylab.find(zoo[:,0] == label[:-1])]
            file.write('%s: '%label)
            file.write('%s\n'%(','.join(i for i in n)))
        file.close()


    if False: # % removed
        
        for method in ['dbscan', '480rmmin', 'qrscosd005', 'rcosdmean']:
                
            data_path = r'C:\work\python\Cardioprint\BiometricsPyKit\falc_temp\outlier\%s'%method
            list_files = glob.glob(os.path.join(data_path, 'output-*'))
            
            res = []
            
            for fn in list_files:
                
                if fn.find('png') >= 0: continue
        
                fd = gzip.open(fn, 'rb')
                d = cPickle.load(fd)
                fd.close()
                
                lg = 1.*len(d['0'])
                lb = len(d['-1'])
                
                res.append(lg/(lg+lb))
            
            m = scipy.mean(res)
            sd = scipy.std(res)
            
            print '\n\n%s kept %0.3f +- %0.3f segments'%(method, m, sd)
            
            
              
        
            