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
    
    
    #----------------------------------------------------------------------------------------------------------------          
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
    #----------------------------------------------------------------------------------------------------------------
    
    #----------------------------------------------------------------------------------------------------------------        
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
    #----------------------------------------------------------------------------------------------------------------
    
    #----------------------------------------------------------------------------------------------------------------
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
    #----------------------------------------------------------------------------------------------------------------     
    
    #----------------------------------------------------------------------------------------------------------------    
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
    #----------------------------------------------------------------------------------------------------------------
    
    #----------------------------------------------------------------------------------------------------------------
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
    #----------------------------------------------------------------------------------------------------------------
    
    #----------------------------------------------------------------------------------------------------------------                
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
    #---------------------------------------------------------------------------------------------------------------- 
    
    #----------------------------------------------------------------------------------------------------------------
    if True: # PLOT mean curves T1 Vs T2 CVP
            
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
            fd = gzip.open('D:/experiments/CVP/data/outlier/%s/output-%d'%(outlier_method, i), 'rb')
            outlier_results[i] = cPickle.load(fd)
            fd.close()
            
        mdata = st.dbmetada()        
        
        data_path = 'D:/experiments/CVP/data/segmentation/engzee/output-%d'
        
        output_folder = 'falc_temp/%s/%d-%s-%s.%s'                    
        
        for exp in ['Enfermagem', 'Fisioterapia', 'Cardiopneumologia']:
            for post in ['Sitting']:#, 'Recumbent']:
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
                ax11 = fig1.add_subplot(211)      
                ax22 = fig1.add_subplot(212)
                
                for subject in subjects:
                    
                    ddiffhr = {}
                    
                    
                    for rid, color, lbl in izip([subject[1][0], subject[2][0]], ['b', 'g'], [info['train_set'][1], info['test_set'][1]]):
                    
                        fd = gzip.open(data_path%rid, 'rb')
                        data = cPickle.load(fd)
                        fd.close()
                        rpeaks = data['R']
                        data = scipy.array(data['segments'])
                        
                        # outliers
                        idxs = map(lambda i: int(i), misc.merge_clusters(outlier_results[rid])) 
                        outliers = outlier_results[rid]
                        
                        if idxs != []: data = data[idxs]
                        
                        #data -= scipy.mean(data)
                        #data /= scipy.median(numpy.max(data, 1))
                        #data /= scipy.mean(numpy.max(data, 1))                        
                        
                        if idxs != []:
                            mean_curve = scipy.mean(data,0)
                            sd_curve = scipy.std(data,0)
                            
                            hr = 60./(scipy.diff(rpeaks)/1000.)
                            hr_m, hr_sd = scipy.mean(hr), scipy.std(hr)
                            
                            p = peakd.sgndiff(mean_curve[:199])['Peak']
                            p = p[scipy.argmax(mean_curve[p])]
                            
                            q = peakd.sgndiff(-mean_curve[:199])['Peak']
                            q = q[scipy.argmin(mean_curve[q])]
                            
                            s = peakd.sgndiff(-mean_curve[201:])['Peak']+201
                            s = s[scipy.argmin(mean_curve[s])]
                            
                            t = peakd.sgndiff(mean_curve[301:])['Peak']+301
                            t = t[scipy.argmax(mean_curve[t])]
                            
                            ddiffhr[rid] = scipy.diff(rpeaks)                            
                        
                            
                            #ax11.plot(mean_curve, color, linewidth=5., label=lbl)
                            if color == 'b':
                                ax11.set_title('record:%s, HR: %d+-%d'%(subject[1][0], hr_m, hr_sd))
                                ax11.plot(mean_curve, linewidth=5., color='k', label=lbl)
                                ax11.plot(mean_curve+sd_curve, 'k--', label=lbl)
                                ax11.plot(mean_curve-sd_curve, 'k--', label=lbl)
                                ax11.vlines([p,q,s,t], min(mean_curve), max(mean_curve), 'r', '--')
                            else:
                                ax22.set_title('record:%s, HR: %d+-%d'%(subject[2][0], hr_m, hr_sd))
                                ax22.plot(mean_curve, linewidth=5., color='k', label=lbl)
                                ax22.plot(mean_curve+sd_curve, 'k--', label=lbl)
                                ax22.plot(mean_curve-sd_curve, 'k--', label=lbl)
                                ax22.vlines([p,q,s,t], min(mean_curve), max(mean_curve), 'r', '--')
                    ax11.grid()
                    ax22.grid()
                    ax11.autoscale(enable=True, axis='both', tight=True)
                    ax22.autoscale(enable=True, axis='both', tight=True)
                    #ax11.legend(loc=0)  
                    
                    tlt = '%s-%s'%(mdata[rid]['source'], outlier_method)
                    fig1.savefig(output_folder%('temp', mdata[rid]['subject'], str(subject[1][0])+'-'+str(subject[2][0]), tlt, 'png'))
                    
                    fig1.clear()
                    ax11 = fig1.add_subplot(211)              
                    ax22 = fig1.add_subplot(212)
                    dfd = gzip.open('falc_temp/%d-diff-r'%mdata[rid]['subject'], 'wb')
                    cPickle.dump(ddiffhr, dfd)
                    dfd.close()
                    
    #----------------------------------------------------------------------------------------------------------------
    
    #----------------------------------------------------------------------------------------------------------------                
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
    #----------------------------------------------------------------------------------------------------------------
    
    #----------------------------------------------------------------------------------------------------------------    
    if False: # mimic mean curves
        data_folder = r'D:\data\mimic2wdb\segmentation\essf'
        data_path = r'D:\data\mimic2wdb\segmentation\essf\%s'
        outlier_path = r'D:\data\mimic2wdb\outlier\rcosdmean-mimicdb\output-%s'
        
        select = numpy.loadtxt('D:/data/mimic2wdb/segmentation/essf/totalT1.txt', str) 
        
        # Check T1 T2 duration      
        records = {}
        list_files = glob.glob(os.path.join(data_folder, '*-*'))
        for fn in list_files:
            nm = fn.split('\\')[-1]
            
            if nm not in select: continue
            
            parts = nm.split('-')
            sub = parts[0]
            duration = parts[2]
            #if duration != '3600': continue
            if not records.has_key(sub): records[sub] = []
            records[sub].append(nm)
        # get subject info
        subjects = []
        for sub in records:
            ls_recds = [sub]
            for r in records[sub]:
                ls_recds.append([r])
            subjects.append(ls_recds)
            #if len(records[sub]) > 1:
                #subjects.append([sub, [records[sub][0]], [records[sub][1]]])   
        pylab.ioff()
        fig = pylab.figure(1, figsize=(16,9))
        #ax1 = fig.add_subplot(211)
        #ax2 = fig.add_subplot(212)
        
        for sub in subjects:
            print sub
            try:
                #if os.path.exists('falc_temp/temp/meancurves/%s.png'%sub[0]): continue
                lrs = len(sub[1:])
                i=1
                for ridf in sub[1:]:
                    rid = ridf[0]
                    ax = fig.add_subplot(lrs, 1, i)
                    i += 1
                #for rid, ax in izip([sub[1][0], sub[2][0]], [ax1, ax2]):
                    
                    fd = gzip.open(outlier_path%rid, 'rb')
                    good_idxs = cPickle.load(fd)['0']
                    fd.close()
                    
                    fd = gzip.open(data_path%rid, 'rb')
                    data = scipy.array(cPickle.load(fd)['segments'])[good_idxs]
                    fd.close()
                
                    m = scipy.mean(data, 0)
                    sd = scipy.std(data, 0)
                
                    ax.plot(m, 'k')
                    ax.plot(m+sd, 'k--')
                    ax.plot(m-sd, 'k--')
                    ax.grid()
                    ax.axis('tight')
                    ax.set_title('record %s'%rid)
                
                fig.suptitle('Subject %s'%sub[0])
                fig.savefig('falc_temp/temp/meancurves/%s.png'%sub[0])   
                fig.clear() 
                #ax1 = fig.add_subplot(211)
                #ax2 = fig.add_subplot(212)
            except Exception as e:
                print sub, e
    #----------------------------------------------------------------------------------------------------------------         
        
    #----------------------------------------------------------------------------------------------------------------    
    if False: # plot mean curves
        
        config = {'source': 'HDF5',
                  'path': r'\\193.136.222.220\cybh\data\CVP\hdf5',
                  'experiments': ['T1-Sitting', 'T2-Sitting'],# 'T1-Recumbent', 'T2-Recumbent'],
                  'mapper': {'raw': 'signals/ECG/hand/raw',}
                  }
        st = datamanager.Store(config)
        info = {
                'train_set': ['T1-Sitting'],                    # train set
                'test_set': ['T2-Sitting'],                               # test set
                'train_time': (0, 10),                                   # train time window in seconds
                'test_time': (0, 10),                                    # test time window in seconds
                }
        # get subject info
        subjects = st.subjectTTSets(info) # name, train record ids, test record ids        
        
        data_path = r'C:\work\python\Cardioprint\BiometricsPyKit\falc_temp\segmentation\engzee\output-%d'
        outlier_path = r'C:\work\python\Cardioprint\BiometricsPyKit\falc_temp\outlier\rcosdmean\output-%d'
        
        
        doves = [30,29,61,35,23,24,27,20,46,47,44,43,40,1,3,2,5,4,6,8,13,12,17,15,32,58,10,59,22,19,54,31,56,51,36,53,52,33,55,37,18,57,50]
        idmyman = [1,3,4,11,13,15,17,18,23,30,31,35,36,40,43,47,50,51,52,54,55,57,58,59,60,61]
        
        pylab.ioff()
        fig = pylab.figure(1)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        for sub in subjects:
            for rid, ax in izip([sub[1][0], sub[2][0]], [ax1, ax2]):
                
                fd = gzip.open(outlier_path%rid, 'rb')
                good_idxs = cPickle.load(fd)['0']
                fd.close()
                
                fd = gzip.open(data_path%rid, 'rb')
                data = scipy.array(cPickle.load(fd)['segments'])[good_idxs]
                fd.close()
            
                m = scipy.mean(data, 0)
                sd = scipy.std(data, 0)
            
                ax.plot(m, 'k')
                ax.plot(m+sd, 'k--')
                ax.plot(m-sd, 'k--')
                ax.grid()
                ax.axis('tight')
                ax.set_title('record %d'%rid)
            
            fig.suptitle('Subject %d'%sub[0])
            fig.savefig('falc_temp/temp/meancurves/%d-%d-%d.png'%(sub[0], sub[1][0], sub[2][0]))   
            if sub[0] in doves:
                fig.savefig('falc_temp/temp/meancurves/doves/%d-%d-%d.png'%(sub[0], sub[1][0], sub[2][0]))
            if sub[0] in idmyman:
                fig.savefig('falc_temp/temp/meancurves/bestid/%d-%d-%d.png'%(sub[0], sub[1][0], sub[2][0]))
            fig.clear() 
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
    #----------------------------------------------------------------------------------------------------------------
    
    #----------------------------------------------------------------------------------------------------------------
    if False: # R M m COSD OUTLIER
                
        #data_path = r'C:\work\python\Cardioprint\BiometricsPyKit\falc_temp\segmentation\engzee'
        #list_files = glob.glob(os.path.join(data_path, 'output-*'))
        
        data_path = r'D:\data\mimic2wdb\segmentation\essf'
        out_path = r'D:\data\mimic2wdb\outlier\rcosdmean-mimicdb\output-%s'
        list_files = glob.glob(os.path.join(data_path, '*-*'))
        
        r_position = 25
        cosd_th = 0.5
        
        pylab.ioff()
        fig = pylab.figure(1)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        for fn in list_files:
            
            try:
            
                rid = int(fn.split('-')[-1])
                rid = fn.split('\\')[-1]
                
                dur = rid.split('-')[-1]
                #if dur != '3600': continue
                #if int(dur) < 3500: continue
                
                if os.path.exists(out_path%rid): continue
                
                print '\n', fn
                
                fd = gzip.open(fn, 'rb')
                d = cPickle.load(fd)
                fd.close()
                
                segs = scipy.array(d['segments'])
                
                # ---------------------------------------
                # mimic2db specific
                good = []
                seg_size = len(segs[0])
                zers = scipy.zeros(seg_size)
                for i, s in enumerate(segs):
                    if sum(s == zers) == seg_size: pass
                    else: good.append(i)
                segs = segs[good]
                # ---------------------------------------
                
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
                    if R != r_position:
                        outliers.append(i)
                    elif max(segs[i]) > M:
                        outliers.append(i)
                    elif min(segs[i]) < m:
                        outliers.append(i)
                    else:
                        #cosd = misc.wavedistance(segs[i], [mean_curve], misc.cosdistance)[0]
                        cosd = cosds[i]
                        th = scipy.mean(cosds)+cosd_th*scipy.std(cosds)
                        if cosd > th:
                            outliers.append(i)
                    
                goods = list(alls - set(outliers))
                
                fd = gzip.open(out_path%rid, 'wb')
                cPickle.dump({'0': goods, '-1': outliers}, fd)
                fd.close()               
                
    #            if goods != []:
    #                ax1.plot(segs[goods].T, 'k')
    #                ax1.axis('tight')
    #                ax1.grid()
    #            if outliers != []:
    #                ax2.plot(segs[outliers].T, 'r')
    #                ax2.axis('tight')
    #                ax2.grid()
    #            fig.savefig(out_path%rid+'.png')
    #            ax1.cla() 
    #            ax2.cla()
            except Exception as e:
                print e
     
            
    if False: # R M m inside SD OUTLIER
                
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
            
    if False: # histograms and zoo plot
        data_path =  r'C:\work\python\Cardioprint\BiometricsPyKit\falc_temp\tests\allT1T2-rmcosd\5-5-meanwaves\cosine'
        
        data_path =  'C:/work/python/Cardioprint/BiometricsPyKit/falc_temp/distances/'
        
        data_path =  'D:/data/mimic2wdb/distances/'
        
        
        data_path =  'D:/experiments/CVP/Cardiopneumologia+Enfermagem+Fisioterapia/T1(5-median)-T2(5-median)-cosd-cosine/distances/'
        
        list_files = glob.glob(os.path.join(data_path, '0-0-*'))
        
        os.chdir(data_path)
        
        if not os.path.exists('hists'): os.mkdir('hists')
        
        res = {}
                
        print "loading"
        fd = gzip.open('0-0-all.all', 'rb')
        alld = cPickle.load(fd)
        fd.close()
        
        for test_user in alld[0]:
            print test_user 
            for train_user in alld[0][test_user][0]:
                if not res.has_key(test_user): res[test_user] = {'genuine': [], 'imposter': []}
                            
                if test_user == train_user: label = 'genuine'
                else: label = 'imposter'
                    
                #if not all(alld[0][test_user][0][train_user]['distances'] == scipy.Inf):
                if True:
                    for dis in alld[0][test_user][0][train_user]['distances']:
                        res[test_user][label] = scipy.hstack((res[test_user][label], dis))        
        
        #        i=0
        #        for fn in list_files:
        #            if numpy.mod(i,1000) == 0: print '%d '%i,
        #            i += 1
        #            
        #            rid1 = fn.split('-')
        #            rid2 = rid1[-2]
        #            rid1 = rid1[-1].split('.')[0]
        #        
        #            fd = gzip.open(fn, 'rb')
        #            d = cPickle.load(fd)
        #            fd.close()
        #            
        #            if not res.has_key(rid1): res[rid1] = {'genuine': [], 'imposter': []}
        #                        
        #            if rid1 == rid2: label = 'genuine'
        #            else: label = 'imposter'
        #                
        #            for dis in d['distances']:
        #                res[rid1][label] = scipy.hstack((res[rid1][label], dis))
        
        fd = gzip.open('hists/hists', 'wb')
        cPickle.dump(res, fd)
        fd.close()
                
                #fd = gzip.open('hists/hists', 'rb')
                #res = cPickle.load(fd)
                #fd.close()    
        
        pylab.ioff()
        fig = pylab.figure(1)
        
        bins = scipy.arange(0, 1.0, 0.005)
        
        zooplot = []
        
        for rid in res:
            try:
                if res[rid]['genuine'] != []: pdfg, _, _ = pylab.hist(res[rid]['genuine'], bins, normed=True, color='g', alpha=0.5, label='genuine')
                if res[rid]['imposter'] != []: pdfi, _, _ = pylab.hist(res[rid]['imposter'], bins, normed=True, color='r', alpha=0.5, label='imposter')
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
                fig.savefig('hists/hist-%s.png'%rid)
                pylab.cla()
                
                zooplot.append([rid, aux[i][0], aux[i][1]])
            except Exception as e:
                print rid, e
        
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
        
        fn = open('hists/zoo.txt', 'wb')
        for label in ['phantoms', 'doves', 'chameleons', 'worms', 'worm-phantoms', 'dove-chameleons', 'dove-phantoms', 'worm-chameleons', 'bit of alls']:
            n = zoo[:,1][pylab.find(zoo[:,0] == label[:-1])]
            fn.write('%s: '%label)
            fn.write('%s\n'%(','.join(i for i in n)))
        fn.close()


    if False: # outlier detection method: % of data remaining
        
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
            
              
    if False: # COSINE DISTANCE HISTOGRAMS
                
        data_path = r'C:\work\python\Cardioprint\BiometricsPyKit\falc_temp\segmentation\engzee'
        list_files = glob.glob(os.path.join(data_path, 'output-*'))
        
        fig = pylab.figure(1)
        ax1 = fig.add_subplot(211)
        
        cosd_hist = []
        
        for fn in list_files:
            
            rid = int(fn.split('-')[-1])
            
            print '\n', fn
            
            fd = gzip.open(fn, 'rb')
            d = cPickle.load(fd)
            fd.close()
            
            segs = scipy.array(d['segments'])
            mean_curve = scipy.mean(segs, 0)
            cosds = misc.wavedistance(mean_curve, segs, misc.cosdistance)
            
            for i in cosds:
                cosd_hist = scipy.hstack((cosd_hist, i)) 
                
        fd = gzip.open('falc_temp/temp/cosdhist', 'wb')
        cPickle.dump({'dis': cosd_hist}, fd)
        fd.close()
        
    if False:   # select best users for identification
        from evaluation import evaluation
        
        dpath = r'C:\work\python\Cardioprint\BiometricsPyKit\falc_temp\tests\allT1T2-rmcosd\5-5-meanwaves\cosine'
        
        dpath = 'C:/work/python/Cardioprint/BiometricsPyKit/falc_temp/distances/'
        
        os.chdir(dpath)
        if not os.path.exists('idscores'): os.mkdir('idscores')
           
        rejection_thresholds = scipy.arange(0.0, 0.1, 0.001)
        results = {0: {}}
        for fn in glob.glob(os.path.join(dpath, '*.dict')):
            print fn,
            fd = gzip.open(fn, 'rb')
            d = cPickle.load(fd)
            fd.close()
            test_user = int(fn.split('-')[-2])
            train_user = int(fn.split('-')[-1].replace('.dict', ''))
        
            out = {
                   'distances': d['distances'],
                   'labels': d['labels'], 
                   'true_labels': d['true_labels'],
                   'rejection_thresholds': rejection_thresholds}
            try: results[0][test_user][0][train_user] = out
            except: results[0][test_user]= {0: {train_user: out}}
        
        user_perf = []
        pylab.ioff()
        pylab.figure(1)
        for i in results[0].keys():
            res = evaluation.identification_results(results, [i])
            
            EERi = scipy.argmin(abs(res['FAR']-res['FRR']))
            
            user_perf.append([rejection_thresholds[EERi], res['FAR'][EERi], i])
            
            pylab.cla()
            pylab.plot(rejection_thresholds, res['FAR'], label='FAR')
            pylab.plot(rejection_thresholds, res['FRR'], label='FRR')
            pylab.vlines(rejection_thresholds[EERi], 0, 1, 'r')
            pylab.text(rejection_thresholds[EERi], 0.5, '%0.3f'%res['FAR'][EERi])
            pylab.xlabel('Threshold')
            pylab.grid()
            pylab.axis([-0.001, 1.01, -0.01, 1.01])
            pylab.legend()
            pylab.savefig('idscores/%d.png'%i)
        
        pylab.cla()
        for i in user_perf:
            pylab.plot(i[0], i[1], 'o')
            pylab.text(i[0], i[1], '%d'%i[2])
        pylab.xlabel('Threshold')
        pylab.ylabel('EER')
        pylab.title('EER per user')
        pylab.grid()
        pylab.savefig('idscores/all.png')
        
        user_perf = scipy.array(user_perf)
        restrict = []
        for i in user_perf[:,2][pylab.find(user_perf[:,1] <= 0.05)]:
            restrict.append(int(i))
            
        fn = open('idscores/restrict.txt', 'wb')
        fn.write('%s\n'%restrict)
        fn.close()
            
        # a = array(user_perf)
        # users = find(a[:,1] <= 0.02)
        # res = evaluation.authentication_results(results, users)
        # pylab.figure()
        # pylab.plot(rejection_thresholds, res['FAR'], label='FAR')
        # pylab.plot(rejection_thresholds, res['FRR'], label='FRR')
        # pylab.vlines(rejection_thresholds[EERi], 0, 1, 'r')
        # pylab.text(rejection_thresholds[EERi], 0.5, '%0.3f'%res['FAR'][EERi])
        # pylab.xlabel('Threshold')
        # pylab.grid()
        # pylab.axis([-0.01, rejection_thresholds[-1], -0.01, 1.01])
        # pylab.legend()
        # #pylab.savefig('aut-best-82%users.png')
        
    if False: # COSINE DISTANCE to self threshold 
                
        data_path = r'C:\work\python\Cardioprint\BiometricsPyKit\falc_temp\segmentation\engzee'
        list_files = glob.glob(os.path.join(data_path, 'output-*'))
        
        for fn in list_files:
            
            rid = int(fn.split('-')[-1])
            
            print '.',
            
            fd = gzip.open(fn, 'rb')
            d = cPickle.load(fd)
            fd.close()
            
            segs = scipy.array(d['segments'])
            
            dis = []
            for s in segs:
                cosd = misc.wavedistance(s, segs, misc.cosdistance)
                cosd.sort()
                dis = scipy.hstack((dis, cosd[0]))
        print scipy.mean(dis), scipy.std(dis), min(dis), max(dis) 
            
            
    if False: # Failure to enroll
        
        
        config = {'source': 'HDF5',
                  'path': r'\\193.136.222.220\cybh\data\CVP\hdf5',
                  'experiments': ['T1-Sitting', 'T2-Sitting'],
                  'mapper': {'raw': 'signals/ECG/hand/raw',
                             'filtered': 'signals/ECG/hand/filtered/fir5to20',
                             'segments': 'signals/ECG/hand/segments/engzee',
                             'R': 'events/ECG/hand/R',
                             'outlier': 'events/ECG/hand/outlier/dbscan'}
                  }
        st = datamanager.Store(config)   
        info = {
                #'train_set': ['Enfermagem', config['experiments'][0]],                              # train set
                #'test_set': ['Enfermagem', config['experiments'][1]],                               # test set
                'train_set': ['T1-Sitting'],                              # train set
                'test_set': ['T2-Sitting'],                               # test set
                'train_time': (0, 10),                                   # train time window in seconds
                'test_time': (0, 10),                                    # test time window in seconds
                }         
        subjects = st.subjectTTSets(info)
        
        print len(subjects)
        
        data_path_segs = r'C:\work\python\Cardioprint\BiometricsPyKit\falc_temp\segmentation\engzee\output-%d'
        data_path_outs = r'C:\work\python\Cardioprint\BiometricsPyKit\falc_temp\outlier\rcosdmean\output-%d'
        
        failed2enroll = []
        
        sampling_rate = 1000.
        number_segs_to_enroll = 15
        
        m = []
        
        h_g = []
        h_b = []
        
        for s in subjects:
            fd = gzip.open(data_path_segs%s[1][0], 'rb')
            data = cPickle.load(fd)
            fd.close()
            
            
            
            fd = gzip.open(data_path_outs%s[1][0], 'rb')
            idxs = cPickle.load(fd)
            fd.close()
            
            good_ones = idxs['0']
            bad_ones = idxs['-1']
            
            h_b = scipy.hstack((h_b, data['R'][bad_ones]/1000.))
            h_g = scipy.hstack((h_g, data['R'][good_ones]/1000.))
            
#            m.append([len(data['R'][good_ones]), len(data['R']), data['R'][-1]])
#            
#            a = pylab.find(data['R'][good_ones] <= info['train_time'][1]*sampling_rate)
#            b = pylab.find(data['R'][good_ones] >= info['train_time'][0]*sampling_rate)
#            
#            if len(scipy.intersect1d(a,b)) < number_segs_to_enroll:
#                failed2enroll.append(s[0])            
            
        tlt = '%0.3f %s of data before 30 seconds'
        pdf, bns,_ = pylab.hist(h_b, scipy.arange(0, max(h_b), 5), normed=True, color='k')
        d = scipy.sum(pdf[:7]*scipy.diff(bns[:8]))
        pylab.xlabel('R position (sec)')
        pylab.title('Before outlier removal')
        pylab.text(80, 0.015, tlt%(d*100., '%'))
        pylab.grid()
        pylab.show()
        
        pdf, bns,_ = pylab.hist(h_g, scipy.arange(0, max(h_g), 5), normed=True, color='k')
        d = scipy.sum(pdf[:7]*scipy.diff(bns[:8]))
        pylab.xlabel('R position (sec)')
        pylab.title('After outlier removal')
        pylab.text(80, 0.0095, tlt%(d*100., '%'))
        pylab.grid()
        pylab.show()        
        
        
#        print "Failure to enroll: %0.3f"%(1.*len(failed2enroll)/len(subjects))
#        print "%d users failed to enroll (number segments < %s in %d seconds of data)"%(len(failed2enroll), number_segs_to_enroll, info['train_time'][1]-info['train_time'][0])
#        print "users: %s"%failed2enroll     

#        m.sort()
#        print "\n\n"
#        for i in m:
#            print i
#            
#        m=scipy.array(m)
#        
#        print scipy.mean(m[:,0]), scipy.std(m[:,0])
#        
#        print scipy.mean(m[:,1]), scipy.std(m[:,1])
#        
#        print scipy.mean(m[:,2]), scipy.std(m[:,2])
