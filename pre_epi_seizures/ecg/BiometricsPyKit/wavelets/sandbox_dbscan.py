'''
Created on 26 de Nov de 2012

@author: Carlos
'''

# imports
import os
import glob
import h5py
import json
import numpy as np
import pylab
import matplotlib.pyplot as pl
import matplotlib.patches as patches
import matplotlib.path as mPath
from sklearn.cluster import DBSCAN
from multiprocessing import Process, Queue
from Queue import Empty


def waveDist(coefficients1, coefficients2, thr=1000., mean=False):
    if mean:
        coefficients1 = coefficients1 - coefficients1.mean(axis=0)
        coefficients2 = coefficients2 - coefficients2.mean(axis=0)
    
    # dimensions (levels, nb of coefficients)
    nb, L = coefficients1.shape
    
    # normalization factor
    aux = np.empty(shape=(nb, L, 2))
    aux[:, :, 0] = np.abs(coefficients1)
    aux[:, :, 1] = np.abs(coefficients2)
    nrm = np.max(aux, axis=2)
    nrm[nrm < thr] = thr
    
    # absolute difference weighted by the normalization
    dist = np.abs(coefficients1 - coefficients2) / nrm
    
    dist = np.sum(np.sum(dist))
    
    return dist


def do_work(queue):
    while 1:
        try:
            # get a job
            q = queue.get(timeout=5)
            rec = q['record']
            segs = q['data']
            parameters = q['parameters']
            function = q['function']
            path = q['path']
            
            # distances
            n = len(segs)
            dists = np.zeros((n, n))
            
            for i in xrange(n):
                for j in xrange(i+1, n):
                    dists[i, j] = dists[j, i] = function(segs[i], segs[j], **parameters)
            
            # save
            if not os.path.exists(path):
                os.makedirs(path)
            with h5py.File(os.path.join(path, 'dists','dist_%d.hdf5' % rec), 'w') as fid:
                fid.create_dataset('dists', data=dists)
            
        except Empty:
            break


def computeDistances(path, recPath, records, function, parameters):
    # compute the distances
    
    NUMBER_PROCESSES = 10
    
    # get data
    data = {}
    for rec in records:
        with h5py.File(os.path.join(recPath, 'rec_%d.hdf5' % rec), 'r') as fid:
            data[rec] = fid['signals/ECG/hand/Wavelets/rbio3-3//Segments/4to6/signal0'][...].swapaxes(0, 2).swapaxes(1, 2)
            # data[sub] = fid['signals/ECG/hand/Wavelets/rbio3-3/Reconstruction/Segments/signal0'][...].swapaxes(0, 1)
    
    # compute distances
    work_queue = Queue()
    
    print "Filling work queue ..."
    for rec in data:
        work_queue.put({'record': rec,
                        'data': data[rec],
                        'function': function,
                        'parameters': parameters,
                        'path': path})

    processes = [Process(target=do_work, args=(work_queue,)) for _ in range(NUMBER_PROCESSES)]
    print "Starting processes ..."
    for p in processes: p.start()
    print "Waiting ..."
    for p in processes: p.join()
    print "Done"
    for p in processes: p.terminate()


def dbscan(queue):
    
    while 1:
        try:
            # get a job
            q = queue.get(timeout=5)
            D = q['D']
            min_samples = q['min_samples']
            eps = q['eps']
            imgPath = q['imgPath']
            segs = q['segs']
            
            # dbscan
            S = 1 - (D / np.max(D))
            # Compute DBSCAN
            db = DBSCAN(eps=eps, metric='euclidean', min_samples=min_samples)
            db.fit(S)
            # core_samples = db.core_sample_indices_
            labels = db.labels_
            # Output
            res = {}
            for c in set(labels):
                if c == -1: lbl = '-1'
                else: lbl = '0'
                res[lbl] = list(pylab.find(labels==c))
            
            try:
                bad = res['-1']
            except KeyError:
                bad = []
            
            good = list(set(range(len(segs))) - set(bad))
            
            tgood = 100 * (len(good) / float(len(segs)))
            tbad = 100 * (len(bad) / float(len(segs)))
            
            # plot
            fig = pl.figure(figsize=(16, 9))
            for k in xrange(3):
                ax1 = fig.add_subplot(3, 2, 2*(k+1)-1)
                if len(good) > 0:
                    ax1.plot(segs[good, :, k].T)
                ax2 = fig.add_subplot(3, 2, 2*(k+1))
                if len(bad) > 0:
                    ax2.plot(segs[bad, :, k].T)
                
                if k == 0:
                    ax1.set_title('Retained: %d (%2.2f %%)' % (len(good), tgood))
                    ax2.set_title('Removed: %d (%2.2f %%)' % (len(bad), tbad))
                
                if k == 2:
                    ax1.set_xlabel('Normal')
                    ax2.set_xlabel('Outliers')
            fig.savefig(os.path.join(imgPath), dpi=200, bbox_inches='tight')
            pl.close(fig)
        
        except Empty:
            break
    
    return res


def computeDbscan(path, recPath, records, epsV):
    NUMBER_PROCESSES = 9
    
    # compute distances
    work_queue = Queue()
    
    # fill queue
    print "Filling work queue ..."
    for rec in records:
        with h5py.File(os.path.join(recPath, 'rec_%d.hdf5' % rec), 'r') as fid:
            segs = fid['signals/ECG/hand/Wavelets/rbio3-3//Segments/4to6/signal0'][...].swapaxes(0, 2).swapaxes(1, 2)
        
        # load dists
        with h5py.File(os.path.join(path, 'dists', 'dist_%d.hdf5' % rec), 'r') as fid:
            D = fid['dists'][...]
        
        # dbsacn
        for eps in epsV:
            imgPath = os.path.join(path, 'eps_%2.2f' % eps)
            if not os.path.exists(imgPath):
                os.makedirs(imgPath)
            imgPath = os.path.join(imgPath, 'rec_%d.png' % rec)
            
            work_queue.put({'D': D,
                            'segs': segs,
                            'eps': eps,
                            'imgPath': imgPath,
                            'min_samples': 10})

    processes = [Process(target=dbscan, args=(work_queue,)) for _ in range(NUMBER_PROCESSES)]
    print "Starting processes ..."
    for p in processes: p.start()
    print "Waiting ..."
    for p in processes: p.join()
    print "Done"
    for p in processes: p.terminate()
            


def computeHistograms(path, thrV, nbins):
    # compute the histograms
    
    for thr in thrV:
        mi, ma = [], []
        # first find min and max
        data = []
        for fn in glob.glob(os.path.join(path, 'dist_%4.4f_*.hdf5' % thr)):
            print fn
            
            # subjects
            bits = os.path.split(fn)[1].split('_')
            sub1 = int(bits[2])
            sub2 = int(bits[3].split('.')[0])
            
            # get distances from file
            with h5py.File(fn) as fid:
                dists = fid['dists'][...]
                pairs = fid['pairs'][...]
            
            data.append({'sub1': sub1, 'sub2': sub2, 'dists': dists, 'pairs': pairs, 'fn': fn})
            
            aux = dists.min()
            if aux > 0:
                mi.append(dists.min())
            ma.append(dists.max())
        
        mi = np.log10(min(mi))
        ma = np.log10(max(ma))
        
        # bins for histogram
        bins = np.linspace(mi, ma, nbins)
        
        # now get the distances
        intraD = np.zeros(len(bins) - 1)
        interD = np.zeros(len(bins) - 1)
        
        for item in data:
            print item['fn']
            
            # get data
            dists = item['dists']
            # pairs = item['pairs']
            # a = max([p[0] for p in pairs])
            # b = max([p[1] for p in pairs])
            
            vec = np.log10(dists)
            out = np.histogram(vec, bins)
            if item['sub1'] == item['sub2']:
                intraD += out[0]
            else:
                interD += out[0]
        
        # normalize
        intraD = intraD / intraD.sum()
        interD = interD / interD.sum()
        
        # save
        with h5py.File(os.path.join(path, 'Img', 'hist_%4.4f.hdf5' % thr), 'w') as fid:
            fid.create_dataset('bins', data=bins)
            fid.create_dataset('intraD', data=intraD)
            fid.create_dataset('interD', data=interD)


def plotHistograms(path, thrV):
    
    # load the histograms
    data = []
    mi, ma = [], []
    for thr in thrV:
        with h5py.File(os.path.join(path, 'Img', 'hist_%4.4f.hdf5' % thr), 'r') as fid:
            data.append({'bins': fid['bins'][...], 'intraD': fid['intraD'][...], 'interD': fid['interD'][...], 'thr': thr})
            ma.append(np.max(fid['bins']))
            mi.append(np.min(fid['bins']))
    
    mi = min(mi)
    ma = max(ma)
    
    for item in data:
        
        bins = item['bins']
        intraD = item['intraD']
        interD = item['interD']
        thr = item['thr']
        
        fig = pl.figure()
        ax = fig.add_subplot(111)
        
        # get the corners of the rectangles for the histogram
        left = np.array(bins[:-1])
        right = np.array(bins[1:])
        bottom = np.zeros(len(left))
        intraTop = bottom + intraD
        intraMax = intraD.max()
        interTop = bottom + interD
        interMax = interD.max()
        
        intraXY = np.array([[left,left,right,right], [bottom,intraTop,intraTop,bottom]]).T
        interXY = np.array([[left,left,right,right], [bottom,interTop,interTop,bottom]]).T
        
        # get the Path object
        intraBarpath = mPath.Path.make_compound_path_from_polys(intraXY)
        interBarpath = mPath.Path.make_compound_path_from_polys(interXY)
        
        # make a patch out of it
        intraPatch = patches.PathPatch(intraBarpath, facecolor='blue', edgecolor='blue', alpha=0.8, label='Intra')
        ax.add_patch(intraPatch)
        interPatch = patches.PathPatch(interBarpath, facecolor='red', edgecolor='red', alpha=0.8, label='Inter')
        ax.add_patch(interPatch)
        ax.legend()
        
        # update the view limits
        ax.set_xlim(mi, ma)
        ax.set_ylim(bottom.min(), np.max([intraMax, interMax]))
        ax.set_title('Threshold: %4.4f' % thr)
        
        # save figure
        fig.savefig(os.path.join(path, 'Img', 'hist_%4.4f.png' % thr), dpi=300, bbox_inches='tight')
        pl.close(fig)



if __name__ == '__main__':
    # parameters
    # thrV = [0.0001, 0.001, 0.01, 0.1, 1., 10, 100, 1000, 10000, 100000]
    path = os.path.abspath(os.path.expanduser('~/testWaveletDist/dbscan'))
    recPath = '/home/biomesh/BioMESH/CVP'
    
    # records = [82, 84, 86, 88, 90, 92, 94, 96, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 122, 124, 126, 128, 130]
    records = range(252)
    
    # compute distances
    # computeDistances(path, recPath, records, waveDist, {'mean': False})
    
    # apply dbscan
    epsV = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    computeDbscan(path, recPath, records, epsV)
        
        
    
    
    
