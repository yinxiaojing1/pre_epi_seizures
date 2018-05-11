'''
Created on 20 de Nov de 2012

@author: Carlos
'''

# imports
import os
import glob
import h5py
import json
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.patches as patches
import matplotlib.path as mPath
import scipy.spatial.distance as sd
from multiprocessing import Process, Queue
from Queue import Empty



def do_work(queue):
    while 1:
        try:
            # get a job
            q = queue.get(timeout=5)
            sub = q['subject']
            segs = q['data']
            path = q['path']
            metrics = q['metrics']
            
            # distances
            dists = {}
            for metric in metrics:
                dists[metric] = sd.pdist(segs, metric=metric)
            
            # save
            with h5py.File(os.path.join(path, 'dist_%d_%d.hdf5' % (sub, sub)), 'w') as fid:
                for metric in metrics:
                    fid.create_dataset(metric, data=dists[metric])
            
        except Empty:
            break


def computeDistances(path, recPath, records, metrics):
    # compute the distances
    
    NUMBER_PROCESSES = 10
    
    # get data
    data = {}
    for rec in records:
        with h5py.File(os.path.join(recPath, 'rec_%d.hdf5' % rec), 'r') as fid:
            sub = json.loads(fid.attrs['json'])['subject']
            data[sub] = fid['signals/ECG/hand/Wavelets/rbio3-3/Segments/Reconstruction/signal0'][...].T
    
    # compute distances
    work_queue = Queue()
    
    print "Filling work queue ..."
    for sub in data:
        work_queue.put({'subject': sub,
                        'data': data[sub],
                        'metrics': metrics,
                        'path': path})
    
    processes = [Process(target=do_work, args=(work_queue,)) for _ in range(NUMBER_PROCESSES)]
    print "Starting processes ..."
    for p in processes: p.start()
    print "Waiting ..."
    for p in processes: p.join()
    print "Done"
    for p in processes: p.terminate()


def computeHistograms(path, nbins):
    # compute the histograms
    
    mi, ma = [], []
    # first find min and max
    data = []
    for fn in glob.glob(os.path.join(path, 'dist_*.hdf5')):
        print fn
        
        # subjects
        bits = os.path.split(fn)[1].split('_')
        sub = int(bits[2])
        
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
#            pairs = item['pairs']
#            a = max([p[0] for p in pairs])
#            b = max([p[1] for p in pairs])
        
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
        ax.set_title('Threshold: %4.4f' & thr)
        
        # save figure
        fig.savefig(os.path.join(path, 'Img', 'hist_%4.4f.png' % thr), dpi=250, bbox_inches='tight')
        pl.close(fig)



if __name__ == '__main__':
    # parameters
    # thrV = [0.0001, 0.001, 0.01, 0.1, 1., 10, 100, 1000]
    # path = os.path.abspath(os.path.expanduser('~/testWaveletDist'))
    # recPath = '/home/biomesh/BioMESH/CVP'
    recPath = 'D:\\BioMESH\\Databases\\CVP'
    records = [82, 84, 86, 88, 90, 92, 94, 96, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 122, 124, 126, 128, 130]
    
    # compute distances
    # computeDistances(path, recPath, records, thrV, True)
    
    # compute histograms
    # computeHistograms(path, thrV, 200)
    
    # plot histograms
    # plotHistograms(path, thrV)
    
    ['euclidean', 'cosine', 'correlation']
    
    for rec in records:
        print rec
        with h5py.File(os.path.join(recPath, 'rec_%d.hdf5' % rec)) as fid:
            segs = fid['signals/ECG/hand/Wavelets/rbio3-3/Reconstruction/Segments/signal0'][...]
        
        fig = pl.figure(figsize=(16, 9))
        ax = fig.add_subplot(111)
        
        ax.plot(segs)
        ax.set_title('Record: %d' % rec)
        
        fig.savefig('C:\\Users\\Carlos\\testWaveletDist\\waveletReconstruction\\waveletRec_%d.png' % rec, dpi=300, bbox_inches='tight')
        pl.close(fig)
        
        
    
    
    
