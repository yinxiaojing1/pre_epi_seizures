"""
.. module:: visualization
   :platform: Unix, Windows
   :synopsis: This module provides various functions to...

.. moduleauthor:: Carlos Carreiras, Andre Lourenco


"""

# Imports
from itertools import cycle
import copy
import csv
import numpy as np
import scipy
import pylab
from matplotlib import cm
import matplotlib.pyplot as plt



def selector(method):
    # given a string, return the correct function
    
    if method == 'rocCurves':
        return rocCurves
    elif method == 'FARFRR':
        return FARFRRCurves
    elif method == 'plotSegments':
        return plotSegments
    elif method == 'plotOutliers':
        return plotOutliers
    elif method == 'clusterOverTime':
        return clusterOverTime
    elif method == 'confusionMatrix':
        return confusionMatrix
    else:
        raise ValueError, "Unknown visualization method (%s)." % method


def figure(*args, **kwargs):
    return plt.figure(*args, **kwargs)


def close(fig=None):
    if fig is None:
        plt.close()
    else:
        plt.close(fig)


def show():
    plt.show()


def buildCSVTable(filePath, resultsCollection):
    # built a CSV table from the collection of results
    
    fid = open(filePath, 'wb')
    writer = csv.writer(fid)
    
    # write header
    writer.writerows(resultsCollection['header'])
    
    # write rows
    writer.writerows(resultsCollection['rows'])
    
    fid.close()


def buildLatexTable(resultsCollection, keyOrder):
    pass


def rocCurves(results, outfile=None, figNum=None, label=None):
    #results is a dictionary with keys ['FAR'] and ['TPR'] corresponding to lists of values
    
    #to enable sobreposition of ROCs
    if figNum: pylab.figure()
    else: pylab.figure(figNum)
        
    if label: pylab.plot(results['FAR'], results['TPR'])
    else: pylab.plot(results['FAR'], results['TPR'], label=str)
    pylab.grid()
    
    if outfile: pylab.savefig(outfile), pylab.close()
    else: pylab.show() 


def addFARFRR(ax, thresholds, rates, lw=1, colors=None, alpha=1, drawEER=False, labels=False):
    # add a FAR-FRR curve to axes
    
    if colors is None:
        colors = ['b', 'g', 'r']
    
    if labels:
        # FAR
        ax.plot(thresholds, rates['FAR'], colors[0], lw=lw, alpha=alpha, label='FAR')
        # FRR
        ax.plot(thresholds, rates['FRR'], colors[1], lw=lw, alpha=alpha, label='FRR')
    else:
        # FAR
        ax.plot(thresholds, rates['FAR'], colors[0], lw=lw, alpha=alpha)
        # FRR
        ax.plot(thresholds, rates['FRR'], colors[1], lw=lw, alpha=alpha)
    
    # EER
    if drawEER:
        EER = rates['EER']
        ax.vlines(EER[0, 0], 0, 1, colors[2], lw=lw)
        ax.set_title('EER = %0.2f %%' % (100. * EER[0, 1]))
        # ax.text(EER[0, 0], 0.9, ' EER = %0.2f %%' % (100. * EER[0, 1]))


def superFARFRRCurves(results, figsize=None):
    # plot FAR-FRR curves and EER for global results and for each subject
    
    fig = pylab.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    ths = results['rejection_thresholds']
    
    # subject results
    colors = ['#008bff', '#8dd000', 'r']
    for s in results['subject'].iterkeys():
        addFARFRR(ax, ths, results['subject'][s]['authentication']['rates'], lw=1, alpha=0.4, colors=colors)
    
    # global results
    colors = ['#0037ff', 'g', 'r']
    addFARFRR(ax, ths, results['global']['authentication']['rates'], lw=3, drawEER=True, labels=True, colors=colors)
    
    # legend and stuff
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Rate')
    ax.grid()
    ax.legend()
    
    return fig


def addEERCurve(ax, thresholds, rates, variables, lw=1, colors=None, alpha=1, drawEER=False, labels=False, scale=1.):
    # add a EER curve to axes
    
    if colors is None:
        colors = ['b', 'g', 'r']
    
    if labels:
        # FAR
        ax.plot(thresholds, rates[variables[0]], colors[0], lw=lw, alpha=alpha, label=variables[0])
        # FRR
        ax.plot(thresholds, rates[variables[1]], colors[1], lw=lw, alpha=alpha, label=variables[1])
    else:
        # FAR
        ax.plot(thresholds, rates[variables[0]], colors[0], lw=lw, alpha=alpha)
        # FRR
        ax.plot(thresholds, rates[variables[1]], colors[1], lw=lw, alpha=alpha)
    
    # EER
    if drawEER:
        EER = rates['EER']
        ax.vlines(EER[0, 0], 0, scale, colors[2], lw=lw)
        ax.set_title('EER = %0.2f %%' % (100. * EER[0, 1]))


def EERCurves(results, idx, figsize=None):
    # plot authentication and identification EER curves
    
    ths = results['rejection_thresholds']
    
    fig = pylab.figure(figsize=figsize)
    colorsS = ['#008bff', '#8dd000', 'r']
    colorsG = ['#0037ff', 'g', 'r']
    
    # authentication
    ax = fig.add_subplot(121)
    
    # subject results
    for s in results['subject'].iterkeys():
        addEERCurve(ax, ths, results['subject'][s]['authentication']['rates'], ('FAR', 'FRR'), lw=1, alpha=0.4, colors=colorsS)
    
    # global results
    addEERCurve(ax, ths, results['global']['authentication']['rates'], ('FAR', 'FRR'), lw=3, drawEER=True, labels=True, colors=colorsG)
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Authentication')
    ax.grid()
    ax.legend()
    
    # identification
    ax = fig.add_subplot(122)
    # subject results
    for s in results['subject'].iterkeys():
        addEERCurve(ax, ths, results['subject'][s]['identification']['rates'], ('MR', 'RR'), lw=1, alpha=0.4, colors=colorsS)
    
    # global results
    addEERCurve(ax, ths, results['global']['identification']['rates'], ('MR', 'RR'), lw=3, drawEER=True, labels=True, colors=colorsG)
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Identification')
    ax.grid()
    ax.legend()
    
    fig.tight_layout()
    
    return fig


def singleFARFRRCurve(thresholds, rates, figsize=None):
    # plot a single FAR-FRR curve and EER
    
    fig = pylab.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    addFARFRR(ax, thresholds, rates, lw=2, drawEER=True, labels=True)
    
    # legend and stuff
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Rate')
    ax.grid()
    ax.legend()
    
    return fig


def FARFRRCurves(ths, results, figsize=None):
    # plot FAR-FRR curves, and EER
    
    fig = pylab.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    EERi = scipy.argmin(abs(results['FAR']-results['FRR']))
    ax.plot(ths, results['FAR'], label='FAR')
    ax.plot(ths, results['FRR'], label='FRR')
    ax.vlines(ths[EERi], 0, 1, 'r')
    ax.text(ths[EERi], 0.5, '%0.3f'%results['FAR'][EERi])
    ax.set_xlabel('Threshold')
    ax.grid()
    ax.legend()
    
    return fig


def plotClusters(data, clusters, figsize=None, alpha=1):
    """
    
    Plot a clustering partition.
    
    Kwargs:
        data (array): The (2D) feature array; each line corresponds to a sample.
        
        clusters (dict): The data partition.
        
        figsize (tuple): The size of the figure to create.
        
        alpha (float): The transparency of the plot lines.
    
    Kwrvals:
        fig (matplotlib.figure.Figure): The figure object.
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1]
        
    """
    
    fig = pylab.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    ks = clusters.keys()
    ks.sort()
    nb = len(ks)
    
    colors = cm.get_cmap('jet', nb)
    
    handles = []
    for k, cl in enumerate(ks):
        h = ax.plot(data[clusters[cl]].T, color=colors(k), alpha=alpha)
        handles.append(h[0])
    
    ax.grid()
    ax.set_title('Number of clusters: %s' % nb)
    
    if nb > 4:
        ncol = int(np.floor(np.sqrt(nb)))
    else:
        ncol = 1
    
    ax.legend(handles, ks, ncol=ncol, loc='best')
    
    return fig


def plotSegments(data, labels, linewidth=2, legenda=None, figureNumber=1, figsize=None):
#labels represent the class labels [1xn]
    if type(labels) is dict:
        ks = labels["clusters"].keys()
        n = 0
        for k in ks:
            n += len(labels["clusters"][k])
        c = np.zeros(n)
        for k in ks:
            c[labels["clusters"][k]] = k
        labels = c

    fig = pylab.figure(figureNumber, figsize=figsize)
    fig.clf()
    ax = fig.add_subplot(111)
    labels_unique = np.unique(labels)   #also sorts the output (so if -1 in the vector, it will be the first value)
    n_clusters_ = len(labels_unique)
    
    l = []
    if legenda is None:
        for k in xrange(n_clusters_):
            l.append(str(k))
    else:
        l = legenda
    
    colors = cm.get_cmap('jet', n_clusters_)
    
    for k in xrange(n_clusters_):
        my_members = labels == labels_unique[k]
        ax.plot(data[my_members].T, color=colors(k), linewidth=linewidth, alpha=0.7)
        
    ax.set_title('Number of clusters: %d' % n_clusters_)
    nl = len(l)
    if nl > 5:
        ncol = int(np.floor(np.sqrt(nl)))
        ax.legend(l, ncol=ncol, loc='best')
    else:
        ax.legend(l, loc='best')
    ax.grid()
    
    return fig


def plotOutliers(data, partition, alpha=0.7, figsize=None):
    # plot outliers
    
    nb = len(data)
    nn = len(partition['0'])
    no = nb - nn
    
    fig = plt.figure(figsize=figsize)
    
    # good segs
    ax = fig.add_subplot(211)
    if nn > 0:
        ax.plot(data[partition['0']].T, 'k', alpha=alpha)
    ax.grid()
    ax.axis('tight')
    ax.set_ylabel('%d of %d' % (nn, nb))
    
    # bad segs
    ax = fig.add_subplot(212)
    if no > 0:
        ax.plot(data[partition['-1']].T, 'r', alpha=alpha)
    ax.grid()
    ax.axis('tight')
    ax.set_ylabel('%d of %d' % (no, nb))
    
    return fig

    
def plotMeanWaves(data,labels=None,figureNumber=1):
#can also be used without outliers (if labels are provided with -1)
    
    
    n=len(data)
    
    if labels!= None:
        outliers = pylab.find(labels==-1)
        notoutliers =  list( set(range(n)) - set(outliers))
    else:
        notoutliers = range(n)
    
    fig=pylab.figure(figureNumber).clf()
    ax = fig.add_subplot(111)
    c=[0, 0, 0]#[.75,.75,.75]
    l=2.
    
    #alternativa minimalista:
    #map(lambda i:pl.plot(x[i,:],color=c,alpha=.45,linewidth=l),pl.arange(0,pl.shape(x)[0],1))
    
    for i in range(len(data)):
        if i in notoutliers:
            #h2, = ax.plot(data[i], color=c,alpha=.15,linewidth=l)
            ax.plot(data[i], color=c,alpha=.15,linewidth=l)
        else:
            ax.plot(data[i], color=[1, 0, 0],alpha=.15,linewidth=l)
            
    pylab.xlabel('t [ms]')
    l=2.
    c=[0,0,0]
    ax.plot(pylab.mean(data[notoutliers],0)-pylab.std(data[notoutliers],0),'--',color=c, linewidth=l)
    ax.plot(pylab.mean(data[notoutliers],0)+pylab.std(data[notoutliers],0),'--',color=c, linewidth=l)
    ax.plot(pylab.mean(data[notoutliers],0),'k', linewidth=5.)
    pylab.grid()
    handles, labels = ax.get_legend_handles_labels()
    #pylab.axis('off')
    pylab.axis('tight')
    #stri=datapath+"ECGMosaic-u"+str(inc) #+ ".pdf"
    #pylab.savefig(stri)
    #font = "sans-serif"
    # pl.text(100, 200,  "user "+str(inc), ha= "center", family=font, size=14)

def clusterOverTime(clusterlabels, linelabels=None, xlabel=None, ylabel=None, ncols=20, figureNumber=1, figsize=None):
# this function enables the visualization of clusterings over time; it displays the clustering information in terms of a matrix
   
    if type(clusterlabels) is dict:
        ks = clusterlabels["clusters"].keys()
        n = 0
        for k in ks:
            n += len(clusterlabels["clusters"][k])
        c = np.zeros(n)
        for k in ks:
            c[clusterlabels["clusters"][k]] = k
        clusterlabels = c
    
    n = len(clusterlabels)
    clusterlabels_unique = np.unique(clusterlabels)   #also sorts the output (so if -1 in the vector, it will be the first value)
    n_clusters_ = len(clusterlabels_unique)
    base = clusterlabels_unique[0] - 1
    
    if linelabels is None:
        # nlines = n / ncols
        nlines = int(np.ceil((float(n)) / ncols))
        linelabels = np.zeros(n)
        for i in range(nlines):
            linelabels[i*ncols:(i+1)*ncols] = i
    
    linelabels_unique = np.unique(linelabels)
    nlines = len(linelabels_unique)
    clusterPerline = np.zeros(len(linelabels_unique))
    for i,k in enumerate(linelabels_unique):
        clusterPerline[i] = len(pylab.find(linelabels == k))
    # print clusterPerline
    ncolumns = max(clusterPerline)
    m = base * np.ones((nlines, ncolumns))
    # print "("+str(nlines)+','+str(ncolumns)+")" 
    count = 0
    for i,k in enumerate(linelabels_unique):
        # print count
        m[i,:clusterPerline[i]] = clusterlabels[count:count+clusterPerline[i]]
        count += clusterPerline[i]
    
    # plot
    fig = pylab.figure(figureNumber, figsize=figsize)
    fig.clf()
    ax = fig.add_subplot(111)     
    
    # color map
    cmap = cm.get_cmap('jet', n_clusters_ + 1)
    mat = ax.matshow(m, cmap=cmap)
    
    # color bar
    step = float((clusterlabels_unique[-1] - base)) / (n_clusters_ + 1)
    ticks = np.arange(base+(step/2), clusterlabels_unique[-1]+(step/2), step)
    ticklabels = ['Bck']
    ticklabels.extend([str(int(item)) for item in clusterlabels_unique])
    
    cbar = pylab.colorbar(mat, orientation='horizontal', ticks=ticks)
    cbar.set_ticklabels(ticklabels)
    
    # xlabel
    if xlabel:
        ax.set_xlabel(xlabel)
    
    # ylabel
    if ylabel:
        ax.set_ylabel(ylabel)
    
    return fig

def confusionMatrix(confMatrix):
    #http://stackoverflow.com/questions/5821125/how-to-plot-confusion-matrix-with-string-axis-rather-than-integer-in-python
    pass


def multiBarPLot(data, xticklabels, yerr=None, labels=None,
                 xlabel=None, ylabel=None, xlim=None, ylim=None,
                 width=0.15, loc='best', legendAnchor=None, figsize=None, rotation=30,
                 xtickssize=12, vlines=False, btext=False, btextsize=12, xgap=3.):
    
    # font parameters
    font = {'family': 'Bitstream Vera Sans',
            'weight': 'normal',
            'size': 16}
    plt.rc('font', **font)
    
    # make figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # fill axis
    multiBarPLotAxis(ax, data, xticklabels, yerr=yerr, labels=labels,
                     xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim,
                     width=width, loc=loc, legendAnchor=legendAnchor, rotation=rotation,
                     xtickssize=xtickssize, vlines=vlines, btext=btext, btextsize=btextsize, xgap=xgap)
    
    return fig


def multiBarPLotAxis(ax, data, xticklabels, yerr=None, labels=None,
                     xlabel=None, ylabel=None, xlim=None, ylim=None,
                     width=0.15, loc='best', legendAnchor=None, rotation=30,
                     xtickssize=12, vlines=False, btext=False, btextsize=12, xgap=3.):
    
    # fill colors and styles
    colors = cycle(['blue', 'yellow', 'cyan', 'green', 'red', 'gray', 'magenta'])
    hatches = cycle(['', '//', '\\', 'x', '++', 'o', 'O', '-', '||', '.', '*'])
    
    # x indexes
    width = float(width)
    # gap = width / 2.
    gap = xgap * width
    ns = len(xticklabels)
    nb = len(data)
    ind = width * np.arange(ns * nb).reshape((ns, nb)).T + gap * np.arange(ns)
    
    # bars
    rects = []
    if yerr is None:
        for i in xrange(nb):
            out = ax.bar(ind[i], data[i], width, color=colors.next(), hatch=hatches.next())
            rects.append(out[0])
    else:
        for i in xrange(nb):
            c = colors.next()
            out = ax.bar(ind[i], data[i], width, yerr=yerr[i], color=c, ecolor=c, hatch=hatches.next())
            rects.append(out[0])
    
    # values text
    if btext:
        for i in xrange(nb):
            for j in xrange(ns):
                ax.text(ind[i][j], data[i][j], '%2.2f' % data[i][j],
                        horizontalalignment='left', verticalalignment='bottom',
                        fontsize=btextsize)
    
    # set xticks
    x = (nb/2.) * width + ind[0, :]
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
        label.set_size(xtickssize)
        label.set_horizontalalignment('right')
        label.set_verticalalignment('top')
    
    # plot dividing lines
    if vlines:
        xv = ind[-1, :] + width + gap / 2.
        for p in xv[:-1]:
            ax.axvline(p, color='0.55', linestyle='--')
    
    # set xlim
    if xlim is None:
        ax.set_xlim(-1 * width, ind[-1, -1] + 2 * width)
    else:
        ax.set_xlim(xlim[0] * width, xlim[1] * width)
    
    # set ylim
    if ylim is None:
        pass
    else:
        ax.set_ylim(ylim[0], ylim[1])
    
    # xlabel
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    
    # ylabel
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    # legend
    if labels is not None:
        if legendAnchor is None:
            ax.legend(tuple(rects), labels,
                      loc=loc,
                      fancybox=True,
                      shadow=True,
                      ncol=int(round(float(nb) / 2)))
        else:
            ax.legend(tuple(rects), labels,
                      loc=loc,
                      bbox_to_anchor=legendAnchor, # (0.75, 1.115)
                      fancybox=True,
                      shadow=True,
                      ncol=int(round(float(nb) / 2)))
    
    # grid
    ax.grid(which='major', axis='y', linewidth=1.5)
    ax.grid(which='minor', axis='y', linewidth=0.5)
    
    return None


def multiBarPLotAxisText(ax, data, ypos=0, textalpha=4.5):
    # plot text
    
    nb = len(data)
    
    # get x ticks
    x = ax.get_xticks()
    
    # get full x length to resize text
    xlim = ax.get_xlim()
    xlen = xlim[1] - xlim[0]
    S = int(round(textalpha * 12. / xlen))
    
    for i in xrange(nb):
        ax.text(x[i], ypos, data[i], horizontalalignment='center', verticalalignment='bottom',
                fontsize=S)
    
    return None


def nestedMultiBarPLot(data, xticklabels1, xticklabels2, yerr=None, labels=None,
                       xlabel1=None, xlabel2=None, ylabel=None, xlim=None,
                       ylim=None, width=0.15, loc='best', legendAnchor=(0.75, 1.115),
                       figsize=None, rotation=30):
    
    # font parameters
    font = {'family': 'Bitstream Vera Sans',
            'weight': 'normal',
            'size': 16}
    plt.rc('font', **font)
    
    # fill colors and styles
    colors = cycle(['blue', 'yellow', 'cyan', 'green', 'red', 'gray', 'magenta'])
    hatches = cycle(['', '//', '\\', 'x', '++', 'o', 'O', '-', '||', '.', '*'])
    
    # make figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # x indexes
    lx1 = len(xticklabels1)
    lx2 = len(xticklabels2)
    ind = np.arange(lx1 * lx2)
    nb = len(data)
    
    # bars
    rects = []
    if yerr is None:
        for i in xrange(nb):
            out = ax.bar(ind + i * width, data[i], width, color=colors.next(), hatch=hatches.next())
            rects.append(out[0])
    else:
        for i in xrange(nb):
            out = ax.bar(ind + i * width, data[i], width, yerr=yerr, color=colors.next(), hatch=hatches.next())
            rects.append(out[0])
    
    # set xticks
    baseTicks = ind + (nb / 2) * width
    ax.set_xticks(baseTicks)
    ax.set_xticklabels(xticklabels1 * lx2)
    
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
    
    # extra x axes to plot second line of x labels
    ax2 = ax.twiny()
    topTicks = [np.mean(baseTicks[i*lx1:(i+1)*lx1]) for i in xrange(lx2)]
    ax2.set_xticks(topTicks)
    ax2.set_xticklabels(xticklabels2)
    
    # plot dividing lines
    interTopTicks = topTicks[:-1] + np.diff(topTicks) / 2
    for x in interTopTicks:
        ax2.axvline(x, color='0.75', linestyle='--')
    
    # set xlim
    if xlim is None:
        ax.set_xlim(-2*width, ind[-1] + (nb-1) * width + 3*width)
        ax2.set_xlim(-2*width, ind[-1] + (nb-1) * width + 3*width)
    else:
        ax.set_xlim(xlim[0], xlim[1])
        ax2.set_xlim(xlim[0], xlim[1])
    
    # set ylim
    if ylim is None:
        pass
    else:
        ax.set_ylim(ylim[0], ylim[1])
    
    # xlabel
    if xlabel1 is not None:
        ax.set_xlabel(xlabel1)
    
    if xlabel2 is not None:
        ax2.set_xlabel(xlabel2)
    
    # ylabel
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    # legend
    if labels is not None:
        ax.legend(tuple(rects), labels,
                  loc=loc,
                  bbox_to_anchor=legendAnchor, # (0.75, 1.115)
                  fancybox=True,
                  shadow=True,
                  ncol=int(round(float(nb) / 2)))
    
    # grid
    ax.grid(which='major', axis='y', linewidth=1.5)
    ax.grid(which='minor', axis='y', linewidth=0.5)
    
    return fig


def plotTTTemplates(x, trainData, testData, alpha=0.7, linewidth=1, xlabel=None, ylabel=None, title=None, loc='best', figsize=None):
    # plot to compare train and test templates on the same axes
    
    # font parameters
    font = {'family': 'Bitstream Vera Sans',
            'weight': 'normal',
            'size': 16}
    plt.rc('font', **font)
    
    # make figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # plot
    
    ts = ax.plot(x, testData.T, 'g', alpha=alpha, linewidth=linewidth)
    tn = ax.plot(x, trainData.T, 'r', alpha=alpha, linewidth=linewidth)
    
    # grid
    ax.grid()
    
    # xlabel
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    
    # ylabel
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    # title
    if title is not None:
        ax.set_title(title)
    
    # legend
    ax.legend((tn[0], ts[0]), ('Train', 'Test'), loc=loc)
    
    return fig


def clfConfusionMatrix(subjects, rejection_thresholds, clfResults, assessResults):
    # plot identification and authentication confusion matrices
    
    nbs = len(subjects)
    nbt = len(rejection_thresholds)
    
    # identification
    idmat = np.zeros((nbs, nbs)) # (true ID, predicted ID)
    for s in xrange(nbs):
        uniq = np.unique(clfResults[subjects[s]]['identification'])
        counts = np.bincount(clfResults[subjects[s]]['identification'])
        idmat[s, uniq-1] = counts[uniq] / float(counts.sum())
    
    # authentication
    autmat = np.zeros((nbt, nbs, nbs))# (th, true ID, authentication ID) 
    
    
if __name__=='__main__':
    
    #===========================================================================
    # ECG Segments Outlier detection using the Eleazar & Eskin's algorithm
    print "Testing Visualizations.\n"
    #m=pylab.randint(0,4,(10,5))
    #n=len(m.reshape(-1))
    clusterlabels = pylab.randint(0, 4, 50)
    linelabels = pylab.randint(0, 3, 50)
    n = len(clusterlabels)
    # clusterOverTime(clusterlabels, linelabels)
    clusterOverTime(clusterlabels, linelabels)
    
    clusterlabelsD = {"clusters": {}}
    for i in range(len(np.unique(clusterlabels))):
        clusterlabelsD["clusters"][str(i)] = pylab.find(clusterlabels==i)
    clusterOverTime(clusterlabelsD, linelabels)
    
    pylab.show()
    
    