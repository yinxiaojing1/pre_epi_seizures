"""
.. module:: evaluation
   :platform: Unix, Windows
   :synopsis: This module provides various functions to...

.. moduleauthor:: Filipe Canento, Carlos Carreiras


"""

# Imports
# built-in
import copy

# 3rd party
import numpy as np
import scipy
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, PageBreak
from scipy import interpolate, optimize

# BiometricsPyKit
from classifiers import rules
from Cloud import parallel
from misc import misc



def results2report(title, info, image_path, outfile='report.pdf'):
    
    ks = ['database', 
          'filter','segmentation_method', 'outlier_detection', 
          'train_set', 'number_train_runs', 'number_mean_waves_train', 'train_time', 
          'test_set', 'number_test_runs', 'number_mean_waves_test', 'test_time',
          'quantization', 'normalization']
   
    styles = getSampleStyleSheet()
    styleNormal = styles['Normal']
    styleHeading = styles['Heading1']
    styleHeading.alignment = 1 # centre text (TA_CENTRE)
    
    story = []
    # Text is added using the Paragraph class
    story.append(Paragraph('%s'%title, styleHeading))
    story.append(Spacer(inch, .25*inch))
    story.append(PageBreak())
    
    story.append(Paragraph('Parameters', styleNormal))
    story.append(Spacer(inch, .25*inch))
    for k in ks:
        story.append(Paragraph('%s: %s'%(k, info[k]), styleNormal))
        story.append(Spacer(inch, .25*inch))
        
    story.append(PageBreak())
        
    # Images just need a filename and dimensions it should be printed at
    story.append(Paragraph('Authentication:', styleHeading))
    story.append(Image(image_path+'results-authentication.png', 8*inch, 6*inch))
    
    story.append(PageBreak())
    
    story.append(Paragraph('Identification:', styleHeading))
    story.append(Image(image_path+'results-identification.png', 8*inch, 6*inch))
    story.append(PageBreak())
#    # Data can be best presented in a table. A list of lists needs to declared first
#    story.append(Paragraph('Some table', styleHeading))
#    tableData = [ ['Value 1', 'Value 2', 'Sum'],
#    [34, 78, 112],
#    [67,56, 123],
#    [75,23, 98]]
#    story.append(Table(tableData))

    doc = SimpleDocTemplate(outfile, pagesize = A4, title = "Some Report ", author = "Some author")
    doc.build(story)


def intraDistances(data, distanceFcn=None):
    # intra-subject distances
    
    dists = []
    for i in xrange(len(data) - 1):
        dists.extend(misc.wavedistance(data[i], data[i+1:], distanceFcn))
    
    dists = np.array(dists)
    
    return {'min': dists.min(), 'sum': dists.sum(), 'max': dists.max(), 'N': len(dists)}


def interDistances(data, data2=None, distanceFcn=None):
    # inter-subject distances
    
    dists = []
    for i in xrange(len(data)):
        dists.extend(misc.wavedistance(data[i], data2, distanceFcn))
    
    dists = np.array(dists)
    
    return {'min': dists.min(), 'sum': dists.sum(), 'max': dists.max(), 'N': len(dists)}


def partialDistStats(pdata):
    # compute stats from partial data
    
    mi, ma, su, N = [], [], [], []
    for item in pdata:
        mi.append(item['min'])
        ma.append(item['max'])
        su.append(item['sum'])
        N.append(item['N'])
    
    mi = np.min(mi)
    ma = np.max(ma)
    me = np.sum(su) / float(np.sum(N))
    
    return {'min': mi, 'max': ma, 'mean': me}


def distanceStats(data, distanceFcn):
    # get basic statistics (min, max, mean) of the distance between the items in data
    # item in data <=> subject
    
    nb = len(data)
    workQ = parallel.getQueue()
    
    interTasks = []
    intraTasks = []
    
    taskid = 0
    for i in xrange(nb):
        # intra-subject distances
        workQ.put({'function': intraDistances,
                   'data': data[i],
                   'parameters': {'distanceFcn': distanceFcn},
                   'taskid': taskid,
                   })
        intraTasks.append(taskid)
        taskid += 1
        
        # inter-subject distances
        for j in xrange(i+1, nb):
            workQ.put({'function': interDistances,
                       'data': data[i],
                       'parameters': {'data2': data[j],
                                      'distanceFcn': distanceFcn,
                                      },
                       'taskid': taskid,
                       })
            interTasks.append(taskid)
            taskid += 1
    
    # run in multiprocessing (data mode)
    store = parallel.getDictManager()
    parallel.runMultiprocess(workQ, store, log2file=False)
    
    # gather results
    interD = [store[t] for t in interTasks]
    intraD = [store[t] for t in intraTasks]
    
    out = {'inter': partialDistStats(interD),
           'intra': partialDistStats(intraD),
           }
    
    return out


def computeAuthRates(TP, FP, TN, FN, thresholds):
    # compute authentication rates
    
    Acc = (TP + TN) / (TP + TN + FP + FN)
    
    # avoid division by zero
    SA = TP + FN # should accepts
    SA[SA <= 0] = 1.
    SR = TN + FP # should rejects
    SR[SR <= 0] = 1.
    
    TAR = TP / SA
    FAR = FP / SR
    FRR = FN / SA
    TRR = TN / SR
    EER = findEqual(thresholds, FAR, thresholds, FRR)
    
    output = {'Acc': Acc,
              'TAR': TAR,
              'FAR': FAR,
              'FRR': FRR,
              'TRR': TRR,
              'EER': EER,
              }
    
    return output


def computeIdRates(H, M, R, N, thresholds):
    # compute identifcation results
    
    Acc = H / N
    Err = 1 - Acc
    MR = M / N
    RR = R / N
    EER = findEqual(thresholds, MR, thresholds, RR)
    EID = findEqual(thresholds, Err, thresholds, np.min(Err) * np.ones(len(thresholds), dtype='float'))
    
    output = {'Acc': Acc,
              'Err': Err,
              'MR': MR,
              'RR': RR,
              'EER': EER,
              'EID': EID,
              }
    
    return output


def subjectResults(results, subject, rejection_thresholds, subjects, subjectDict, subjectIdx):
    # subject is the true label
    
    nth = len(rejection_thresholds)
    auth_res = results['authentication']
    id_res = results['identification']
    ns = auth_res.shape[2]
    
    # some sanity checks
    if auth_res.shape[0] != id_res.shape[0]:
        raise ValueError, "Authentication and identification number of thresholds do not match."
    if auth_res.shape[0] != nth:
        raise ValueError, "Number of thresholds in vector does not match biometric results."
    if auth_res.shape[2] != id_res.shape[1]:
        raise ValueError, "Authentication and identification number of tests do not match."
    
    label = subjectDict[subject]
    
    # authentication vars
    TP = np.zeros(nth, dtype='float')
    FP = np.zeros(nth, dtype='float')
    TN = np.zeros(nth, dtype='float')
    FN = np.zeros(nth, dtype='float')
    
    # identification vars
    H = np.zeros(nth, dtype='float')
    M = np.zeros(nth, dtype='float')
    R = np.zeros(nth, dtype='float')
    CM = []
    
    for i in xrange(nth): # for each threshold
        # authentication
        for k, lbl in enumerate(subjectIdx): # for each subject
            subject_tst = subjects[k]
            
            d = auth_res[i, lbl, :]
            if subject == subject_tst:
                # true positives
                aux = np.sum(d)
                TP[i] += aux
                # false negatives
                FN[i] += (ns - aux)
            else:
                # false positives
                aux = np.sum(d)
                FP[i] += aux
                # true negatives
                TN[i] += (ns - aux)
        
        # identification
        res = id_res[i, :]
        hits = res == label
        nhits = np.sum(hits)
        rejects = res == -1
        nrejects = np.sum(rejects)
        misses = np.logical_not(np.logical_or(hits, rejects))
        nmisses = ns - (nhits + nrejects)
        missCounts = {subjectDict[:ms]: np.sum(res == ms) for ms in np.unique(res[misses])}
        
        # appends
        H[i] = nhits
        M[i] = nmisses
        R[i] = nrejects
        CM.append(missCounts)
    
    output = {
              'authentication': {
                                 'confusionMatrix': {
                                                     'TP': TP,
                                                     'FP': FP,
                                                     'TN': TN,
                                                     'FN': FN,
                                                     },
                                 'rates': computeAuthRates(TP, FP, TN, FN, rejection_thresholds),
                                 },
              'identification': {
                                 'confusionMatrix': {
                                                     'H': H,
                                                     'M': M,
                                                     'R': R,
                                                     'CM': CM,
                                                     },
                                 'rates': computeIdRates(H, M, R, ns, rejection_thresholds),
                                 },
              }
    
    return output


def assessClassification(results, rejection_thresholds, ignSubjects=[], dstPath=None, log2file=False):
    # assess the results of a classifier evaluation run
    
    # multiprocessing
    workQ = parallel.getQueue()
    if dstPath is None:
        store = parallel.getDictManager()
    else:
        store = dstPath
    
    # test subjects
    subjectDict = results['subjectDict']
    subParent = results['subjectList']
    subjects = list(set(subParent) - set(ignSubjects))
    subIdx = [subParent.index(item) for item in subjects]
    subIdx.sort()
    subjects = [subParent[item] for item in subIdx]
    
    # add to queue
    for test_user in subjects:
        workQ.put({
                   'function': subjectResults,
                   'data': results[test_user],
                   'parameters': {
                                  'subject': test_user,
                                  'rejection_thresholds': rejection_thresholds,
                                  'subjects': subjects,
                                  'subjectDict': subjectDict,
                                  'subjectIdx': subIdx,
                                  },
                   'taskid': test_user,
                   })
    
    # run in multiprocessing (data mode)
    parallel.runMultiprocess(workQ, store, log2file=log2file)
    
    # global results
    output = {
              'global': {
                         'authentication': {
                                            'confusionMatrix': {
                                                                'TP': 0.,
                                                                'TN': 0.,
                                                                'FP': 0.,
                                                                'FN': 0.,
                                                                },
                                            },
                         'identification': {
                                            'confusionMatrix': {
                                                                'H': 0.,
                                                                'M': 0.,
                                                                'R': 0.,
                                                                },
                                            },
                         },
              'subject': {},
              'rejection_thresholds': rejection_thresholds,
              }
    
    nth = len(rejection_thresholds)
    C = np.zeros((nth, len(subjects)), dtype='float')
    
    # update variables
    auth = output['global']['authentication']['confusionMatrix']
    authM = ['TP', 'TN', 'FP', 'FN']
    iden = output['global']['identification']['confusionMatrix']
    idenM = ['H', 'M', 'R']
    for test_user in subjects:
        aux = parallel.loadStore(store, test_user)
        # copy to subject
        output['subject'][test_user] = copy.deepcopy(aux)
        # authentication
        for m in authM:
            auth[m] += aux['authentication']['confusionMatrix'][m]
        # identification
        for m in idenM:
            iden[m] += aux['identification']['confusionMatrix'][m]
        # subject misses
        for i, item in enumerate(aux['identification']['confusionMatrix']['CM']):
            for k, sub in enumerate(subjects):
                try:
                    C[i, k] += item[sub]
                except KeyError:
                    pass
    
    # normalize subject misses
    sC = C.sum(axis=1).reshape((nth, 1))
    sC[sC <= 0] = 1. # avoid division by zero
    CR = C / sC
    # update subjects
    for k, sub in enumerate(subjects):
        output['subject'][sub]['identification']['confusionMatrix']['C'] = C[:, k]
        output['subject'][sub]['identification']['rates']['CR'] = CR[:, k]
    
    # compute rates
    output['global']['authentication']['rates'] = computeAuthRates(auth['TP'], auth['FP'], auth['TN'], auth['FN'], rejection_thresholds)
    
    # identification
    Ns = iden['H'] + iden['M'] + iden['R']
    output['global']['identification']['rates'] = computeIdRates(iden['H'], iden['M'], iden['R'], Ns, rejection_thresholds)
    
    return output


def assessRuns(runResults, subjects):
    # assess the results of multiple runs
    # runResults is a list of dictionaries produced by assessClassification for each run
    
    # check inputs
    nb = len(runResults)
    if nb == 0:
        return None
    elif nb == 1:
        return runResults[0]
    
    # output dict
    output = {'global': {'authentication': {'confusionMatrix': {'TP': 0.,
                                                                'TN': 0.,
                                                                'FP': 0.,
                                                                'FN': 0.,
                                                                },
                                            },
                         'identification': {'confusionMatrix': {'H': 0.,
                                                                'M': 0.,
                                                                'R': 0.,
                                                                },
                                            },
                         },
              'subject': {},
              'rejection_thresholds': None,
              }
    rejection_thresholds = output['rejection_thresholds'] = runResults[0]['rejection_thresholds']
    
    # global helpers
    auth = output['global']['authentication']['confusionMatrix']
    iden = output['global']['identification']['confusionMatrix']
    authM = ['TP', 'TN', 'FP', 'FN']
    idenM1 = ['H', 'M', 'R', 'C']
    idenM2 = ['H', 'M', 'R']
    
    for sub in subjects:
        # create subject confusion matrix, rates
        output['subject'][sub] = {'authentication': {'confusionMatrix': {'TP': 0.,
                                                                         'TN': 0.,
                                                                         'FP': 0.,
                                                                         'FN': 0.,
                                                                         },
                                                     'rates': {},
                                                     },
                                  'identification': {'confusionMatrix': {'H': 0.,
                                                                         'M': 0.,
                                                                         'R': 0.,
                                                                         'C': 0.,
                                                                         },
                                                     'rates': {},
                                                     },
                                  }
        
        # subject helpers
        authS = output['subject'][sub]['authentication']['confusionMatrix']
        idenS = output['subject'][sub]['identification']['confusionMatrix']
        
        # update confusions
        for run in runResults:
            # authentication
            auth_run = run['subject'][sub]['authentication']['confusionMatrix']
            for m in authM:
                auth[m] += auth_run[m]
                authS[m] += auth_run[m]
            
            # identification
            iden_run = run['subject'][sub]['identification']['confusionMatrix']
            for m in idenM1:
                idenS[m] += iden_run[m]
            for m in idenM2:
                iden[m] += iden_run[m]
        
        # compute subject mean
        # authentication
        for m in authM:
            authS[m] /= float(nb)
        
        # identification
        for m in idenM1:
            idenS[m] /= float(nb)
        
        # compute subject rates
        output['subject'][sub]['authentication']['rates'] = computeAuthRates(authS['TP'], authS['FP'], authS['TN'], authS['FN'], rejection_thresholds)
        Ns = idenS['H'] + idenS['M'] + idenS['R']
        output['subject'][sub]['identification']['rates'] = computeIdRates(idenS['H'], idenS['M'], idenS['R'], Ns, rejection_thresholds)
        M = np.array(idenS['M'], copy=True)
        M[M <= 0] = 1.
        output['subject'][sub]['identification']['rates']['CR'] = idenS['C'] / M
        
    
    # compute global mean
    # authentication
    for m in authM:
        auth[m] /= float(nb)
    
    # identification
    for m in idenM2:
        iden[m] /= float(nb)
    
    # compute rates
    output['global']['authentication']['rates'] = computeAuthRates(auth['TP'], auth['FP'], auth['TN'], auth['FN'], rejection_thresholds)
    Ns = iden['H'] + iden['M'] + iden['R']
    output['global']['identification']['rates'] = computeIdRates(iden['H'], iden['M'], iden['R'], Ns, rejection_thresholds)
    
    return output


def pdiff(x, p1, p2):
    # find difference between interpolators
    return (p1(x) - p2(x))**2


def findEqual(x1, y1, x2, y2, alpha=1.5, xtol=1e-6, ytol=1e-6):
    # find (one) intersection between two lines, using Piecewise Polynomial interpolation
    
    # guarantee everything is numpy
    x1 = np.array(x1, copy=False)
    y1 = np.array(y1, copy=False)
    x2 = np.array(x2, copy=False)
    y2 = np.array(y2, copy=False)
    
    # interpolate
    if scipy.__version__ >= '0.14.0':
        p1 = interpolate.BPoly.from_derivatives(x1, y1[:, np.newaxis])
        p2 = interpolate.BPoly.from_derivatives(x2, y2[:, np.newaxis])
    else:
        p1 = interpolate.PiecewisePolynomial(x1, y1[:, np.newaxis])
        p2 = interpolate.PiecewisePolynomial(x2, y2[:, np.newaxis])
    
    # combine x intervals
    x = np.r_[x1, x2]
    x_min = x.min()
    x_max = x.max()
    npoints = int(len(np.unique(x)) * alpha)
    x = np.linspace(x_min, x_max, npoints)
    
    # initial estimates
    pd = np.abs(p1(x) - p2(x))
    xi = x[pd < ytol]
    
    # search for solutions
    roots = set()
    for v in xi:
        root, _, ier, _ = optimize.fsolve(pdiff, v, (p1, p2), full_output=True, xtol=xtol)
        if ier == 1 and x_min <= root <= x_max:
            roots.add(root[0])
    
    if len(roots) == 0:
        # no solution was found => give the best from the initial estimates
        roots = [x[pd.argmin()]]
    
    # compute values
    roots = list(roots)
    roots.sort()
    values = np.mean(np.vstack((p1(roots), p2(roots))), axis=0)
    
    return np.vstack((roots, values)).T


def combineSubjectClass(data, label2subject, subject2label):
    # combine identification and authentication of a subject
    
    out = {'identification': [],
           'authentication': {},
           }
    
    # identification
    for i in xrange(len(data[0]['identification'])):
        aux = [subject2label[c['identification'][i]] for c in data]
        # plurality rule (most frequent label)
        out['identification'].append(label2subject[rules.pluralityRule(aux)[0]])
    
    out['identification'] = np.array(out['identification'])
    
    # authentication
    out['authentication'] = data[0]['authentication']
    
    return out


def combineClassifiers(clf, *p):
    # combine the evaluation results of various classifiers
    # clf is the evaluation output from the classifier
    
    # classifier list
    clfs = [clf]
    for c in p:
        clfs.append(c)
    
    if len(clfs) == 1:
        # only one classifier
        return clf
    else:
        subjects = clf.keys()
        nbSubjects = len(subjects)
        
        # build helper dicts
        label2subject = {}
        subject2label = {}
        for i in xrange(nbSubjects):
            sub = subjects[i]
            label2subject[i] = sub
            subject2label[sub] = i
        
        # fill work queue
        workQ = parallel.getQueue()
        for i in xrange(nbSubjects):
            sub = subjects[i]
            workQ.put({'function': combineSubjectClass,
                       'data': [c[sub] for c in clfs],
                       'parameters': {'label2subject': label2subject,
                                      'subject2label': subject2label,
                                      },
                       'taskid': i,
                       })
        
        # run in multiprocessing
        store = parallel.getDictManager()
        parallel.runMultiprocess(workQ, store)
        
        # load results
        results = {}
        for i in xrange(nbSubjects):
            sub = subjects[i]
            results[sub] = parallel.loadStore(store, i)
        
        return results

