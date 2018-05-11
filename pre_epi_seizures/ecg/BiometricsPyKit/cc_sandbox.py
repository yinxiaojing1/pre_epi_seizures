'''
Created on Apr 8, 2013

@author: Carlos
'''

# Imports
import copy
import os

import numpy as np
import matplotlib
# change to a back-end that does not use DISPLAY (for Linux), not interactive
matplotlib.use('Agg')
import xmltodict
from lxml import etree

from Cloud import client
from datamanager import datamanager
from evaluation import evaluation
from visualization import visualization



mapper = {'database': {'items': [('Name', 'DBConnection/dbName'), ],
                       'name': 'Database',
                       },
          
          'starting_data': {'items': [],
                            'name': 'Starting Data',
                            },
          
          'train': {'items': [('Set', 'tags'),
                             ('Outliers', 'outlier/method'),
                             ('Mean Waves', 'prepareParameters/number_mean_waves'),
                             ('Median Waves', 'prepareParameters/number_median_waves'),
                             ],
                    'name': 'Train',
                    },
          
          'test': {'items': [('Set', 'tags'),
                            ('Outliers', 'outlier/method'),
                            ('Mean Waves', 'prepareParameters/number_mean_waves'),
                            ('Median Waves', 'prepareParameters/number_median_waves'),
                            ],
                   'name': 'Test',
                   },
          
          'dimreduction': {'items': [],
                           'name': 'Dimensionality Reduction',
                           },
          
          'classifier': {'items': [('C', 'parameters/C'),
                                   ('k', 'parameters/k'),
                                   ],
                         'instances': {'SVM': ['clf-svm'],
                                      'k-NN': ['clf-knn'],
                                       },
                         'name': 'Classifier',
                         },
          }


def readCardioXML(path, name):
    # read a Philips ECG XML file
    
    # open file and read XML
    with open(os.path.join(path, name), 'r') as fid:
        # load to string
        doc_str = fid.read()
        # load to xml parser
        fid.seek(0)
        doc_parser = etree.parse(fid)
    
    # load schema file
    with open(os.path.join(path, 'SierraECG.xsd'), 'r') as fid:
        schema_doc = etree.parse(fid)
    schema = etree.XMLSchema(schema_doc)
        
    # validate schema
    assert schema.validate(doc_parser), "XML file failed schema validation."
    
    # convert to dict
    doc = xmltodict.parse(doc_str)
    
    # return what we need
    output = {}
    
    # sample frequency
    output['sampleRate'] = float(doc['restingecgdata']['dataacquisition']['signalcharacteristics']['samplingrate'])
    
    # resolution
    output['resolution'] = int(doc['restingecgdata']['dataacquisition']['signalcharacteristics']['bitspersample'])
    
    # number of channels
    #nb = int(doc['restingecgdata']['dataacquisition']['signalcharacteristics']['numberchannelsallocated'])
    nb = int(doc['restingecgdata']['dataacquisition']['signalcharacteristics']['numberchannelsvalid'])
    
    # channel labels
    labels = doc['restingecgdata']['reportinfo']['reportformat']['waveformformat']['mainwaveformformat']['#text']
    output['labels'] = labels.split(' ')
    
    # patient ID
    output['subject'] = doc['restingecgdata']['patient']['generalpatientdata']['patientid']
    
    # waveform data
    compress = doc['restingecgdata']['waveforms']['parsedwaveforms']['@compressflag']
    assert compress == 'False', "Compressed waveforms; please decompress first."
    
    data = np.fromstring(doc['restingecgdata']['waveforms']['parsedwaveforms']['#text'], sep=' ')
    data = data.reshape((nb, len(data) / nb))
    output['signal'] = data
    
    return output


def assembleTables(connection, path, taskList):
    # assemble latex tables
    
    # Set A (templates from T1 to train, mean waves from T1 to test)
    tabA = """\\begin{table}
\\centering
\\caption{Authentication and identification performance for within-session analysis.}
\\label{tab:setA}
\\begin{tabular}{cccccc}
\\multirow{2}{*}{\\textbf{Outlier}} & \\multicolumn{2}{c}{\\textbf{Template Selection}} & \\multirow{2}{*}{\\textbf{Classifier}} & \\multirow{2}{*}{\\textbf{EER ($\\%$)}} & \\multirow{2}{*}{\\textbf{EID ($\\%$)}} \\\\
~ & \\textit{Clustering} & \\textit{Method} & ~ & ~ & ~ \\\\
\\hline
"""
    
    # Set B (templates from T1 to train, mean waves from T2 to test)
    tabB = """\\begin{table}
\\centering
\\caption{Authentication and identification performance for across-session (without fusion) analysis.}
\\label{tab:setB}
\\begin{tabular}{cccccc}
\\multirow{2}{*}{\\textbf{Outlier}} & \\multicolumn{2}{c}{\\textbf{Template Selection}} & \\multirow{2}{*}{\\textbf{Classifier}} & \\multirow{2}{*}{\\textbf{EER ($\\%$)}} & \\multirow{2}{*}{\\textbf{EID ($\\%$)}} \\\\
~ & \\textit{Clustering} & \\textit{Method} & ~ & ~ & ~ \\\\
\\hline
"""
    
    # Set C (templates from T1+T2 to train, mean waves from T1 to test)
    tabC = """\\begin{table}
\\centering
\\caption{Authentication and identification performance for across-session (with fusion, first session) analysis.}
\\label{tab:setC}
\\begin{tabular}{cccccc}
\\multirow{2}{*}{\\textbf{Outlier}} & \\multicolumn{2}{c}{\\textbf{Template Selection}} & \\multirow{2}{*}{\\textbf{Classifier}} & \\multirow{2}{*}{\\textbf{EER ($\\%$)}} & \\multirow{2}{*}{\\textbf{EID ($\\%$)}} \\\\
~ & \\textit{Clustering} & \\textit{Method} & ~ & ~ & ~ \\\\
\\hline
"""
    
    # Set D (templates from T1+T2 to train, mean waves from T2 to test)
    tabD = """\\begin{table}
\\centering
\\caption{Authentication and identification performance for across-session (with fusion, second session) analysis.}
\\label{tab:setD}
\\begin{tabular}{cccccc}
\\multirow{2}{*}{\\textbf{Outlier}} & \\multicolumn{2}{c}{\\textbf{Template Selection}} & \\multirow{2}{*}{\\textbf{Classifier}} & \\multirow{2}{*}{\\textbf{EER ($\\%$)}} & \\multirow{2}{*}{\\textbf{EID ($\\%$)}} \\\\
~ & \\textit{Clustering} & \\textit{Method} & ~ & ~ & ~ \\\\
\\hline
"""
    
    # helper 1
    def processTask(path, taskID, task):
        # get authentication results
        auth_svm = datamanager.gzLoad(os.path.join(path, 'Exp-%d' % taskID, 'results', 'clf-svm', 'authentication-0-0.dict'))
        th = np.arange(len(auth_svm['FAR']))
        res = evaluation.findEqual(th, auth_svm['FAR'], th, auth_svm['FRR'])
        EER_svm = 100. * np.min(res[:, 1])
        
        auth_knn = datamanager.gzLoad(os.path.join(path, 'Exp-%d' % taskID, 'results', 'clf-knn', 'authentication.dict'))
        sel = slice(None, None, 2)
        th = np.arange(len(auth_knn['FAR']))[sel]
        res = evaluation.findEqual(th, auth_knn['FAR'][sel], th, auth_knn['FRR'][sel])
        EER_knn = 100. * np.min(res[:, 1])
        
        # get identification results
        id_svm = datamanager.gzLoad(os.path.join(path, 'Exp-%d' % taskID, 'results', 'clf-svm', 'identification-0-0.dict'))
        id_svm = 100. * id_svm['Err']
        
        id_knn = datamanager.gzLoad(os.path.join(path, 'Exp-%d' % taskID, 'results', 'clf-knn', 'identification.dict'))
        id_knn = 100. * id_knn['IDError']
        
        # build line info
        items_svm = {}
        items_svm['outlier'] = task['train']['outlier']['method'].upper()
        try:
            items_svm['clustering'] = task['train']['cluster']['parameters']['method'].capitalize()
        except KeyError:
            items_svm['clustering'] = 'None'
        items_svm['templates-met'] = task['train']['templateSelection']['method'].capitalize()
        items_svm['templates-n'] = task['train']['templateSelection']['parameters']['ntemplatesPerCluster']
        items_svm['classifier'] = 'SVM'
        items_svm['EER'] = EER_svm
        items_svm['EID'] = id_svm
        
        items_knn = {}
        items_knn['outlier'] = task['train']['outlier']['method'].upper()
        try:
            items_knn['clustering'] = task['train']['cluster']['parameters']['method'].capitalize()
        except KeyError:
            items_knn['clustering'] = 'None'
        items_knn['templates-met'] = task['train']['templateSelection']['method'].capitalize()
        items_knn['templates-n'] = task['train']['templateSelection']['parameters']['ntemplatesPerCluster']
        items_knn['classifier'] = '$k$-NN'
        items_knn['EER'] = EER_knn
        items_knn['EID'] = id_knn
        
        return [items_svm, items_knn]
    
    # helper 2
    def add2Tab(tab, lines):
        fmt = '%s & %s & %s, %d templates & %s & %2.1f & %2.1f \\\\\n'
        for line in lines:
            items = (line['outlier'], line['clustering'], line['templates-met'], line['templates-n'],
                     line['classifier'], line['EER'], line['EID'])
            tab += fmt % items
        
        return tab
    
    # parse the experiments
    linesA = []
    linesB = []
    linesC = []
    linesD = []
    for taskID in taskList:
        print taskID
        # get configuration dict
        task = connection.getTaskInfo(taskID)
        
        # only process completed tasks
        if task['status']  == 'finished':
#            # plot templates
#            fpath = os.path.join(path, 'Exp-%d' % taskID, 'results', 'plots')
#            if not os.path.exists(fpath):
#                os.mkdir(fpath)
#            for i in xrange(1, 64):
#                fig = plotTemplates(path, taskID, i)
#                fig.savefig(os.path.join(fpath, 'templates-%d.png' % i), bbox_inches='tight', dpi=200)
#                visualization.close(fig)
            
            # determine the set
            if task['train']['tags'] == ['Sitting', 'T1']:
                if task['test']['tags'] == ['Sitting', 'T1']:
                    # Set A
                    linesA.extend(processTask(path, taskID, task))
                elif task['test']['tags'] == ['Sitting', 'T2']:
                    # Set B
                    linesB.extend(processTask(path, taskID, task))
            elif task['train']['tags'] == ['Sitting']:
                if task['test']['tags'] == ['Sitting', 'T1']:
                    # Set C
                    linesC.extend(processTask(path, taskID, task))
                elif task['test']['tags'] == ['Sitting', 'T2']:
                    # Set D
                    linesD.extend(processTask(path, taskID, task))
    
    # sort lists
    linesA = sorted(linesA, key=lambda x: (x['outlier'], x['templates-met'], x['templates-n'], x['clustering'], x['classifier']))
    linesB = sorted(linesB, key=lambda x: (x['outlier'], x['templates-met'], x['templates-n'], x['clustering'], x['classifier']))
    linesC = sorted(linesC, key=lambda x: (x['outlier'], x['templates-met'], x['templates-n'], x['clustering'], x['classifier']))
    linesD = sorted(linesD, key=lambda x: (x['outlier'], x['templates-met'], x['templates-n'], x['clustering'], x['classifier']))
    
    # save lists
    datamanager.gzStore(os.path.join(path, 'linesA.dict'), linesA)
    datamanager.gzStore(os.path.join(path, 'linesB.dict'), linesB)
    datamanager.gzStore(os.path.join(path, 'linesC.dict'), linesC)
    datamanager.gzStore(os.path.join(path, 'linesD.dict'), linesD)
    
    # add lines to tab
    tabA = add2Tab(tabA, linesA)
    tabB = add2Tab(tabB, linesB)
    tabC = add2Tab(tabC, linesC)
    tabD = add2Tab(tabD, linesD)
    
    # finalize tables
    fin = """\\hline
\\end{tabular}
\\end{table}
"""
    tabA += fin
    tabB += fin
    tabC += fin
    tabD += fin
    
    tabs = tabA + '\n\n' + tabB + '\n\n' + tabC + '\n\n' + tabD
    
    # save to file
    with open(os.path.join(path, 'tables.txt'), 'w') as fid:
        fid.write(tabs)


def makeMeans(path, label='EER'):
    # make means
    
    tests = ['A', 'B', 'C', 'D']
    tests.sort()
    omethods = ['DBSCAN', 'DMEAN']
    omethods.sort()
    cmethods = ['SVM', '$k$-NN']
    cmethods.sort()
    
    results = {}
    for c in tests:
        # load the list
        lines = datamanager.gzLoad(os.path.join(path, 'lines%s.dict' % c))
        
        results[c] = {}
        for om in omethods:
            results[c][om] = {}
            
            # filter on outlier method
            outList = filter(lambda x: x['outlier'] == om, lines)
            
            for cm in cmethods:
                results[c][om][cm] = {}
                
                # filter on classifier method
                classList = filter(lambda x: x['classifier'] == cm, outList)
                
                # get values
                values = np.array([item[label] for item in classList])
                results[c][om][cm]['values'] = values
                
                # compute mean and std
                if len(values) == 0:
                    results[c][om][cm]['mean'] = 0.
                    results[c][om][cm]['std'] = 0.
                else:
                    results[c][om][cm]['mean'] = values.mean()
                    results[c][om][cm]['std'] = values.std(ddof=1)
    
    # begin latex table
    if label == 'EER':
        tab = """\\begin{table}
\\centering
\\caption{Comparison of the EER ($\\mu \\pm \\sigma$) for the outlier detection methods and classifiers.}
\\label{tab:outliers}
\\begin{tabular}{c r@{$\\pm$}l r@{$\\pm$}l r@{$\\pm$}l r@{$\\pm$}l}
\\multirow{2}{*}{\\textbf{Test}} & \\multicolumn{4}{c}{\\textbf{DBSCAN}} & \\multicolumn{4}{c}{\\textbf{DMEAN}} \\\\
~ & \\multicolumn{2}{c}{\\textit{$k$-NN}} & \\multicolumn{2}{c}{\\textit{SVM}} & \\multicolumn{2}{c}{\\textit{$k$-NN}} & \\multicolumn{2}{c}{\\textit{SVM}} \\\\
\\hline
"""
    else:
        tab = """\\begin{table}
\\centering
\\caption{Comparison of the EID ($\\mu \\pm \\sigma$) for the outlier detection methods and classifiers.}
\\label{tab:outliers}
\\begin{tabular}{c r@{$\\pm$}l r@{$\\pm$}l r@{$\\pm$}l r@{$\\pm$}l}
\\multirow{2}{*}{\\textbf{Test}} & \\multicolumn{4}{c}{\\textbf{DBSCAN}} & \\multicolumn{4}{c}{\\textbf{DMEAN}} \\\\
~ & \\multicolumn{2}{c}{\\textit{$k$-NN}} & \\multicolumn{2}{c}{\\textit{SVM}} & \\multicolumn{2}{c}{\\textit{$k$-NN}} & \\multicolumn{2}{c}{\\textit{SVM}} \\\\
\\hline
"""
    
    # add lines
    fmt = '%s & %2.1f & %2.1f & %2.1f & %2.1f & %2.1f & %2.1f & %2.1f & %2.1f \\\\\n'
    
    for c in tests:
        items = [c]
        for om in omethods:
            for cm in cmethods:
                items.append(results[c][om][cm]['mean'])
                items.append(results[c][om][cm]['std'])
        tab += fmt % tuple(items)
    
    # finalize table
    fin = """\hline
\\end{tabular}
\\end{table}
"""
    tab += fin
    
    # save table
    with open(os.path.join(path, 'outliersMeanTable.txt'), 'w') as fid:
        fid.write(tab)
    
    return results


def makeBarPlots(path, dlabel, xlabel1, xlabel2, ylabel, case, selection, pretty, ylim):
    # make plots
    
    # helper function
    def prettify(item, pretty):
        try:
            res = pretty[item]
        except KeyError:
            res = item
        return res
    
    # load the list
    lines = datamanager.gzLoad(os.path.join(path, 'lines%s.dict' % case))
    
    # filter bases on selection dict
    fLines = filter(lambda item: np.all([item[key] == selection[key] for key in selection.iterkeys()]), lines)
    
    # format the data for plot
    xticklabels1 = list(set([prettify(item[xlabel1], pretty) for item in fLines]))
    xticklabels1.sort()
    xticklabels2 = list(set([prettify(item[xlabel2], pretty) for item in fLines]))
    xticklabels2.sort()
    labels = list(set([prettify(item[ylabel], pretty) for item in fLines]))
    labels.sort()
    data = []
    for lbl in labels:
        aux = []
        for xlbl2 in xticklabels2:
            for xlbl1 in xticklabels1:
                sel = filter(lambda item: np.all([item[ylabel] == lbl, item[xlabel1] == xlbl1, item[xlabel2] == xlbl2]), fLines)
                length = len(sel)
                if length > 1:
                    raise ValueError, "There are more variables than those accounted for in the selection dicionary."
                elif length == 0:
                    aux.append(0.)
                else:
                    aux.append(sel[0][dlabel])
        data.append(aux)
    
    # plot
    fig = visualization.nestedMultiBarPLot(data, xticklabels1, xticklabels2, labels,
                                     prettify(xlabel1, pretty), prettify(xlabel2, pretty), '%s (%%)' % dlabel,
                                     xlim=None, ylim=ylim,
                                     width=0.15, figsize=(12, 9),
                                     loc='best', legendAnchor=None)
    
    return fig


def plotTemplates(connection, path, taskList):
    # plot train and test templates
    
    for taskID in taskList:
        print taskID
        # get configuration dict
        task = connection.getTaskInfo(taskID)
        
        # only process completed tasks
        if task['status']  == 'finished':
            fpath = os.path.join(path, 'Exp-%d' % taskID, 'results', 'plots')
            if not os.path.exists(fpath):
                os.mkdir(fpath)
            nbSubjects = 0
            # for i in xrange(1, 64):
            for i in [7, 20]:
                # load train templates
                train = datamanager.gzLoad(os.path.join(path, 'Exp-%d' % taskID, 'templates', 'final', 'train', 'output-%d' % i))
                if len(train) < 3:
                    continue
                # load test templates
                test = datamanager.gzLoad(os.path.join(path, 'Exp-%d' % taskID, 'templates', 'final', 'test', 'output-%d' % i))
                if len(test) < 3:
                    continue
                nbSubjects += 1
                # plot
                time = np.arange(-200, 400, 1.)
                fig = visualization.plotTTTemplates(time, train, test,
                                                    xlabel='Time (ms)', ylabel='Amplitude',
                                                    title=None,
                                                    alpha=0.5, linewidth=2)
                
                # save figure
                fig.savefig(os.path.join(fpath, 'templates-%d.pdf' % i), bbox_inches='tight', dpi=500)
                visualization.close(fig)
            print taskID, nbSubjects



# starting configuration
#expConfig = {'database': {
#                          'source': 'StorageBIT',
#                          'DBConnection': {
#                                           'dbName': 'CVP',
#                                           'host': '193.136.222.234',
#                                           'dstPath': '',
#                                           'sync': False
#                                           },
#                          'experiments': [
#                                          'T1-Sitting',
#                                          ],
#                          'mapper': {
#                                     'raw': 'signals/ECG/hand/raw',
#                                     'filtered': 'signals/ECG/hand/zee5to20',
#                                     'segments': 'signals/ECG/hand/zee5to20/Segments/engzee',
#                                     'R': 'events/ECG/hand/zee5to20/R/engzee',
#                                     }
#                          },
#             
#             'starting_data': {
#                               'data_type': 'segments',
#                               'refine': None,
#                               'tasks': True,
#                               'mdataSplit': True,
#                               'concatenate': False,
#                               'axis': 0,
#                              },
#             
#             'train': {
#                       'tags': ['Sitting', 'T1'],
#                       
#                       'processing_sequence': ['outlier', 'featureSelection', 'cluster', 'templateSelection'],
#                       
#                       'min_nb_templates': 3,
#                       
#                       'number_runs': 1,
#                       
#                       'filter': {
#                                  'src': '',
#                                  'method': 'zee5to20',
#                                  'parameters': {},
#                                  },
#                       
#                       'segment': {
#                                   'src': '',
#                                   'method': 'engzee',
#                                   'parameters': {},
#                                   },
#                       
#                       'outlier': {
#                                   'src': 'segments',
##                                   'method': 'dbscan',
##                                   'parameters': {
##                                                  'metric': 'euclidean',
##                                                  'min_samples': 10,
##                                                  'eps': 0.9,
##                                                  },
#                                   'method': 'dmean',
#                                   'parameters': {
#                                                  'metric': 'cosine',
#                                                  'alpha': 0.5,
#                                                  'R_Position': 200,
#                                                  },
#                                   },
#                       
#                       'featureSelection': {
#                                            'src': 'outliers/templates',
#                                            'parameters': {
#                                                           'normalization': None,
#                                                           'subpattern': False,
#                                                           'number_mean_waves': 5,
#                                                           'number_median_waves': 0,
#                                                           'quantization': -1,
#                                                           'patterns2strings': False,
#                                                           },
#                                            },
#                       
#                       'cluster': {
#                                   'src': 'features',
#                                   'method': 'hierarchical',
#                                   'parameters': {
#                                                  'method': 'average',
#                                                  'k': 0,
#                                                  'metric': 'euclidean',
#                                                  'showDendrogram': False,
#                                                  },
#                                   },
#                       
#                       'templateSelection': {
#                                             'src': 'features',
#                                             'method': 'mdist',
#                                             'parameters': {
#                                                            'ntemplatesPerCluster': 15,
#                                                            'distMeasure': 'Euclidean',
#                                                            },
#                                             },
#                       
#                       'prepareParameters': {
#                                             'src': 'clusters/templates',
#                                             'normalization': None,
#                                             'subpattern': False,
#                                             'number_mean_waves': 0,
#                                             'number_median_waves': 0,
#                                             'quantization': -1,
#                                             'patterns2strings': False,
#                                             },
#                       },
#             
#             'test': {
#                      'tags': ['Sitting', 'T1'],
#                       
#                       'processing_sequence': ['outlier'],
#                       
#                       'min_nb_templates': 3,
#                       
#                       'number_runs': 1,
#                       
#                       'filter': {
#                                  'src': '',
#                                  'method': 'zee5to20',
#                                  'parameters': {},
#                                  },
#                       
#                       'segment': {
#                                   'src': '',
#                                   'method': 'engzee',
#                                   'parameters': {},
#                                   },
#                       
#                       'outlier': {
#                                   'src': 'segments',
##                                   'method': 'dbscan',
##                                   'parameters': {
##                                                  'metric': 'euclidean',
##                                                  'min_samples': 10,
##                                                  'eps': 0.9,
##                                                  },
#                                   'method': 'dmean',
#                                   'parameters': {
#                                                  'metric': 'cosine',
#                                                  'alpha': 0.5,
#                                                  'R_Position': 200,
#                                                  },
#                                   },
#                       
#                       'prepareParameters': {
#                                             'src': 'outliers/templates',
#                                             'normalization': None,
#                                             'subpattern': False,
#                                             'number_mean_waves': 5,
#                                             'number_median_waves': 0,
#                                             'quantization': -1,
#                                             'patterns2strings': False,
#                                             },
#                      },
#             
#             'dimreduction': {
#                              'method': '',
#                              'parameters': {
#                                             },
#                              },
#             
#             'classifier': {
#                            'clf-knn': {
#                                        'method': 'knn',
#                                        'rejection_thresholds': np.arange(0., 2001, 1.).tolist(), # segments
#                                        'parameters': {
#                                                       'k': 3,
#                                                       'metric': 'euclidean',
#                                                       },
#                                        },
#                            
#                            'clf-svm': {
#                                        'method': 'svm',
#                                        'rejection_thresholds': np.arange(0.3, 1.05, 0.05).tolist(),
#                                        'parameters': {
#                                                       'kernel': 'linear',
#                                                       'C': 1.0,
#                                                       'class_weight': 'auto',
#                                                       'whiten': False,
#                                                       },
#                                        },
#                            
#                            },
#             }


#expConfig = {'database': {
#                          'source': 'StorageBIT',
#                          'DBConnection': {
#                                           'dbName': 'CVP',
#                                           'host': '193.136.222.234',
#                                           'dstPath': '',
#                                           'sync': False
#                                           },
#                          'experiments': [
#                                          'T1-Sitting',
#                                          'T2-Sitting'
#                                          ],
#                          'mapper': {
#                                     'raw': 'signals/ECG/hand/raw',
#                                     'filtered': 'signals/ECG/hand/zee5to20',
#                                     'segments': 'signals/ECG/hand/zee5to20/Segments/engzee',
#                                     'R': 'events/ECG/hand/zee5to20/R/engzee',
#                                     }
#                          },
#             
#             'starting_data': {
#                               'data_type': 'segments',
#                               'refine': None,
#                               'tasks': True,
#                               'mdataSplit': True,
#                               'concatenate': False,
#                               'axis': 0,
#                              },
#             
#             'train': {
#                       'tags': ['Sitting', 'T1'],
#                       
#                       'processing_sequence': [],
#                       
#                       'min_nb_templates': 3,
#                       
#                       'number_runs': 1,
#                       
#                       'filter': {
#                                  'src': '',
#                                  'method': 'zee5to20',
#                                  'parameters': {},
#                                  },
#                       
#                       'segment': {
#                                   'src': '',
#                                   'method': 'engzee',
#                                   'parameters': {},
#                                   },
#                       
#                       'prepareParameters': {
#                                             'src': 'segments',
#                                             'normalization': None,
#                                             'subpattern': False,
#                                             'number_mean_waves': 0,
#                                             'number_median_waves': 0,
#                                             'quantization': -1,
#                                             'patterns2strings': False,
#                                             },
#                       },
#             
#             'test': {
#                      'tags': ['Sitting', 'T1'],
#                       
#                       'processing_sequence': [],
#                       
#                       'min_nb_templates': 3,
#                       
#                       'number_runs': 1,
#                       
#                       'filter': {
#                                  'src': '',
#                                  'method': 'zee5to20',
#                                  'parameters': {},
#                                  },
#                       
#                       'segment': {
#                                   'src': '',
#                                   'method': 'engzee',
#                                   'parameters': {},
#                                   },
#                       
#                       'prepareParameters': {
#                                             'src': 'segments',
#                                             'normalization': None,
#                                             'subpattern': False,
#                                             'number_mean_waves': 0,
#                                             'number_median_waves': 0,
#                                             'quantization': -1,
#                                             'patterns2strings': False,
#                                             },
#                      },
#             
#             'dimreduction': {
#                              'method': '',
#                              'parameters': {},
#                              },
#             
#             'classifier': {
#                            'clf-knn': {
#                                        'method': 'knn',
#                                        'rejection_thresholds': np.arange(0., 2001, 2.).tolist(), # segments
#                                        'parameters': {
#                                                       'k': 3,
#                                                       'metric': 'euclidean',
#                                                       'io': ('classifiers', 'clf-knn', 'trainingData'),
#                                                       },
#                                        },
#                            
##                            'clf-svm': {
##                                        'method': 'svm',
##                                        'rejection_thresholds': np.arange(0.3, 1.05, 0.05).tolist(),
##                                        'parameters': {
##                                                       'kernel': 'linear',
##                                                       'C': 1.0,
##                                                       'class_weight': 'auto',
##                                                       'whiten': False,
##                                                       },
##                                        },
##                            
##                            'clf-fisher': {
##                                        'method': 'fisher',
##                                        'rejection_thresholds': np.arange(0.3, 1.05, 0.05).tolist(),
##                                        'parameters': {
##                                                       'cStep': 0.1,
##                                                       },
##                                        },
#                            
#                            },
#             }


#expConfig = {'database': {
#                          'source': 'StorageBIT',
#                          'DBConnection': {
#                                           'dbName': 'CVP',
#                                           'host': '193.136.222.234',
#                                           'dstPath': '',
#                                           'sync': False
#                                           },
#                          'experiments': [
#                                          'T1-Sitting',
#                                          'T2-Sitting'
#                                          ],
#                          'mapper': {
#                                     'raw': 'signals/ECG/hand/raw',
#                                     'filtered': 'signals/ECG/hand/zee5to20',
#                                     'segments': 'signals/ECG/hand/zee5to20/Segments/engzee',
#                                     'R': 'events/ECG/hand/zee5to20/R/engzee',
#                                     }
#                          },
#             
#             'starting_data': {
#                               'data_type': 'segments',
#                               'refine': None,
#                               'tasks': True,
#                               'mdataSplit': True,
#                               'concatenate': False,
#                               'axis': 0,
#                              },
#             
#             'train': {
#                       'tags': None,
#                       
#                       'processing_sequence': ['outlier'],
#                       
#                       'min_nb_templates': 3,
#                       
#                       'number_runs': 1,
#                       
#                       'filter': {
#                                  'src': '',
#                                  'method': 'zee5to20',
#                                  'parameters': {},
#                                  },
#                       
#                       'segment': {
#                                   'src': '',
#                                   'method': 'engzee',
#                                   'parameters': {},
#                                   },
#                       
#                       'outlier': {
#                                   'src': 'segments',
#                                   'method': 'dmean',
#                                   'parameters': {
#                                                  'metric': 'cosine',
#                                                  'alpha': 0.5,
#                                                  'R_Position': 200,
#                                                  },
#                                   },
#                       
#                       'prepareParameters': {
#                                             'src': 'outliers/templates',
#                                             'normalization': None,
#                                             'subpattern': False,
#                                             'number_mean_waves': 0,
#                                             'number_median_waves': 5,
#                                             'quantization': -1,
#                                             'patterns2strings': False,
#                                             },
#                       },
#             
#             'test': {
#                      'tags': None,
#                       
#                       'processing_sequence': ['outlier'],
#                       
#                       'min_nb_templates': 3,
#                       
#                       'number_runs': 1,
#                       
#                       'filter': {
#                                  'src': '',
#                                  'method': 'zee5to20',
#                                  'parameters': {},
#                                  },
#                       
#                       'segment': {
#                                   'src': '',
#                                   'method': 'engzee',
#                                   'parameters': {},
#                                   },
#                      
#                      'outlier': {
#                                   'src': 'segments',
#                                   'method': 'dmean',
#                                   'parameters': {
#                                                  'metric': 'cosine',
#                                                  'alpha': 0.5,
#                                                  'R_Position': 200,
#                                                  },
#                                   },
#                       
#                       'prepareParameters': {
#                                             'src': 'outliers/templates',
#                                             'normalization': None,
#                                             'subpattern': False,
#                                             'number_mean_waves': 0,
#                                             'number_median_waves': 5,
#                                             'quantization': -1,
#                                             'patterns2strings': False,
#                                             },
#                      },
#             
#             'dimreduction': {
#                              'method': '',
#                              'parameters': {},
#                              },
#             
#             'classifier': {
##                            'clf-knn': {
##                                        'method': 'knn',
##                                        'rejection_thresholds': np.arange(0., 2001, 2.).tolist(), # segments
##                                        'parameters': {
##                                                       'k': 3,
##                                                       'metric': 'euclidean',
##                                                       'io': ('classifiers', 'clf-knn', 'trainingData'),
##                                                       },
##                                        },
#                            },
#             }

expConfig = {'database': {
                          'source': 'StorageBIT',
                          'DBConnection': {
                                           'dbName': 'CVP',
                                           'host': '193.136.222.234',
                                           'dstPath': '',
                                           'sync': False
                                           },
                          'experiments': [
                                          'T1-Sitting',
                                          'T2-Sitting'
                                          ],
                          'mapper': {
                                     'raw': 'signals/ECG/hand/raw',
                                     'filtered': 'signals/ECG/hand/zee5to20',
                                     'segments': 'signals/ECG/hand/zee5to20/Segments/engzee',
                                     'R': 'events/ECG/hand/zee5to20/R/engzee',
                                     }
                          },
             
             'starting_data': {
                               'data_type': 'segments',
                               'refine': None,
                               'tasks': True,
                               'mdataSplit': True,
                               'concatenate': False,
                               'axis': 0,
                              },
             
             'train': {
                       'tags': None,
                       
                       'processing_sequence': ['outlier'],
                       
                       'min_nb_templates': 3,
                       
                       'number_runs': 1,
                       
                       'filter': {
                                  'src': '',
                                  'method': 'zee5to20',
                                  'parameters': {},
                                  },
                       
                       'segment': {
                                   'src': '',
                                   'method': 'engzee',
                                   'parameters': {},
                                   },
                       
                       'outlier': {
                                   'src': 'segments',
                                   'method': 'dmean',
                                   'parameters': {
                                                  'metric': 'cosine',
                                                  'alpha': 0.5,
                                                  'R_Position': 200,
                                                  },
                                   },
                       
                       'prepareParameters': {
                                             'src': 'outliers/templates',
                                             'normalization': None,
                                             'subpattern': False,
                                             'number_mean_waves': 5,
                                             'number_median_waves': 0,
                                             'quantization': -1,
                                             'patterns2strings': False,
                                             },
                       },
             
             'test': {
                      'tags': None,
                       
                       'processing_sequence': ['outlier'],
                       
                       'min_nb_templates': 3,
                       
                       'number_runs': 1,
                       
                       'filter': {
                                  'src': '',
                                  'method': 'zee5to20',
                                  'parameters': {},
                                  },
                       
                       'segment': {
                                   'src': '',
                                   'method': 'engzee',
                                   'parameters': {},
                                   },
                      
                      'outlier': {
                                   'src': 'segments',
                                   'method': 'dmean',
                                   'parameters': {
                                                  'metric': 'cosine',
                                                  'alpha': 0.5,
                                                  'R_Position': 200,
                                                  },
                                   },
                       
                       'prepareParameters': {
                                             'src': 'outliers/templates',
                                             'normalization': None,
                                             'subpattern': False,
                                             'number_mean_waves': 5,
                                             'number_median_waves': 0,
                                             'quantization': -1,
                                             'patterns2strings': False,
                                             },
                      },
             
             'dimreduction': {
                              'method': '',
                              'parameters': {},
                              },
             
             'classifier': {
                            'clf-knn': {
                                        'method': 'knn',
                                        'rejection_thresholds': np.arange(0., 2001, 2.).tolist(), # segments
                                        'parameters': {
                                                       'k': 1,
                                                       'metric': 'euclidean',
                                                       'io': ('classifiers', 'clf-knn', 'trainingData'),
                                                       },
                                        },
                            },
             }



if __name__ == '__main__':
    
    taskList = []
    
    # T1 vs T1
    task = copy.deepcopy(expConfig)
    task['train']['tags'] = ['T1', 'Sitting']
    task['test']['tags'] = ['T1', 'Sitting']
    taskList.append(task)
    
    # T1 vs T2
    task = copy.deepcopy(expConfig)
    task['train']['tags'] = ['T1', 'Sitting']
    task['test']['tags'] = ['T2', 'Sitting']
    taskList.append(task)
    
#    svmTmpl = {'method': 'svm',
#               'rejection_thresholds': np.arange(0.3, 1.05, 0.05).tolist(),
#               'parameters': {
#                              'kernel': 'linear',
#                              'C': 1.0,
#                              'class_weight': 'auto',
#                              'whiten': False,
#                              },
#               }
#    
#    C = map(lambda i: 10.**i, np.arange(1, 4, 0.5))
#    
#    knnTmpl = {'method': 'knn',
#               'rejection_thresholds': np.arange(0., 2001, 2.).tolist(), # segments
#               'parameters': {'k': 3,
#                              'metric': 'euclidean',
#                              'io': ('classifiers', 'clf-knn', 'trainingData'),
#                              },
#               }
#    
#    K = [1, 5, 7, 9]
#    expConfig['train']['min_nb_templates'] = 9
#    expConfig['test']['min_nb_templates'] = 9
#    
#    # all, P, QRS, T, PQRS, QRST
#    subpatterns = [None, [0, 120], [120, 300], [300, 600], [0, 300], [120, 600]]
#    T1 = ['Sitting', 'T1']
#    T2 = ['Sitting', 'T2']
#    TTSets = [{'train': T1, 'test': T2}, {'train': T2, 'test': T1}]
#    
#    for ts in TTSets:
#        for s in subpatterns:
#            task = copy.deepcopy(expConfig)
#            task['train']['tags'] = ts['train']
#            task['test']['tags'] = ts['test']
#            task['train']['prepareParameters']['subpattern'] = s
#            task['test']['prepareParameters']['subpattern'] = s
#            
#            for i in xrange(len(K)):
#                clf = copy.deepcopy(knnTmpl)
#                clf['parameters']['k'] = K[i]
#                task['classifier']['clf-knn-%d' % i] = clf
##            for i in xrange(len(C)):
##                clf = copy.deepcopy(svmTmpl)
##                clf['parameters']['C'] = C[i]
##                task['classifier']['clf-svm-%d' % i] = clf
#            
#            taskList.append(task)
    
    
    
#    # all T1 to train, all T1 to test
#    task = copy.deepcopy(expConfig)
#    taskList.append(task)
#    
#    # all T1 to train, all T2 to test
#    task = copy.deepcopy(expConfig)
#    task['test']['tags'] = ['Sitting', 'T2']
#    taskList.append(task)
#    
#    # all T2 to train, all T1 to test
#    task = copy.deepcopy(expConfig)
#    task['train']['tags'] = ['Sitting', 'T2']
#    taskList.append(task)
#    
#    # all T2 to train, all T2 to test
#    task = copy.deepcopy(expConfig)
#    task['train']['tags'] = ['Sitting', 'T2']
#    task['test']['tags'] = ['Sitting', 'T2']
#    taskList.append(task)
#    
#    outlierConfig = {
#                     'src': 'segments',
#                     'method': 'dmean',
#                     'parameters': {
#                                    'metric': 'cosine',
#                                    'alpha': 0.5,
#                                    'R_Position': 200,
#                                    },
#                     }
#    
#    # T1 (no outliers) to train, all T1 to test
#    task = copy.deepcopy(expConfig)
#    task['train']['processing_sequence'] = ['outlier']
#    task['train']['outlier'] = copy.deepcopy(outlierConfig)
#    task['train']['prepareParameters']['src'] = 'outliers/templates'
#    taskList.append(task)
#    
#    # T1 (no outliers) to train, all T2 to test
#    task = copy.deepcopy(expConfig)
#    task['train']['processing_sequence'] = ['outlier']
#    task['train']['outlier'] = copy.deepcopy(outlierConfig)
#    task['train']['prepareParameters']['src'] = 'outliers/templates'
#    task['test']['tags'] = ['Sitting', 'T2']
#    taskList.append(task)
#    
#    # T2 (no outliers) to train, all T1 to test
#    task = copy.deepcopy(expConfig)
#    task['train']['tags'] = ['Sitting', 'T2']
#    task['train']['processing_sequence'] = ['outlier']
#    task['train']['outlier'] = copy.deepcopy(outlierConfig)
#    task['train']['prepareParameters']['src'] = 'outliers/templates'
#    taskList.append(task)
#    
#    # T2 (no outliers) to train, all T2 to test
#    task = copy.deepcopy(expConfig)
#    task['train']['tags'] = ['Sitting', 'T2']
#    task['train']['processing_sequence'] = ['outlier']
#    task['train']['outlier'] = copy.deepcopy(outlierConfig)
#    task['train']['prepareParameters']['src'] = 'outliers/templates'
#    task['test']['tags'] = ['Sitting', 'T2']
#    taskList.append(task)
#    
#    # T1 (no outliers) to train, T1 (no outliers) to test
#    task = copy.deepcopy(expConfig)
#    task['train']['processing_sequence'] = ['outlier']
#    task['train']['outlier'] = copy.deepcopy(outlierConfig)
#    task['train']['prepareParameters']['src'] = 'outliers/templates'
#    task['test']['processing_sequence'] = ['outlier']
#    task['test']['outlier'] = copy.deepcopy(outlierConfig)
#    task['test']['prepareParameters']['src'] = 'outliers/templates'
#    taskList.append(task)
#    
#    # T1 (no outliers) to train, T2 (no outliers) to test
#    task = copy.deepcopy(expConfig)
#    task['train']['processing_sequence'] = ['outlier']
#    task['train']['outlier'] = copy.deepcopy(outlierConfig)
#    task['train']['prepareParameters']['src'] = 'outliers/templates'
#    task['test']['tags'] = ['Sitting', 'T2']
#    task['test']['processing_sequence'] = ['outlier']
#    task['test']['outlier'] = copy.deepcopy(outlierConfig)
#    task['test']['prepareParameters']['src'] = 'outliers/templates'
#    taskList.append(task)
#    
#    # T2 (no outliers) to train, T1 (no outliers) to test
#    task = copy.deepcopy(expConfig)
#    task['train']['tags'] = ['Sitting', 'T2']
#    task['train']['processing_sequence'] = ['outlier']
#    task['train']['outlier'] = copy.deepcopy(outlierConfig)
#    task['train']['prepareParameters']['src'] = 'outliers/templates'
#    task['test']['processing_sequence'] = ['outlier']
#    task['test']['outlier'] = copy.deepcopy(outlierConfig)
#    task['test']['prepareParameters']['src'] = 'outliers/templates'
#    taskList.append(task)
#    
#    # T2 (no outliers) to train, T2 (no outliers) to test
#    task = copy.deepcopy(expConfig)
#    task['train']['tags'] = ['Sitting', 'T2']
#    task['train']['processing_sequence'] = ['outlier']
#    task['train']['outlier'] = copy.deepcopy(outlierConfig)
#    task['train']['prepareParameters']['src'] = 'outliers/templates'
#    task['test']['tags'] = ['Sitting', 'T2']
#    task['test']['processing_sequence'] = ['outlier']
#    task['test']['outlier'] = copy.deepcopy(outlierConfig)
#    task['test']['prepareParameters']['src'] = 'outliers/templates'
#    taskList.append(task)
#    
#    # T1 (no outliers, mean waves) to train, T1 (no outliers, mean waves) to test
#    task = copy.deepcopy(expConfig)
#    task['train']['processing_sequence'] = ['outlier']
#    task['train']['outlier'] = copy.deepcopy(outlierConfig)
#    task['train']['prepareParameters']['src'] = 'outliers/templates'
#    task['train']['prepareParameters']['number_mean_waves'] = 5
#    task['test']['processing_sequence'] = ['outlier']
#    task['test']['outlier'] = copy.deepcopy(outlierConfig)
#    task['test']['prepareParameters']['src'] = 'outliers/templates'
#    task['test']['prepareParameters']['number_mean_waves'] = 5
#    taskList.append(task)
#    
#    # T1 (no outliers, mean waves) to train, T2 (no outliers, mean waves) to test
#    task = copy.deepcopy(expConfig)
#    task['train']['processing_sequence'] = ['outlier']
#    task['train']['outlier'] = copy.deepcopy(outlierConfig)
#    task['train']['prepareParameters']['src'] = 'outliers/templates'
#    task['train']['prepareParameters']['number_mean_waves'] = 5
#    task['test']['tags'] = ['Sitting', 'T2']
#    task['test']['processing_sequence'] = ['outlier']
#    task['test']['outlier'] = copy.deepcopy(outlierConfig)
#    task['test']['prepareParameters']['src'] = 'outliers/templates'
#    task['test']['prepareParameters']['number_mean_waves'] = 5
#    taskList.append(task)
#    
#    # T2 (no outliers, mean waves) to train, T1 (no outliers, mean waves) to test
#    task = copy.deepcopy(expConfig)
#    task['train']['tags'] = ['Sitting', 'T2']
#    task['train']['processing_sequence'] = ['outlier']
#    task['train']['outlier'] = copy.deepcopy(outlierConfig)
#    task['train']['prepareParameters']['src'] = 'outliers/templates'
#    task['train']['prepareParameters']['number_mean_waves'] = 5
#    task['test']['processing_sequence'] = ['outlier']
#    task['test']['outlier'] = copy.deepcopy(outlierConfig)
#    task['test']['prepareParameters']['src'] = 'outliers/templates'
#    task['test']['prepareParameters']['number_mean_waves'] = 5
#    taskList.append(task)
#    
#    # T2 (no outliers, mean waves) to train, T2 (no outliers, mean waves) to test
#    task = copy.deepcopy(expConfig)
#    task['train']['tags'] = ['Sitting', 'T2']
#    task['train']['processing_sequence'] = ['outlier']
#    task['train']['outlier'] = copy.deepcopy(outlierConfig)
#    task['train']['prepareParameters']['src'] = 'outliers/templates'
#    task['train']['prepareParameters']['number_mean_waves'] = 5
#    task['test']['tags'] = ['Sitting', 'T2']
#    task['test']['processing_sequence'] = ['outlier']
#    task['test']['outlier'] = copy.deepcopy(outlierConfig)
#    task['test']['prepareParameters']['src'] = 'outliers/templates'
#    task['test']['prepareParameters']['number_mean_waves'] = 5
#    taskList.append(task)
#    
#    # T1 (no outliers, median waves) to train, T1 (no outliers, median waves) to test
#    task = copy.deepcopy(expConfig)
#    task['train']['processing_sequence'] = ['outlier']
#    task['train']['outlier'] = copy.deepcopy(outlierConfig)
#    task['train']['prepareParameters']['src'] = 'outliers/templates'
#    task['train']['prepareParameters']['number_median_waves'] = 5
#    task['test']['processing_sequence'] = ['outlier']
#    task['test']['outlier'] = copy.deepcopy(outlierConfig)
#    task['test']['prepareParameters']['src'] = 'outliers/templates'
#    task['test']['prepareParameters']['number_median_waves'] = 5
#    taskList.append(task)
#    
#    # T1 (no outliers, median waves) to train, T2 (no outliers, median waves) to test
#    task = copy.deepcopy(expConfig)
#    task['train']['processing_sequence'] = ['outlier']
#    task['train']['outlier'] = copy.deepcopy(outlierConfig)
#    task['train']['prepareParameters']['src'] = 'outliers/templates'
#    task['train']['prepareParameters']['number_median_waves'] = 5
#    task['test']['tags'] = ['Sitting', 'T2']
#    task['test']['processing_sequence'] = ['outlier']
#    task['test']['outlier'] = copy.deepcopy(outlierConfig)
#    task['test']['prepareParameters']['src'] = 'outliers/templates'
#    task['test']['prepareParameters']['number_median_waves'] = 5
#    taskList.append(task)
#    
#    # T2 (no outliers, median waves) to train, T1 (no outliers, median waves) to test
#    task = copy.deepcopy(expConfig)
#    task['train']['tags'] = ['Sitting', 'T2']
#    task['train']['processing_sequence'] = ['outlier']
#    task['train']['outlier'] = copy.deepcopy(outlierConfig)
#    task['train']['prepareParameters']['src'] = 'outliers/templates'
#    task['train']['prepareParameters']['number_median_waves'] = 5
#    task['test']['processing_sequence'] = ['outlier']
#    task['test']['outlier'] = copy.deepcopy(outlierConfig)
#    task['test']['prepareParameters']['src'] = 'outliers/templates'
#    task['test']['prepareParameters']['number_median_waves'] = 5
#    taskList.append(task)
#    
#    # T2 (no outliers, median waves) to train, T2 (no outliers, median waves) to test
#    task = copy.deepcopy(expConfig)
#    task['train']['tags'] = ['Sitting', 'T2']
#    task['train']['processing_sequence'] = ['outlier']
#    task['train']['outlier'] = copy.deepcopy(outlierConfig)
#    task['train']['prepareParameters']['src'] = 'outliers/templates'
#    task['train']['prepareParameters']['number_median_waves'] = 5
#    task['test']['tags'] = ['Sitting', 'T2']
#    task['test']['processing_sequence'] = ['outlier']
#    task['test']['outlier'] = copy.deepcopy(outlierConfig)
#    task['test']['prepareParameters']['src'] = 'outliers/templates'
#    task['test']['prepareParameters']['number_median_waves'] = 5
#    taskList.append(task)
    
    conn = client.connection(host='193.136.222.235', user='ccarreiras', passwd='capcarr')
    
    for item in taskList:
        expId = conn.addTask(item)
        print expId
    
    conn.close()
    

#    taskList = []
#    # templates T1 to train, mean waves T1 to test
#    # Exp 0
#    task = copy.deepcopy(expConfig)
#    taskList.append(task)
#    
#    # Exp 1
#    task = copy.deepcopy(expConfig)
#    task['train']['templateSelection']['method'] = 'centroids'
#    taskList.append(task)
#    
#    # Exp 2
#    task = copy.deepcopy(expConfig)
#    task['train']['cluster']['parameters']['method'] = 'complete'
#    taskList.append(task)
#    
#    # Exp 3
#    task = copy.deepcopy(expConfig)
#    task['train']['cluster']['parameters']['method'] = 'complete'
#    task['train']['templateSelection']['method'] = 'centroids'
#    taskList.append(task)
#    
#    # Exp 5
#    task = copy.deepcopy(expConfig)
#    task['train']['processing_sequence'] = ['outlier', 'featureSelection', 'templateSelection']
#    task['train']['cluster']['method'] = ''
#    task['train']['cluster']['parameters'] = {}
#    taskList.append(task)
#    
#    
#    # templates T1 to train, mean waves T2 to test
#    expConfig2 = copy.deepcopy(expConfig)
#    expConfig2['database']['experiments'] = ['T1-Sitting', 'T2-Sitting']
#    expConfig2['test']['tags'] = ['Sitting', 'T2']
#    
#    # Exp 6
#    taskList.append(expConfig2)
#    
#    # Exp 7
#    task = copy.deepcopy(expConfig2)
#    task['train']['templateSelection']['method'] = 'centroids'
#    taskList.append(task)
#    
#    # Exp 8
#    task = copy.deepcopy(expConfig2)
#    task['train']['cluster']['parameters']['method'] = 'complete'
#    taskList.append(task)
#    
#    # Exp 9
#    task = copy.deepcopy(expConfig2)
#    task['train']['cluster']['parameters']['method'] = 'complete'
#    task['train']['templateSelection']['method'] = 'centroids'
#    taskList.append(task)
#    
#    # Exp 10
#    task = copy.deepcopy(expConfig2)
#    task['train']['processing_sequence'] = ['outlier', 'featureSelection', 'templateSelection']
#    task['train']['cluster']['method'] = ''
#    task['train']['cluster']['parameters'] = {}
#    taskList.append(task)
#    
#    
#    # templates T1+T2 to train, mean waves T1 to test
#    expConfig2 = copy.deepcopy(expConfig)
#    expConfig2['database']['experiments'] = ['T1-Sitting', 'T2-Sitting']
#    expConfig2['train']['tags'] = ['Sitting']
#    expConfig2['test']['tags'] = ['Sitting', 'T1']
#    
#    # Exp 11
#    taskList.append(expConfig2)
#    
#    # Exp 12
#    task = copy.deepcopy(expConfig2)
#    task['train']['templateSelection']['method'] = 'centroids'
#    taskList.append(task)
#    
#    # Exp 13
#    task = copy.deepcopy(expConfig2)
#    task['train']['cluster']['parameters']['method'] = 'complete'
#    taskList.append(task)
#    
#    # Exp 14
#    task = copy.deepcopy(expConfig2)
#    task['train']['cluster']['parameters']['method'] = 'complete'
#    task['train']['templateSelection']['method'] = 'centroids'
#    taskList.append(task)
#    
#    # Exp 15
#    task = copy.deepcopy(expConfig2)
#    task['train']['processing_sequence'] = ['outlier', 'featureSelection', 'templateSelection']
#    task['train']['cluster']['method'] = ''
#    task['train']['cluster']['parameters'] = {}
#    taskList.append(task)
#    
#    # templates T1+T2 to train, mean waves T2 to test
#    expConfig2 = copy.deepcopy(expConfig)
#    expConfig2['database']['experiments'] = ['T1-Sitting', 'T2-Sitting']
#    expConfig2['train']['tags'] = ['Sitting']
#    expConfig2['test']['tags'] = ['Sitting', 'T2']
#    
#    # Exp 16
#    taskList.append(expConfig2)
#    
#    # Exp 17
#    task = copy.deepcopy(expConfig2)
#    task['train']['templateSelection']['method'] = 'centroids'
#    taskList.append(task)
#    
#    # Exp 18
#    task = copy.deepcopy(expConfig2)
#    task['train']['cluster']['parameters']['method'] = 'complete'
#    taskList.append(task)
#    
#    # Exp 19
#    task = copy.deepcopy(expConfig2)
#    task['train']['cluster']['parameters']['method'] = 'complete'
#    task['train']['templateSelection']['method'] = 'centroids'
#    taskList.append(task)
#    
#    # Exp 20
#    task = copy.deepcopy(expConfig2)
#    task['train']['processing_sequence'] = ['outlier', 'featureSelection', 'templateSelection']
#    task['train']['cluster']['method'] = ''
#    task['train']['cluster']['parameters'] = {}
#    taskList.append(task)
#    
    
#    conn = client.connection(host='193.136.222.235', user='ccarreiras', passwd='capcarr')
#    
#    for item in taskList:
#        expId = conn.addTask(item)
#        print expId
#    
#    conn.close()
    
#    path = 'Z:\\BiometricsExperiments'
    
#    conn = client.connection(host='193.136.222.235', user='ccarreiras', passwd='capcarr')
#    
#    assembleTables(conn, path, range(126, 270))
#    
#    conn.close()
#    
#    res = makeMeans(path, label='EER')
    
#    fig = plotTemplates(path, 131, 1)
#    visualization.show()
#    A = [126, 127, 128, 129, 130, 146, 147, 148, 149, 150, 166, 167, 168, 169, 170,
#         186, 187, 188, 189, 190, 206, 207, 208, 209, 210, 226, 227, 228, 229, 230,
#         246, 250, 254, 258, 262, 266]
#    
#    B = [131, 132, 133, 134, 135, 151, 152, 153, 154, 155, 171, 172, 173, 174, 175,
#         191, 192, 193, 194, 195, 211, 212, 213, 214, 215, 231, 232, 233, 234, 235,
#         247, 251, 255, 259, 263, 267]
#    
#    C = [136, 137, 138, 139, 140, 156, 157, 158, 159, 160, 176, 177, 178, 179, 180,
#         196, 197, 198, 199, 200, 216, 217, 218, 219, 220, 236, 237, 238, 239, 240,
#         248, 252, 256, 260, 264, 268]
#    
#    D = [141, 142, 143, 144, 145, 161, 162, 163, 164, 165, 181, 182, 183, 184, 185,
#         201, 202, 203, 204, 205, 221, 222, 223, 224, 225, 241, 242, 243, 244, 245,
#         249, 253, 257, 261, 265, 269]
#    conn = client.connection(host='193.136.222.235', user='ccarreiras', passwd='capcarr')
#    plotTemplates(conn, path, range(126, 270))
#    conn.close()
    
#    pretty = {
#              'outlier': 'Outlier Detection Method',
#              'clustering': 'Clustering Algorithm',
#              'templates-met': 'Template Selection Method',
#              'templates-n': 'Number of Templates',
#              'classifier': 'Classifier',
#              }
#    
#    case = 'D'
#    
#    # template selection method vs number of templates
#    selection = {
#                 'outlier': 'DMEAN',
#                 'classifier': 'SVM',
#                 }
#    fig = makeBarPlots(path, 'EER', 'templates-n', 'clustering', 'templates-met', case, selection, pretty, [0., 6.])
#    # save figure
#    fig.savefig(os.path.join(path, 'barPlot_%s_EER_tempMet_vs_tempN_dmean_svm.pdf' % case), bbox_inches='tight', dpi=500)
#    visualization.close(fig)
#    
#    fig = makeBarPlots(path, 'EID', 'templates-n', 'clustering', 'templates-met', case, selection, pretty, [0., 10.])
#    # save figure
#    fig.savefig(os.path.join(path, 'barPlot_%s_EID_tempMet_vs_tempN_dmean_svm.pdf' % case), bbox_inches='tight', dpi=500)
#    visualization.close(fig)
#    
#    
#    selection = {
#                 'outlier': 'DMEAN',
#                 'classifier': '$k$-NN',
#                 }
#    fig = makeBarPlots(path, 'EER', 'templates-n', 'clustering', 'templates-met', case, selection, pretty, [0., 6.])
#    # save figure
#    fig.savefig(os.path.join(path, 'barPlot_%s_EER_tempMet_vs_tempN_dmean_knn.pdf' % case), bbox_inches='tight', dpi=500)
#    visualization.close(fig)
#    
#    fig = makeBarPlots(path, 'EID', 'templates-n', 'clustering', 'templates-met', case, selection, pretty, [0., 10.])
#    # save figure
#    fig.savefig(os.path.join(path, 'barPlot_%s_EID_tempMet_vs_tempN_dmean_knn.pdf' % case), bbox_inches='tight', dpi=500)
#    visualization.close(fig)
#    
#    
#    
#    # outlier methods vs classifier
#    selection = {
#                 'clustering': 'Complete',
#                 'templates-met': 'Centroids',
#                 }
#    fig = makeBarPlots(path, 'EER', 'templates-n', 'classifier', 'outlier', case, selection, pretty, [0., 10.])
#    # save figure
#    fig.savefig(os.path.join(path, 'barPlot_%s_EER_outlier_vs_classifier_cpl_centroids.pdf' % case), bbox_inches='tight', dpi=500)
#    visualization.close(fig)
#    
#    fig = makeBarPlots(path, 'EID', 'templates-n', 'classifier', 'outlier', case, selection, pretty, [0., 14.])
#    # save figure
#    fig.savefig(os.path.join(path, 'barPlot_%s_EID_outlier_vs_classifier_cpl_centroids.pdf' % case), bbox_inches='tight', dpi=500)
#    visualization.close(fig)
#    
#    
#    
#    selection = {
#                 'clustering': 'Complete',
#                 'templates-met': 'Mdist',
#                 }
#    fig = makeBarPlots(path, 'EER', 'templates-n', 'classifier', 'outlier', case, selection, pretty, [0., 10.])
#    # save figure
#    fig.savefig(os.path.join(path, 'barPlot_%s_EER_outlier_vs_classifier_cpl_mdist.pdf' % case), bbox_inches='tight', dpi=500)
#    visualization.close(fig)
#    
#    fig = makeBarPlots(path, 'EID', 'templates-n', 'classifier', 'outlier', case, selection, pretty, [0., 14.])
#    # save figure
#    fig.savefig(os.path.join(path, 'barPlot_%s_EID_outlier_vs_classifier_cpl_mdist.pdf' % case), bbox_inches='tight', dpi=500)
#    visualization.close(fig)
#    
#    
#    # clustering vs number of templates
#    selection = {
#                 'outlier': 'DMEAN',
#                 'classifier': 'SVM',
#                 }
#    fig = makeBarPlots(path, 'EER', 'templates-n', 'templates-met', 'clustering', case, selection, pretty, [0., 6.])
#    # save figure
#    fig.savefig(os.path.join(path, 'barPlot_%s_EER_clustering_vs_tempN_dmean_svm.pdf' % case), bbox_inches='tight', dpi=500)
#    visualization.close(fig)
#    
#    fig = makeBarPlots(path, 'EID', 'templates-n', 'templates-met', 'clustering', case, selection, pretty, [0., 10.])
#    # save figure
#    fig.savefig(os.path.join(path, 'barPlot_%s_EID_clustering_vs_tempN_dmean_svm.pdf' % case), bbox_inches='tight', dpi=500)
#    visualization.close(fig)
#    
#    
#    selection = {
#                 'outlier': 'DMEAN',
#                 'classifier': '$k$-NN',
#                 }
#    fig = makeBarPlots(path, 'EER', 'templates-n', 'templates-met', 'clustering', case, selection, pretty, [0., 6.])
#    # save figure
#    fig.savefig(os.path.join(path, 'barPlot_%s_EER_clustering_vs_tempN_dmean_knn.pdf' % case), bbox_inches='tight', dpi=500)
#    visualization.close(fig)
#    
#    fig = makeBarPlots(path, 'EID', 'templates-n', 'templates-met', 'clustering', case, selection, pretty, [0., 10.])
#    # save figure
#    fig.savefig(os.path.join(path, 'barPlot_%s_EID_clustering_vs_tempN_dmean_knn.pdf' % case), bbox_inches='tight', dpi=500)
#    visualization.close(fig)
    
    
    
    
    



# fig = plt.figure()
#>>> ax = fig.add_subplot(111)
#>>> pcentroids = ax.plot(time, td_centroids.T, 'g', alpha=0.7, linewidth=2)
#>>> pmdist = ax.plot(time, td_mdist.T, 'r', alpha=0.7, linewidth=2)
#>>> ax.grid()
#>>> ax.set_xlabel('Time (ms)')
#<matplotlib.text.Text object at 0x000000000C7D19B0>
#>>> ax.set_ylabel('Amplitude')
#<matplotlib.text.Text object at 0x0000000008254198>
#ax.legend((pcentroids[0], pmdist[0]), ('Centroids', 'Mdist'), loc='best')
#>>> fig.savefig('C:/Users/Carlos/Desktop/centroids_vs_mdist_D_204_203_7.pdf', bbox_inches='tight', dpi=500)
#>>> plt.show()