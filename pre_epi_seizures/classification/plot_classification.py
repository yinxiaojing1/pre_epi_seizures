import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import seaborn as sns
import numpy as np
import scipy as sp
import os
# sns.set()


def get_color_list():
    colors = dict(mcolors.BASE_COLORS)
    print colors

    colors = [color for color in colors.keys()
          if color != 'k']

    # colors = sns.choose_colorbrewer_palette('sequential', as_cmap=False)

    return colors


def get_label_list(data, labels, colors):
    colors = data[colors]
    labels = data[labels]

    return labels.unique(), colors.unique()


def jointplot(data, x, y, labels, colors):
    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    label_list, color_list = get_label_list(data, labels, colors)


    label = 1
    color = 'g'
    data_to_plot = data.loc[data[labels] == label]
    data_to_plot_x = data_to_plot[x]
    data_to_plot_y = data_to_plot[y]
    print data_to_plot_x
    ax1.scatter(data_to_plot_x, data_to_plot_y, color=color, s=10)


    # label = 2
    # color = 'r'
    # data_to_plot = data.loc[data[labels] == label]
    # data_to_plot_x = data_to_plot[x]
    # data_to_plot_y = data_to_plot[y]
    # print data_to_plot_x
    # ax1.scatter(data_to_plot_x, data_to_plot_y, color=color, s=10)




    ax1.set_xlabel(x)
    ax1.set_ylabel(y)
    ax1.legend(label_list)

    print 'done'
    f1.savefig('joint')
    

def get_roc_curves(data_struct):

    return


def extend_roc_curve(mean_curve_fpr, roc_curve_list):
    ''' Recieves a list with FPR and corresponding TPR'''
    
    # loop for each computed FPR and Correspondig TPR
    mean_tpr_curve = []
    for roc_curve in roc_curve_list:
        
        # get fpr Curve
        roc_curve_fpr = np.asarray(roc_curve[0])
       
        # get tpr curve
        roc_curve_tpr = np.asarray(roc_curve[1])
        
        print roc_curve_fpr
        print roc_curve_tpr
         
        # interpolate with zero kind curve the Tpr points, based on
        # the computed fpr curve
        f = sp.interpolate.interp1d(roc_curve_fpr, roc_curve_tpr, kind='zero')
        
        # Compute interpolated tpr curve
        mean_curve_tpr_aux = f(mean_curve_fpr)
        mean_tpr_curve.append(mean_curve_tpr_aux)
    
    # Compute mean TPR curve
    mean_tpr_curve = np.asarray(mean_tpr_curve)
    mean_tpr_curve = np.mean(mean_tpr_curve, axis=0)
    mean_fpr_curve = mean_curve_fpr
    
    return mean_fpr_curve, mean_tpr_curve
        
        
            
def compute_mean_roc(roc_curve_list):
    ''' Recieves a list with FPR and corresponding TPR'''
    
    # Default Main FPR Curve
    mean_curve_fpr = np.linspace(0, 1, 100)
    
    # Extends The Main FPR Curve 
    mean_fpr_curve, mean_tpr_curve = extend_roc_curve(
                                        mean_curve_fpr,
                                        roc_curve_list)
    
    return mean_fpr_curve, mean_tpr_curve

def configure_figure(intreval, points_per_label, trial):        
    
    # employ default configuration
    plt.figure(figsize=(20,20))
    plt.title('ROC - ' + str(intreval) + '_' + str(points_per_label) + str(trial))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    
def plot_roc_new(path_to_save, intreval, points_per_label, nested_cross_struct, trial):
    
    # Employ default configuration
    configure_figure(intreval, points_per_label, trial)
    
    # Get color list
    colors = get_color_list()
    
    # allocate legend
    legend = []
    
    # allocate mean_curves
    mean_fpr = []
    mean_tpr = []
    
    for i, model in enumerate(nested_cross_struct):
        # get hyper_parameter optimization results
        cv_results = model['cv_results'][0]
                
        # get best model from optimization
        best_model = model['cv_results'][1]
        
        # get ROC
        ROC = model['ROC']
        
        # plot ROC curve
        fpr = ROC[0]['FPR']
        tpr = ROC[0]['TPR']
        plt.plot(fpr, tpr)
        
        # Build mean model
        mean_fpr.append(fpr)
        mean_tpr.append(tpr)
        
        # build legend
        legend.append(str(best_model) + '__' + str(i))
        
    # Plot chance boundary
    legend.append('chance')
    plt.plot([0, 1], [0, 1], 'k--')
    
    # Compute & Plot mean curve
    mean_roc_curve = compute_mean_roc(zip(mean_fpr, mean_tpr))
    plt.plot(mean_roc_curve[0], mean_roc_curve[1], 'k')
    
    # Finnish the figure configuration
    plt.legend(legend)
    plt.savefig(path_to_save + str(intreval) + '_' + str(points_per_label) + str(trial))
    plt.show()  
        
    # Close figure
    plt.close()


def plot_roc(path_to_save, intreval, points_per_label, nested_cross_struct, trial):
    # path_to_save = path_to_save + 'ROC/'
    plt.figure(figsize=(15, 7.5))
    plt.title('ROC - ' + str(intreval) + '_' + str(points_per_label) + str(trial))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    
    # Get color list
    colors = get_color_list()

    # Group by model
    data = nested_cross_struct.groupby(['model'])

    legend = []
    
    # Mean 
    curve = [sz[1][['FPR', 'TPR']]
             for dt in data
             for sz in dt[1].groupby(['nr_seizure'])
             ]
    
    
    compute_mean_roc(curve)
    legend.append('mean')
    
    # For each model
    for dt, color in zip(data, colors):

        model_name = dt[0]

        dt = dt[1]

        # For each tested seizure
        dt_sz = dt.groupby(['nr_seizure'])


        # set color pallete
        # sns.set_palette(sns.dark_palette(color, n_colors=20))
        # sns.set_palette(sns.color_palette('Blues'))

        # compute mean ROC
        #mean_fpr = [sz[1]['FPR'] for sz in dt_sz]
        #mean_fpr = sum(mean_fpr)/len(mean_fpr)

        #mean_tpr = [sz[1]['TPR'] for sz in dt_sz]
        #mean_tpr = sum(mean_tpr)/len(mean_tpr)


        # plot each test seizure
        # sns.set_palette(sns.dark_palette('k', n_colors=20))
        #plt.plot(mean_fpr, mean_tpr, 'k')

        #legend.append('mean --Best chance')
        for sz in dt_sz:
            fpr = sz[1]['FPR']
            tpr = sz[1]['TPR']
            plt.plot(fpr, tpr)
            legend.append(model_name + '__' + str(sz[0]))

        # print color
        # set color pallete
        # sns.set_palette(sns.dark_palette(color, n_colors= len(dt['nr_seizure'].unique())))


    print 'Done'

    # legend = [str(model['model'])
    #           for model in nested_cross_struct]

    # struct_total = pd.concat(nested_cross_struct)
    # # print list(struct)
    # print struct_total    # print list(struct)
    # print struct
    # # print nested_cross_struct[0]
    # stop
    legend.append('chance')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend(legend)
    plt.show()
    #stop
    #plt.savefig(path_to_save + str(intreval) + '_' + str(points_per_label) + str(trial))


def plot_scatter(path_to_save, data_struct, class_metadata):
    plt.figure()
    plt.title('Pairplot')
    sns.pairplot(data=data_struct.drop(class_metadata, axis=1))
    plt.savefig(path_to_save + 'Pair')


def plot_full(path_to_save, data_struct, class_metadata):
    # file_to_save = path_to_save + 'full/'


    for feature in data_struct.drop(class_metadata, axis=1).columns:
        labels = data_struct['labels']
        color = ['r', 'g']
        print feature
        for group in data_struct['group'].unique():
            print 'group'
            print group
            plt.figure()
            # plt.title('Distribution of feature: ' + feature + '_' + 'group: ' + str(group))
            for label, color in zip(labels.unique(), color):
                print 'label'
                print label
                plt.subplot(len(labels.unique()), 1, label)
                data = data_struct.loc[data_struct['labels']==label]
                data = data.loc[data['group']==group]
                data = data.drop(class_metadata, axis=1)
                plt.plot(data[feature], color=color)

            plt.savefig(file_to_save + '_' + feature + '_' + 'group: ' + str(group))


def plot_hist(file_to_save, intreval, points_per_label, data_struct, class_metadata):
    # file_to_save = path_to_save + 'hist/'



    for feature in data_struct.drop(class_metadata, axis=1).columns:
        plt.figure()
        plt.title('Distribution of feature: ' + feature)
        labels = data_struct['labels']
        color = ['r', 'g']

        for i, (label, color) in enumerate(zip(labels.unique(), color)):
            print label
            print i
            plt.subplot(len(labels.unique()), 1, i+1)
            data = data_struct.loc[data_struct['labels']==label]
            data = data.drop(class_metadata, axis=1)
            sns.distplot(data[feature], bins=np.linspace(-4, 4, 100), hist=True, kde=False, color=color)
            plt.xlim([-4, 4])

        # plt.legend(labels.unique())
        # plt.legend(['2', '1'])
        plt.show()
        plt.savefig(file_to_save + '_' + feature + '_' + str(intreval) + '_' + str(points_per_label))