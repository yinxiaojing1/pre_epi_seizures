import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import seaborn as sns
import numpy as np
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


def plot_roc(path_to_save, intreval, points_per_label, nested_cross_struct):
    # path_to_save = path_to_save + 'ROC/'
    plt.figure(figsize=(15, 7.5))
    plt.title('ROC - ' + str(intreval) + '_' + str(points_per_label))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # Get color list
    colors = get_color_list()

    # Group by model
    data = nested_cross_struct.groupby(['model'])

    legend = []

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
        # mean_fpr = [sz[1]['FPR'] for sz in dt_sz]
        # mean_fpr = sum(mean_fpr)/len(mean_fpr)

        # mean_tpr = [sz[1]['TPR'] for sz in dt_sz]
        # mean_tpr = sum(mean_tpr)/len(mean_tpr)


        # plot each test seizure
        # sns.set_palette(sns.dark_palette('k', n_colors=20))
        # plt.plot(mean_fpr, mean_tpr)

        # legend.append(model_name)
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
    plt.savefig(path_to_save + str(intreval) + '_' + str(points_per_label))


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

        for label, color in zip(labels.unique(), color):
            print label
            plt.subplot(len(labels.unique()), 1, label)
            data = data_struct.loc[data_struct['labels']==label]
            data = data.drop(class_metadata, axis=1)
            sns.distplot(data[feature], bins=np.linspace(-4, 4, 100), hist=True, kde=False, color=color)
            plt.xlim([-4, 4])

        # plt.legend(labels.unique())
        # plt.legend(['2', '1'])
        plt.savefig(file_to_save + '_' + feature + '_' + str(intreval) + '_' + str(points_per_label))