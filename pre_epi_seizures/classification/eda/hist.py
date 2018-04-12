import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import utils.estimationfit as es

''' Module containing the various plot functions required to perform
exploratory statistical analysis, the level of abstraction is minimal,
requiring the datasets to be on pandas dataframe format'''


def histogram(path_to_save, 
              grouped_df,
              group_name,
              features_id,
              time_domain_id,
              patient_id,
              seizure_id,
              label_id,
              color_id,
              dist=None,
              bins=None,
              ):


    plt.figure(figsize=(20, 20))

    for i, feature in enumerate(features_id):
        if i == 0:
            
            if dist != None:
                plt.title(str(group_name) + '_' + dist +'__hist')
            else:
                plt.title(str(group_name) + '__hist')
                
        plt.subplot(len(features_id), 1, i+1)
        
        #iterate for each label
        labels = grouped_df[label_id].unique()
        colors = grouped_df[color_id].unique()
        colormap = zip(labels, colors)
        for i, label_color in enumerate(colormap):
                
                # get labels
                label = label_color[0]
                color = label_color[1]

                # get data from a certain label
                univariate_data_label = grouped_df[feature].loc[grouped_df[label_id]==label]

                # plot histogram
                try:
                    if dist=='kde':
                        sns.distplot(univariate_data_label, color=color,
                                     kde=True, bins=bins, norm_hist=True)
                        
                    if dist==None:
                        sns.distplot(univariate_data_label, color=color,
                                     kde=False, bins=bins, norm_hist=True)
                        
                    else:
                        sns.distplot(univariate_data_label, color=color,
                                     kde=False, bins=bins, norm_hist=True)
                        x, pdfitted = es.dist_estimation(univariate_data_label, dist)
                        plt.plot(x, pdfitted, color=color)

                except Exception as e:
                    print e
                    print color
                    print label
                    print 'not estimated'

                
    plt.savefig(path_to_save + str(group_name) + '__HIST.png')
    plt.show()
    plt.legend(labels)      