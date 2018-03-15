import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

''' Module containing the various plot functions required to perform
exploratory statistical analysis, the level of abstraction is minimal,
requiring the datasets to be on pandas dataframe format'''


def histogram(path_to_save, 
              grouped_df,
              features_id,
              time_domain_id,
              patient_id,
              seizure_id,
              label_id,
              color_id,
              bins=None):



    for i, feature in enumerate(features_id):

        plt.subplot(len(features_id), 1, i+1)
        #iterate for each label
        labels = grouped_df[label_id].unique()
        colors = grouped_df[color_id].unique()
        colormap = zip(labels, colors)
        for i, label_color in enumerate(colormap):
            
                label = label_color[0]
                color = label_color[1]

                
                #plt.subplot(len(labels), 1, i + 1)
                # get data from a certain label
                univariate_data_label = grouped_df[feature].loc[grouped_df[label_id]==label]

                # plot histogram
                try:
                    sns.distplot(univariate_data_label, color=color,
                                 kde=False, bins=bins)
                
                except Exception as e:
                    print e
                    print color
                    print label
                    print 'not estimated'

                
                
        plt.legend(labels)      