import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_box_univariate_per_label(full_path, X, y):

    # arrange structure
    columns =  X.columns
    data = X
    data['labels'] = y

    # Compute Univariate box plot

    for feature in columns:
        _plot_box_univariate_per_label(full_path, 
                                       data, feature)



def _plot_box_univariate_per_label(full_path, 
                                   data,
                                   feature):

    # Plot box
    plt.figure()
    # data.boxplot(column=feature, figsize=(20,20),
    #              by='labels')
    sns.boxplot(x='labels', y=feature, data=data,
                 showfliers=False)
    plt.savefig(full_path + '__' + feature)
    plt.show()
