import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from plot import *
''' Module containing the various plot functions required to perform
exploratory statistical analysis, the level of abstraction is minimal,
requiring the datasets to be on pandas dataframe format'''


def andrews_curves(path_to_save, 
                  grouped_df,
                  title,
                  features_id,
                  time_domain_id,
                  patient_id,
                  seizure_id,
                  label_id,
                  color_id):


    # get colormap
    labels = grouped_df[label_id].unique()
    colors = grouped_df[color_id].unique()

    # Set up figure
    plt.figure(figsize=(20,20))

    # Try andrews curve
    X = grouped_df[features_id]
    X[label_id] = grouped_df[label_id]
    pd.plotting.andrews_curves(X, label_id, color=colors)

    # Save
    plt.title(title)
    plt.show()