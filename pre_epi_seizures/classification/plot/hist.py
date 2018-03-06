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
              color_id):
