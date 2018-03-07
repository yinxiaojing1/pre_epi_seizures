import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

''' Module containing the various plot functions required to perform
exploratory statistical analysis, the level of abstraction is minimal,
requiring the datasets to be on pandas dataframe format'''


def time_series_plot(path_to_save, df,
                     features_id,
                     time_domain_id,
                     patient_id,
                     seizure_id,
                     label_id,
                     color_id):

    # Loop for patients
    patients = df[patient_id].unique()
    for patient in patients:
        df_patient = df.loc[df[patient_id] == patient]  # get data

        # Loop for seizures
        seizures = df_patient[seizure_id].unique()
        for seizure in seizures:
            df_seizure = df_patient.loc[
                df_patient[seizure_id] == seizure] # get data

            # Loop for features
            for feature in features_id:

                # Define figure
                plt.figure(figsize=(20, 20))
                save_str = ('Patient: ' + patient +
                            ' Seizure: ' + seizure)
                
                # Loop for labels
                labels = df_seizure[label_id].unique()
                colors = df_seizure[color_id].unique()
                colormap = zip(labels, colors)
                for i, label_color in enumerate(colormap):
                    label = label_color[0]
                    color = label_color[1]
                    df_label = df_seizure.loc[df_seizure[label_id] == label]
                    time_domain = df_label[time_domain_id]
                    time_series = df_label[feature]

                    # Plot data
                    plt.subplot(len(colormap), 1, i+1)
                    plt.plot(time_domain, time_series, color)
                    plt.ylabel(feature)
                    plt.legend([label])
                    
                    # Title
                    if i == 0:
                        plt.title(save_str)
                
                # Save figure
                
                plt.title(save_str)
                plt.show()
                plt.savefig(path_to_save + save_str)



