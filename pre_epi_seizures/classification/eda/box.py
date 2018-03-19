import matplotlib.pyplot as plt
import seaborn as sns

def box_plot(path_to_save, 
                  grouped_df,
                  title,
                  features_id,
                  time_domain_id,
                  patient_id,
                  seizure_id,
                  label_id,
                  color_id):

    for i, feature in enumerate(features_id):

        plt.subplot(len(features_id), 1, i+1)
        
        if i==0:
            plt.title(title)
            
        #iterate for each label
        labels = grouped_df[label_id].unique()
        colors = grouped_df[color_id].unique()
        colormap = zip(labels, colors)

        sns.boxplot(x='label',
                    y=feature,
                    data=grouped_df,
                    showfliers=False,
                    order=list(labels),
                    palette=list(colors))
