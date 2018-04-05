import seaborn as sns
import matplotlib.pyplot as plt


def pair_plot(path_to_save, 
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
    sns.pairplot(X, hue=label_id, hue_order=labels, palette=colors, size=2.5)

    # Save
    plt.title(title)
    plt.savefig(path_to_save + str(title) + '__' + 'PAIR')
    plt.show()
    
