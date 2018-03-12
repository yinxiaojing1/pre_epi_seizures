import os
import matplotlib.pyplot
import matplotlib.image as mpimg
import numpy as np


def plot(plot_function):

    def plot_wrapper(path_to_save, 
                     grouped_df,
                     title,
                     features_id,
                     time_domain_id,
                     patient_id,
                     seizure_id,
                     label_id,
                     color_id,
                     plot_flag):

        # Get name of the figure
        name='fig' + title + '.png'

        # Load figure if appropriate
        if os.path.exists(path_to_save + name) and not plot_flag:
            img=mpimg.imread(name)
            imgplot = plt.imshow(img)

        # Compute the figure again
        else:
            plot_function(path_to_save + name, 
                          grouped_df,
                          title,
                          features_id,
                          time_domain_id,
                          patient_id,
                          seizure_id,
                          label_id,
                          color_id)
