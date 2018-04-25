
# coding: utf-8

# In[1]:


'''BE CAREFUL RUNNIG THIS SCRIPT CAN OVERWRIE THE WHOLE DATA ON DISK'''


# In[2]:


'''BE CAREFUL CHANGING THE LABELS STRUCTURE, MAKE SURE TO CHANGE classification/cross_validation.parse_classification_report'''


# In[3]:


# Imports
# Modelation
import sklearn.svm as svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from classification.keras_neural_nets import *

import os


# In[4]:


from learning_pipelines import supervised_pipeline


# In[ ]:


# LABELING STRUCTURE
label_struct = {
                        'inter_ictal':{
                            'label': 'Inter-Ictal Data Points',
                            'color': 'blue',
                            'intervals_samples': [(1000 * 0 * 60, 1000 * 20 * 60 )]
                                     },
                        'pre_ictal':{
                            'label': 'Pre-Ictal data points',
                            'color': 'yellow',
                            'intervals_samples': [(1000 * 20 * 60, 1000 * 49 * 60 )]
                                     },
                        'ictal':{
                            'label': 'Ictal data points',
                            'color': 'red',
                            'intervals_samples': [(1000 * 50 * 60, 1000 * 55 * 60 )]
                                 },
                        }
baseline_label_struct = {
                         'baseline':{
                            'label': 'Baseline Data Points',
                            'color': 'green',
                            'intervals_samples': [(0, 1000 * 30 * 60 )]
                                    },
                        }



param_grid = [
              {'ANN__epochs': [800],
               'ANN__batch_size': [100],
               'ANN__hidden_layers_nr': [1],
               'ANN__hidden_nodes_per_layer': [i for i in xrange(3, 13)],
               'ANN__hidden_nodes_activation': ['relu']
              }
]
pipe = Pipeline( [('ANN', KerasClassifier(build_fn=dense_network,
                                        input_dim = 7,
                                          output_dim = 3,
                                          verbose=0))])

pipe = Pipeline([('SVC', svm.SVC())])
param_grid = [{'SVC__C': [2**i for i in xrange(-5, 11)], 
                'SVC__gamma':[2**i for i in xrange(-15, 1)]}]

#pipe = Pipeline([('GaussNB', GaussianNB())])
#param_grid = [
#              {'GaussNB__priors': [None]},
#]

#pipe = Pipeline([('KNN', KNeighborsClassifier())])
#param_grid = [{'KNN__n_neighbors': [i for i in xrange(1, 15, 2)]}]




feature_slot = 'hrv_time_features'
hyper_param=0


scaler = StandardScaler()

plot_eda=False
learn_flag=True
compute_all_new=True


# In[ ]:




patient_lists = [[3], [5], [8], [13]]


for patient_list in patient_lists:

    supervised_pipeline(label_struct, baseline_label_struct,
                                                  pipe, scaler, param_grid,
                                                  patient_list,
                                                  feature_slot,
                                                  hyper_param,
                                                  plot_eda,
                                                  learn_flag,
                                                  compute_all_new,
                                                  n_jobs=-1)









# In[ ]:


del y


# In[ ]:


client.compute(y)

