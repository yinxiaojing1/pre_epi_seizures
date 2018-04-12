# MLP for Pima Indians Dataset with 10-fold cross validation via sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy
 
# Function to create model, required for KerasClassifier
def dense_network(output_dim,
                  input_dim,
                  hidden_layers_nr,
                  hidden_nodes_per_layer,
                  hidden_nodes_activation):
    # create model
    model = Sequential()
    
    # Create Hidden Layer Topology
    for layer in xrange(0, hidden_layers_nr):
        model.add(Dense(hidden_nodes_per_layer,
                        input_dim=input_dim,
                        activation=hidden_nodes_activation))
 
    # Add output layer
    model.add(Dense(output_dim, activation='softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model