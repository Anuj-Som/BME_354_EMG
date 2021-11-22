# Module to classify a signal as either SB (SPACE BAR) or Down
# Uses scikit learn Multilayer Perceptron machine learning model to classify using data
# https://scikit-learn.org/stable/modules/neural_networks_supervised.html

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier 
import pickle
from data_wrangle import train_X, train_y

print("Starting training")
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(100, 25), random_state=1)

trained = clf.fit(train_X, train_y)
print("Completed Training")
print(trained)

filename = "trained_model.sav"
print("Saving to {}".format(filename))
pickle.dump(clf, open(filename, 'wb'))
