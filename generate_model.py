# Module to classify a signal as either SB (SPACE BAR) or Down
# Uses scikit learn Multilayer Perceptron machine learning model to classify using data
# https://scikit-learn.org/stable/modules/neural_networks_supervised.html

from sklearn.neural_network import MLPClassifier 
import pickle
from data_wrangle import main

testType = "voltage"            # either "fourier", "fourierdb", or "voltage"
print("Starting training")
clf = MLPClassifier(solver='adam', alpha=1e-5,     # Define hyperparameters
                    hidden_layer_sizes=(150, 50))
# solver='lbfgs'

train_X, train_y, test_X, test_y = main(testType)
trained = clf.fit(train_X, train_y)
print("Completed Training")
print(trained)

filename = "{}_trained_model.sav".format(testType)
print("Saving to {}".format(filename))
pickle.dump(clf, open(filename, 'wb'))
