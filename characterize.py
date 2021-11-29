import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier 
import sklearn
import pickle
from data_wrangle import main

# Evaluate fourier model
def evaluate_model(testType):
    filename = "{}_trained_model.sav".format(testType)
    loaded_model = pickle.load(open(filename, 'rb'))

    train_X, train_y, test_X, test_y = main(testType)

    all_X = np.concatenate((test_X, train_X), axis=0)
    all_y = np.concatenate((test_y, train_y), axis=0)

    print("Model: {}".format(filename))
    print("Over all data: n={}".format(len(all_X)))
    result = loaded_model.score(all_X, all_y)
    print("Accuracy: {}".format(result))

    print("Over test data: n={}".format(len(test_X)))
    result = loaded_model.score(test_X, test_y)
    print("Accuracy: {}".format(result))

    c = sklearn.metrics.confusion_matrix(test_y, loaded_model.predict(test_X))
    print(c)

if __name__ == "__main__":
    evaluate_model("fourier")
    evaluate_model("fourierdb")
    evaluate_model("voltage")

    # filename = "trained_model.sav"
    # loaded_model = pickle.load(open(filename, 'rb'))

    # train_X, train_y, test_X, test_y = main("fourier")

    # all_X = np.concatenate((test_X, train_X), axis=0)
    # all_y = np.concatenate((test_y, train_y), axis=0)

    # print("Model: {}".format(filename))
    # print("Over all data: n={}".format(len(all_X)))
    # result = loaded_model.score(all_X, all_y)
    # print("Accuracy: {}".format(result))

    # print("Over test data: n={}".format(len(test_X)))
    # result = loaded_model.score(test_X, test_y)
    # print("Accuracy: {}".format(result))

    # c = sklearn.metrics.confusion_matrix(test_y, loaded_model.predict(test_X))
    # print(c)