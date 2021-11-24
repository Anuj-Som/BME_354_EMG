import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier 
import pickle

# Data processing & Wrangling to np arrays
SB_train_data = "./master_data/SB_train.csv"
SB_test_data = "./master_data/SB_test.csv"
Down_train_data = "./master_data/Down_train.csv"
Down_test_data = "./master_data/Down_test.csv"
BN_train_data = "./master_data/BN_train.csv"
BN_test_data = "./master_data/BN_test.csv"

def process_data(dataframe, label):
    # Let Label "0" be "SB", "1" be "Down"
    X = dataframe.drop(columns=["label"]).T.to_numpy()
    labelDict = {"SB": 0.0, "Down": 1.0, "BN": 2.0}
    assert label in labelDict.keys()
    y = [labelDict[label] for i in range(len(X))]
    return X, y

SB_train_df = pd.read_csv(SB_train_data)
SB_train_X, SB_train_y = process_data(SB_train_df, "SB")

SB_test_df = pd.read_csv(SB_test_data)
SB_test_X, SB_test_y = process_data(SB_test_df, "SB")

Down_train_df = pd.read_csv(Down_train_data)
Down_train_X, Down_train_y = process_data(Down_train_df, "Down")

Down_test_df = pd.read_csv(Down_test_data)
Down_test_X, Down_test_y = process_data(Down_test_df, "Down")

BN_train_df = pd.read_csv(BN_train_data)
BN_train_X, BN_train_y = process_data(BN_train_df, "BN")

BN_test_df = pd.read_csv(BN_test_data)
BN_test_X, BN_test_y = process_data(BN_test_df, "BN")

train_X = np.concatenate((SB_train_X, Down_train_X, BN_train_X), axis=0)
test_X = np.concatenate((SB_test_X, Down_test_X, BN_test_X), axis=0)
# print(X)
# print(np.shape(X))

train_y = np.concatenate((SB_train_y, Down_train_y, BN_train_y), axis=0)
test_y = np.concatenate((SB_test_y, Down_test_y, BN_test_y), axis=0)
# print(y)
# print(np.shape(y))