import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier 
import pickle
from data_wrangle import test_X, test_y

filename = "trained_model.sav"
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(test_X, test_y)
print(result)