# Convert all training_data.lvm to training_data.csv in the correct folders

import os
import pandas as pd
import numpy as np

def process_lvm_to_csv(file_dir, save_dir):
    f = open(file_dir)
    for i in range(24):
        f.readline()
    
    s = open(save_dir, "w")
    for a in f:
        b = a.replace("\t", ",")
        s.writelines(b)
    
    f.close()
    s.close()

def processFileName(filename, no):
    fromFilePath = "./training_data_lvm/" + filename + "/" + filename
    saveFilePath = "./training_data_csv/" + filename + "/" + filename

    for i in range(1, no+1):
        process_lvm_to_csv(fromFilePath + str(i) + ".lvm", saveFilePath + str(i) + ".csv")


if __name__ == "__main__":
    processFileName("BN", 75)
