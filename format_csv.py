# import os
import pandas as pd
import numpy as np
# from torchvision.io import read_image


def randomize_test_training(split, n):
    # Contingent on data being in training_data_csv
    # Numbered from 1 to n
    fifty_nums = [i for i in range(1, n+1)]
    training = np.random.choice(n, int(n*split), replace=False)+1
    testing = [item for item in fifty_nums if item not in training]
    return training, testing


def clean_csv(fileName, FourierMode, DBMode):
    # If filename not "SB" or "Down" throw error
    assert fileName in ["SB", "Down", "BN"]

    # Clear existing master CSVs
    mode = "fourier" if FourierMode else "voltage"
    db = "db" if DBMode else ""

    testpath = "./master_data/" + fileName + "_test_" + mode + db + ".csv"
    trainpath = "./master_data/" + fileName + "_train_" + mode + db + ".csv"
    open(testpath, 'w').close()
    open(trainpath, 'w').close()


def save_df_as_csv(fileName, test_csv, train_csv, FourierMode, DBMode):
    # If filename not "SB" or "Down" throw error
    assert fileName in ["SB", "Down", "BN"]

    mode = "fourier" if FourierMode else "voltage"
    db = "db" if DBMode else ""
    
    testpath = "./master_data/" + fileName + "_test_" + mode + db + ".csv"
    trainpath = "./master_data/" + fileName + "_train_" + mode + db + ".csv"
    test_csv.to_csv(testpath, index=False)
    train_csv.to_csv(trainpath, index=False)


def generate_master_csv(fileName, training, testing, fourierMode, DBMode):
    # If filename not "SB" or "Down" throw error
    assert fileName in ["SB", "Down", "BN"]

    # Clear existing master CSVs
    clean_csv(fileName, FourierMode, DBMode)

    # Wrangle all data into new CSV
    data_path = "./training_data_csv/" + fileName + "/"  + fileName

    # First with training dataframe
    train_df = pd.DataFrame()
    for count, number in enumerate(training):
        # Get raw data from csv file
        train_path = data_path + str(number) + ".csv"
        raw_df = pd.read_csv(train_path, names=["time", "voltage"])

        # If fourierMode is on, voltage data will be inputted to master CSV
        # as fft magnitude frequency information
        if(fourierMode):
            raw_df['voltage'] = np.abs(np.fft.fft(raw_df['voltage']))
        if(DBMode):
            raw_df['voltage'] = 20*np.log10(raw_df['voltage'])
        
        # Append data to master dataframe
        train_df[str(count)] = raw_df['voltage']
    train_df['label'] = pd.Series([fileName for i in range(len(train_df)+1)])
    
    # Repeat for testing dataframe
    test_df = pd.DataFrame()
    for count, number in enumerate(testing):
        # Get raw data from csv file
        test_path = data_path + str(number) + ".csv"
        raw_df = pd.read_csv(test_path, names=["time", "voltage"])

        # If fourierMode is on, voltage data will be inputted to master CSV
        # as fft magnitude frequency information
        if(fourierMode):
            raw_df['voltage'] = np.abs(np.fft.fft(raw_df['voltage']))
        if(DBMode):
            raw_df['voltage'] = 20*np.log10(raw_df['voltage'])
        
        # Append data to master dataframe
        test_df[str(count)] = raw_df['voltage']
    test_df['label'] = pd.Series([fileName for i in range(len(test_df)+1)])

    return train_df, test_df


def main(fileName, num, FourierMode, DBMode):
    # Designate percent split of training/testing, fileName as ["SB", "Down", "BN"]
    train_nos, test_nos = randomize_test_training(0.8, num)

    clean_csv(fileName, FourierMode, DBMode)
    master_train_df, master_test_df = generate_master_csv(fileName, train_nos, test_nos, FourierMode, DBMode)
    save_df_as_csv(fileName, master_test_df, master_train_df, FourierMode, DBMode)
    # merge_df(master_train_df, master_test_df)

if __name__ == "__main__":
    FourierMode = False
    DBMode = False
    fileNames = ["SB", "Down", "BN"]
    Nums = [250, 250, 75]
    for i in range(len(fileNames)):
        main(fileNames[i], Nums[i], FourierMode, DBMode)
    
