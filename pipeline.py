import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier 
import pickle
from scipy.signal import butter, filtfilt


def process_lvm_to_csv(file_path, file_type, file_name):
    file_dir = "{}/{}/{}.{}".format(file_path, file_type, file_name, file_type)
    save_dir = "{}/csv/{}.csv".format(file_path, file_name)

    f = open(file_dir)
    for i in range(24):
        f.readline()
    
    s = open(save_dir, "w")
    for a in f:
        b = a.replace("\t", ",")
        s.writelines(b)
    
    f.close()
    s.close()

def get_df(file_path, file_name):
    save_dir = "{}/csv/{}.csv".format(file_path, file_name)
    df = pd.read_csv(save_dir, names=["time", "voltage"])
    return df


def butter_lowpass_filter(data):
    # Butterworth Filter requirements.
    T = 1.0         # Sample Period
    fs = 50000       # sample rate, Hz
    cutoff = 25     # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 2       # sin wave can be approx represented as quadratic
    n = int(T * fs)-1 # total number of samples

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

    # Filter the data, and plot both the original and filtered signals.
    # Call this to get filtered data
    y = butter_lowpass_filter(data)


def graph_fft(df):
    voltage_data = df['voltage']

    T = 1/50000
    yf = np.fft.fft(voltage_data)
    N = len(voltage_data)
    xf = np.fft.fftfreq(N, T)[:N//2]

    plt.plot(xf[0:500], (2.0/N * np.abs(yf[0:N//2])[0:500]))
    plt.grid()
    plt.show()


def graph_voltage(df):
    data = df['voltage'].to_numpy()
    data = data - data.mean() # Remove DC offset
    data_filtered = butter_lowpass_filter(df['voltage'] - df['voltage'].mean()) # 25 Hz fc butterworth

    x = df['time'].to_numpy()

    fig, ax = plt.subplots(1, 1, sharey=True)

    ax.plot(x, data, x, data_filtered)

    ax.set(xlabel='Time (s)', ylabel='Voltage (V)',
            title='Raw Signal data')
    ax.legend(['unfiltered signal', 'filtered signal'])
    ax.grid()

    # fig.savefig("./incoming_data/images/test_file.png")
    plt.show()

def main(file_path, file_type, file_name):
    filename = "trained_model.sav"
    loaded_model = pickle.load(open(filename, 'rb'))

    process_lvm_to_csv(file_path, file_type, file_name)
    data_df = get_df(file_path, file_name)

    # Get fft magnitude data to pass through model
    fft_data = np.abs(np.fft.fft(data_df['voltage']))
    print(np.shape(fft_data))

    characterize_dict = {"0": "SB", "1": "Down"}
    prediction = loaded_model.predict([fft_data])
    # print(prediction)
    eval = characterize_dict[str(int(prediction))]
    print("The data is {}".format(eval))

    graph_voltage(data_df)
    graph_fft(data_df)


if __name__ == "__main__":
    # Add new .lvm file in path ./incoming_data/lvm
    file_path = "./incoming_data"
    file_type = "lvm"
    filename = "SB_4"
    main(file_path, file_type, filename)