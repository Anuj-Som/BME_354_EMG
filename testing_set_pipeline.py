import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier 
import pickle
from scipy.signal import butter, filtfilt
import sklearn


def process_lvm_to_csv(file_path, file_type, file_name):
    file_dir = "{}/{}.{}".format(file_path, file_name, file_type)
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
    cutoff = 25     # desired cutoff frequency of the filter, Hz, slightly higher than actual 1.2 Hz
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
    no_to_show = 250    # Toggle highest frequency shown

    T = 1/50000
    yf = np.fft.fft(voltage_data)
    N = len(voltage_data)
    xf = np.fft.fftfreq(N, T)[:N//2]

    plt.plot(xf[0:no_to_show], (2.0/N * np.abs(yf[0:N//2])[0:no_to_show]))
    plt.grid()

    plt.title('Raw FFT signal data')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
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
    # returns fft data
    process_lvm_to_csv(file_path, file_type, file_name)
    data_df = get_df(file_path, file_name)

    # Get fft magnitude data to pass through model
    fft_data = np.abs(np.fft.fft(data_df['voltage']))
    # fft_data = 20*np.log10(np.abs(np.fft.fft(data_df['voltage'])))
    # fft_data = data_df['voltage']
    # print(np.shape(fft_data))

    return fft_data


if __name__ == "__main__":
    # Add new .lvm file in path ./incoming_data/lvm
    # print("Make sure saved lvm file is stored within incoming_data/lvm/")
    # file_paths = ["HighDown", "HighSpaceBar", "LowDown", "LowSpaceBar", "MovingDown", "MovingSpaceBar"]
    # file_paths_names = ["HDown", "HSB", "LDown", "LSB", "MDown", "MSB"]

    # Testing Sara's stuff
    file_paths = ["Sara/DeepDown", "Sara/DeepSB", "Sara/Down", "Sara/SB"]
    file_paths_names = ["DDown", "DSB", "Down", "SB"]
    
    # n = [15, 15, 15, 15, 15, 11]
    # ys =[[1 for i in range(n[0])],
    #      [0 for i in range(n[1])],
    #      [1 for i in range(n[2])],
    #      [0 for i in range(n[3])],
    #      [1 for i in range(n[4])],
    #      [0 for i in range(n[5])]]

    # Testing Sara's stuff
    n = [15, 15, 30, 30]
    ys =[[1 for i in range(n[0])],
         [0 for i in range(n[1])],
         [1 for i in range(n[2])],
         [0 for i in range(n[3])]]
    # print(np.shape(ys))
    
    filename = "fourier_trained_model.sav"
    loaded_model = pickle.load(open(filename, 'rb'))

    file_type = "lvm"
    for i in range(len(file_paths)):
        fft_data = []
        for ii in range(1, n[i]+1):
            file_path = "./TestingSets/" + file_paths[i]
            filename = file_paths_names[i] + str(ii)
            fft_data.append(main(file_path, file_type, filename))
        # print(np.shape(fft_data))
        print("\n")

        print("Over {} test data: n={}".format(file_paths[i], len(fft_data)))
        result = loaded_model.score(fft_data, ys[i])
        print("Accuracy: {}".format(result))
        c = sklearn.metrics.confusion_matrix(ys[i], loaded_model.predict(fft_data))
        print(c)
            




    
