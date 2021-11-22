# BME_354_EMG
Anuj Som &amp; Sara Min BME 354 Final Project files &amp;

## Program Instructions:

This project accepts labview time vs voltage throat EMG files and runs their frequency magnitude content
through a Multilayer Perceptron classifier network. The signals will be classified as either "SB" meaning "Space Bar" or "Down".

## Signal Acquisition

To use the project as intended, it is recommended that the signal is obtained via the Labview SignalAcq applet, found on the desktop of many Duke BME computers. The biopotential amplifier circuit should be attached with two electrodes, one slightly under the chin and the other on the upper throat under the first electrode. The ground lead should be connected to the bony section behind the right ear. 

The signal is acquired with a sampling rate of 50,000 samples/second for 1.5 seconds. Click "Oneshot" and speak; the data will be recorded within a 1.5 second interval. This yields a total of 75,000 data points, which can be saved as a .lvm file.

Download this git repository and save the .lvm file as a desired 'filename.lvm' in the ./incoming_data/lvm folder. Then proceed to implementation section.

## Implementation

To implement this project, you can simply download the repository and run in bash: 

```python3 pipeline.py```

This will direct you to input the filename of the saved .lvm file. Input the filename without ".lvm" and press enter.

The filename.lvm file will automatically be processed, converted into a FFT magnitude array, and passed through the MLP model. The terminal will then characterize the signal and print out either "SB" or "Down". Following this, a plot will appear with the raw and filtered signal (25 Hz Butterworth lowpass). When this plot is closed, a new plot with the frequency spectra will appear. 

## Processing Methodology

This project utilizes a sklearn.neural_network MLPClassifier Multilayer Perceptron network saved within the trained_model.sav pickle file. The network accepts an array of 74,999 numpy float64 (intended to be EMG signal fourier magnitude data), passes it through hidden layer 1 with 100 hidden perceptrons, then hidden layer 2 with 25 hidden perceptrons, then finally characterizes into 2 output perceptrons, either 0 ("SB") or 1 ("Down"). 

To generate a new model based on updated data, refer to generate_model.py. This defaults to using the first 50 saved csv files in master_data csvs for each characterization, so tweak the code in format_csv.py which generates the master_csv files as needed. To see how to read the model from the pickle file, see lines 11-12 in characterize.py. Running characterize.py will run the model on the entire dataset and the specified test dataset and return the percentage of the inputs it labeled correctly. 
