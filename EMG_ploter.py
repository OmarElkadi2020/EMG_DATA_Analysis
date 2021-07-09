import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack

#############( X axis , y axis , title , lable)#########
def emg_plot(time_vec_in_sec , emg_in_mV , title ,label):
    plt.figure(title, figsize=(6, 5))
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude[mv]')
    plt.plot(time_vec_in_sec, emg_in_mV, linewidth=1, label = label)
    plt.legend()

def time_vector_calc(sig, sample_rate) :
    array_length = sig.size
    # create a time vector with a time step = 1/250 seconds
    time_vec = np.array(np.true_divide(np.arange(0, array_length, 1), sample_rate))
    return time_vec


def plot_power(sig, time_step):
    # The FFT of the signal
    sig_fft = fftpack.fft(sig)

    # And the power (sig_fft is of complex dtype)
    power = np.abs(sig_fft)**2

    # The corresponding frequencies
    sample_freq = fftpack.fftfreq(sig.size, d=time_step)

    # Plot the FFT power
    plt.figure(figsize=(6, 5))
    plt.plot(sample_freq, power)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('power')

    # Find the peak frequency: we can focus on only the positive frequencies
    pos_mask = np.where(sample_freq > 0)
    freqs = sample_freq[pos_mask]
    peak_freq = freqs[power[pos_mask].argmax()]


    # An inner plot to show the peak frequency
    axes = plt.axes([0.55, 0.3, 0.3, 0.5])
    plt.title('Peak frequency')
    plt.plot(freqs[:8], power[:8])
    plt.setp(axes, yticks=[])
# scipy.signal.find_peaks_cwt can also be used for more advanced
# peak detection
