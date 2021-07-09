from re import A
from pylab import *
from numpy import *
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.signal import butter, filtfilt

import numpy as np

    
def apply_fft(sig, time_step):
    # The FFT of the signal
    # contains all  frequancy components of the signalsin or cos amplitudes (real part) associated with the
    f_hat = fftpack.fft(sig)  # (f_hat [0] = A0 e^jW0 => A0 (sin( freq0) + j cos (freq0)
                            
    # And the power (f_hat is of complex dtype) = squaring of every element in the array
    power = np.abs(f_hat)**2 # list of the all component powers = the amplitude^2 of the frequancy components

    # The corresponding frequencies
    sample_freq = fftpack.fftfreq(sig.size, d=time_step) # create a fourier sampling signal depends on the given signal size and its sampling frequancy 
    high_freq_fft = f_hat.copy()

    # Find the peak frequency: we can focus on only the positive frequencies
    pos_mask = np.where(sample_freq > 0)
    freqs = sample_freq[pos_mask] #find the positive sample frequancies
    max_power_indcies = power[pos_mask].argmax() # the indcies here is the sample frequancies
    peak_freq = freqs[max_power_indcies]
     # the frequancy with the most powers
    #### if the elememt frequancy more than the frequancy of the peak power set it to 0
    high_freq_fft[np.abs(sample_freq) < peak_freq] = 0 
    filtered_sig = fftpack.ifft(high_freq_fft)
    return filtered_sig


def apply_fft(sig, time_step):
    # The FFT of the signal
    # contains all  frequancy components of the signalsin or cos amplitudes (real part) associated with the
    # (f_hat [0] = A0 e^jW0 => A0 (sin( freq0) + j cos (freq0)
    f_hat = fftpack.fft(sig)

    # And the power (f_hat is of complex dtype) = squaring of every element in the array
    # list of the all component powers = the amplitude^2 of the frequancy components
    power = np.abs(f_hat)**2

    # The corresponding frequencies
    # create a fourier sampling signal depends on the given signal size and its sampling frequancy
    sample_freq = fftpack.fftfreq(sig.size, d=time_step)
    high_freq_fft = f_hat.copy()

    # Find the peak frequency: we can focus on only the positive frequencies
    pos_mask = np.where(sample_freq > 0)
    freqs = sample_freq[pos_mask]  # find the positive sample frequancies
    # the indcies here is the sample frequancies
    max_power_indcies = power[pos_mask].argmax()
    peak_freq = freqs[max_power_indcies]
    # the frequancy with the most powers
    #### if the elememt frequancy more than the frequancy of the peak power set it to 0
    high_freq_fft[np.abs(sample_freq) < peak_freq] = 0
    filtered_sig = fftpack.ifft(high_freq_fft)
    return filtered_sig

def butter_bandpass_filter(emg, lowcut, highcut, SampleRate, order=4):
    nyq = 0.5 * SampleRate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    BPsignal = filtfilt(b, a, emg)
    return BPsignal


def rectify(emg_filtered_fft_BP):
    return abs(emg_filtered_fft_BP)


def BP_fftf_rectified(emg, lowcut, highcut, SampleRate, order=4):
    return rectify(apply_fft(butter_bandpass_filter(
        emg, lowcut, highcut, SampleRate, order=4), 1/SampleRate))


def fftf_BP_rectified(emg, lowcut, highcut, SampleRate, order=4):
    emg_fft = apply_fft(emg, 1/SampleRate)
    return rectify(butter_bandpass_filter(
        emg_fft, lowcut, highcut, SampleRate, order=4))
0


def BP_rectified(emg, lowcut, highcut, SampleRate, order=4):
    return rectify(butter_bandpass_filter(emg, lowcut, highcut, SampleRate, order=4))


def envelop(emg_rectified, SampleRate, order=4, low_pass=  2  ):
    low_pass = low_pass / (SampleRate/2)
    b2, a2 = butter(order, low_pass, btype='lowpass')
    emg_envelope = filtfilt(b2, a2, emg_rectified)
    return emg_envelope


def smooth(sig, window_len=11, window='hanning'):

    s = np.r_[sig[window_len-1:0:-1], sig, sig[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    smoothed = np.convolve(w/w.sum(), s, mode='valid')
    return smoothed

###############################################

############################## apply ButterworthFilter #########################
# emg_BP = butter_bandpass_filter(emgs_PIN_INPUT, lowerCutOffFrequancy, HigherCutOffFrequancy, SampleRate, ButterworthFilterOrder)
# #apply fft filter
# emg_fft = apply_fft(emgs_PIN_INPUT, time_step)

################################ double filtered #############################
# emg_BP_fft = apply_fft(emg_BP, time_step)
# emg_fft_BP = butter_bandpass_filter(
# emg_fft, lowerCutOffFrequancy, HigherCutOffFrequancy, SampleRate, ButterworthFilterOrder)

################################## rectify ##################################
# emg_rectified_BP = rectify(emg_BP)
# emg_rectified_fft = rectify(emg_fft)
# emg_rectified_BP_fft = rectify(emg_BP_fft)
# emg_rectified_fft_BP = rectify(emg_fft_BP)



################################ ratio to the max #############################
 # emg_max_normalized = np.amax(emg_normalized)
# emg_rms_normalized = ana.emg_rms(emg_normalized) * 100
