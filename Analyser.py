
from matplotlib import pyplot as plt
import numpy as np
import FileReader as fr
import SignalFilter as f
from scipy.signal import find_peaks
from EMG_ploter import plot_power, time_vector_calc


def normalize(emg_rectified_BP):
    max = np.amax(emg_rectified_BP)
    print("the max value is " + str(max))
    # threshold = max/10
    # emg_rectified_BP_fft[emg_rectified_BP_fft < threshold] = 0
    normalized = np.true_divide(
        emg_rectified_BP, max)
    return normalized


def emg_rms(emg_filtered_rectified):
    rms = np.sqrt(np.mean(emg_filtered_rectified**2))
    return rms

# muscle code is the index of its' PIN in the file


def mean_all_sifnal_RMSs_oneTraining_one_athele(filtered_rectified_Signals=[]):

    RMSs_Vector = np.array([])
    # add all RMSs ro the Vector
    for emg_filtered_rectified in filtered_rectified_Signals:
        RMSs_Vector = np.append(RMSs_Vector, emg_rms(emg_filtered_rectified))

    return np.mean(RMSs_Vector)

# find the indcies of where the movement (period) ended

# find the indcies with very low bottoms and return the most low bottoms indcies
def cutting_indecies_finder(env_sig, num_of_parts):

    index_value_width_dic = find_bottoms(env_sig)

    best_botom_indcies = np.array([],int)
    #find the indcies of the lowst minimas
    print(index_value_width_dic)
    index_value_width_dic = sorted(index_value_width_dic.items(),
                             key=lambda x: x[1][0], reverse = False)
    # index_value_width_dic = sorted(sorted(index_value_width_dic[:4] , key=lambda x: x[0], reverse = True))
    print("sorted CI" ,index_value_width_dic)

    # lowest five bottoms indcies in signal
    i = 0
    for item in index_value_width_dic:
        if i < num_of_parts :  # select the best num_of_parts cutters
            best_botom_indcies = np.append(
                best_botom_indcies, int(item[0]))
        i += 1

    return best_botom_indcies

def find_bottoms(envelped_sig):

    array_pointer = 600
    bottom_index = 0
    zero_slope = 0.002
    index_value_width_dic = {}
    while array_pointer < envelped_sig.size:  # go over the
        try:
            current_slope = envelped_sig[array_pointer +1] - envelped_sig[array_pointer]
            Before_Slope = 0
            while current_slope < 0:  # if curunt slope < 0 check if the next slop is 0
                Before_Slope +=1
                array_pointer += 1  # go forward
                current_slope = envelped_sig[array_pointer+1] - envelped_sig[array_pointer]  # get the next slope
                bottom_width = 0

            # while 0 then wait -> to avoid a bug
            # the bug occure if it is not positive but very small negative negative also so it will be refused
            # also that will make sure its a null slope position
            while -0.00015 < current_slope <  0.00015:   # while slope = 0 wait
                current_slope = envelped_sig[array_pointer + 1] - envelped_sig[array_pointer]
                array_pointer += 1  # go forwaed
                bottom_width += 1                   
                # get the next slope
                current_slope = envelped_sig[array_pointer +1] - envelped_sig[array_pointer]
                slope_after = 0

            while current_slope > 0 and bottom_width > 0 :  # if slope after 0 becomes positive
                # 2. add it to  our posible cutter bottom_width
                slope_after += 1
                array_pointer += 1
                current_slope = envelped_sig[array_pointer +1] - envelped_sig[array_pointer]
                if current_slope < 0:
                    print(Before_Slope,slope_after)
                     # 1. regist index as a posible cutter index
                    bottom_index = array_pointer - int( bottom_width / 2 ) - slope_after  # to find the middel point7
                    av = np.average([ Before_Slope, slope_after*3,-1* envelped_sig[bottom_index] * 100 ] )
                    index_value_width_dic[bottom_index] = (envelped_sig[bottom_index], bottom_width ,av)
                    Before_Slope = 0
                    slope_after = 0
                    bottom_width = 0 
                    print('new bottom index :', index_value_width_dic)

        except:
            print("array out")
            

        array_pointer += 1

    return index_value_width_dic

# find the biggest two peaks in every repeatation and return thier indcies

# cut - find peaks - compare them - return the biggest 2 peaks


def peaks_finder(rectified_sig=[], cutting_indcies=[], number_of_parts = 3):

    splitted_signals = signal_splitter(
        rectified_sig, cutting_indcies, number_of_parts)

    all_peak_indcies = np.array([], dtype=int)

    # find the peak in evry part
    for part_signal in splitted_signals:

        # get the peak of a part signal
        part_peak_indcies, _ = find_peaks(part_signal, distance = part_signal.size) # only one peak

        dict = {}
        for peak_index in part_peak_indcies:
            dict[peak_index] = part_signal[peak_index]

        dict = sorted(dict.items(), key=lambda x: x[1], reverse=True)
        # peak indcies of one signal part

        best_peak_index = dict[0][0] #extract the index of the best peak

        all_peak_indcies = np.append(all_peak_indcies,  best_peak_index)

        # print(part_peak_indcies)
        # for index in indcies:
        #     plt.axvline(index,0,1, label='pyplot vertical line')
        # print(len(part_peak_indcies) == 2)
        # plt.plot(sig)
        # plt.plot(part_signal)
        # plt.plot(part_peak_indcies, part_signal[part_peak_indcies], "x")
        # plt.show()

    return all_peak_indcies

# returns parts of signal
# the parts elements have the same indexing as at the time they was in the original signal(array)
# thats did by adding zeros in the front the parts to make the samples have the same indecies

def signal_splitter(envelped_sig, cutting_indcies):


    # split the signal to number_of_parts signals
    # if the cutting_indcies are not sorted this may cause a problem
    splitted_signals = np.split(envelped_sig, sorted(cutting_indcies))
    # create new array to recive the "added zeros" splitted arrays
    new_splitted_signals = np.empty_like(splitted_signals, dtype=object)

    zeros_size = 0
    # bulid an array of the splitted signals
    for i in range(0, len(cutting_indcies) + 1 ):
        zeros = np.zeros((1, zeros_size))
        new_splitted_signals[i] = np.append(zeros, splitted_signals[i])
        # update the sizes of zeros to be added
        zeros_size += splitted_signals[i].size

    return new_splitted_signals

# retund the indecies of the start and end of the tails


def tails_finder(enveloped_signal):

    start_end_dic = {}
    index_w_av_dic = find_bottoms(enveloped_signal)

    sorted_index_av_dic = sorted(index_w_av_dic.items(),
                                    key=lambda x: x[0], reverse=True)
    # indcies of lowest three bottoms  in signal
    selected_index_width_dic = sorted_index_av_dic[0]

    start = selected_index_width_dic[0] - int(selected_index_width_dic[1][1] / 2)
    end = start + selected_index_width_dic[1][1]
    start_end_dic[start] = end

    return start_end_dic

# return a signal without tails


def tails_cutter(enve):

    # get the start and end of the tail in form {start1 : end1 , start2 : end2 }
    start_end_dict = tails_finder(enve)
    # convert it to a list of tuples
    start_end_list_of_tuples = start_end_dict.items()

    for start_end_Tuple in start_end_list_of_tuples:
        start_end = np.array(start_end_Tuple)
        S = start_end[0]
        E = start_end[1]
        noise_range = np.arange(S, E ,1)
        np.put(enve, noise_range, 0)
    # plt.figure()
    # plt.plot(enve)
    return enve

def main() :
    time_to_skip_in_sec = 1.8  # sec
    SampleRate_to_skip = time_to_skip_in_sec * 250
    # # emg_PIN_Signal is a lst of a row muscle signal (depends on the muscle code) of a selected athele
    emg_PIN_Signal = fr.P4_P6_P8_Reader('EMG Data/Max/OpenBCI-RAW-Max_Back_Squat.txt',
                                        samples_to_skip=SampleRate_to_skip)[2]

    sig = f.BP_fftf_rectified(emg_PIN_Signal, 20, 100, 250)
    # t_v = time_vector_calc(env, 250)

    low_pass_freq = 0.1

    env = f.envelop(sig, 250, 4, 1)

    # while low_pass_freq < 1:
    #env = f.envelop(sig, 250, 4, low_pass_freq)
    low_pass_freq += 0.1
    # cut the signal tail to avoid the false bottoms at the tail
    # env = env[: sig.size - 10]
    CI = sorted(cutting_indecies_finder(env,2))
    print("CI" , CI)
    Repeatations = np.split(env,CI)

    Movements = np.array([])

    zeros_size = 0
    peaks = np.array([],int)
    for R in Repeatations:
        splitted_Repeatation = np.split(R ,cutting_indecies_finder(R,1))
        zeros = np.zeros((1, zeros_size))
        plt_M = np.append(zeros,R)
        plt.plot(plt_M)
        zeros_size += R.size
        peak ,_ = find_peaks(plt_M, distance=plt_M.size )  # only one peak
        plt.plot(peak, env[peak] , "x")
        peaks = np.append(peaks,peak)
            
    #print(low_pass_freq)
    plt.show()

    # if peaks.size == 6:
    #     for p in peaks:
    #         plt.plot(p, env[p] , "x")
    #     break

    
    # plot x on the bottoms
    # plt.plot(best_bottoms / 250, env[best_bottoms], "x")
    # print('cutters', env[best_bottoms])

    # all_peaks = np.array(peaks_finder(sig, cutting_indcies=best_bottoms))


    # plt.plot( sig)

    # # get the start and end of the tail in form {start1 : end1 , start2 : end2 }
    # start_end_dict = tails_finder(env)
    # # convert it to a list of tuples
    # start_end_list_of_tuples = start_end_dict.items()

    # for start_end_Tuple in start_end_list_of_tuples:
    #      start_end = np.array(start_end_Tuple)
    #      for start_or_end in start_end:
    #          plt.axvline(start_or_end/250, 0, 0.5, label='pyplot vertical line')

    # #plot_power(f.butter_bandpass_filter(emg_PIN_Signal,20,100,250), 1/250)

    plt.plot(env)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude[mv]')
    plt.show()

    # ########################### noise ploter ####################


    # s_e_l = list(start_end_list_of_tuples)[0]
    # noise_part = sig[s_e_l[0]: s_e_l[1]]

    # noise_part_fft = fftpack.fft(noise_part)
    # Amplitudes = np.abs(noise_part_fft)
    # print('amp', Amplitudes)
    # noise_freq = np.array(fftpack.fftfreq(noise_part.size, d=1/250))
    # pos_mask = np.where(noise_freq > 0)
    # pos_noise_freqs = noise_freq[pos_mask]  # find the positive sample frequancies
    # pos_noise_freqs[Amplitudes[pos_mask].argmax()]
    # print('pos noise freq',pos_noise_freqs)
    # print('max amp', np.where(
    #     Amplitudes[pos_mask] == np.amax(Amplitudes[pos_mask])))

    # plt.figure(2)
    # t_v_noise =time_vector_calc(noise_part,250)
    # plt.plot(t_v_noise,noise_part)
    # plt.figure(3)
    # plt.plot(pos_noise_freqs)

    # print(noise_part)
    # print(pos_noise_freqs)

    # tails_cutter(sig)
    plt.show()
