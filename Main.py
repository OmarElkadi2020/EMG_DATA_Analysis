from matplotlib.figure import Figure
from numpy.core.fromnumeric import mean
from numpy.core.numeric import indices
from Analyser import cutting_indecies_finder, signal_splitter, tails_cutter
import SignalFilter as f
import numpy as np
import FileReader as fr
import FileWriter as fw
import EMG_ploter as emg_ploter
from matplotlib.pyplot import axis, close, legend, plot, savefig, show, title ,figure
from scipy.signal import find_peaks


def emg_rmss(emg_filtered_rectified):
    rms = np.sqrt(np.mean(emg_filtered_rectified**2))
    return rms
########################### initialization #######################

effectOfEveryTraining_OnOneMuscle_table = np.empty((0,3))
full_report = np.empty_like(effectOfEveryTraining_OnOneMuscle_table)

lowerCutOffFrequancy = 20  # hz
HigherCutOffFrequancy = 50  # hz
ButterworthFilterOrder = 4
SampleRate = 250  # Hz
time_step = 1/SampleRate  # 1/250 step(samples per seconds)sec

musles_names = ['vastus medialis', 'Biceps femoris', 'Gluteaus Maximus']
athletes_names = ['Rico', 'Patrick', 'Oli', 'Simon' , 'Jannik' , 'Sebastian','Max', 'Moe' ,'Felix', 'Fabio']
path_to_root_folder = './EMG Data/'

def report_generator():  # one muscle
   
    ########################### initialization #######################

    effectOfEveryTraining_OnOneMuscle_table = np.empty((0,3))
    full_report = np.empty_like(effectOfEveryTraining_OnOneMuscle_table)
    ################### get the inputs #########################
    try:
        muscle_code = int(
            input(' Muscle code  => 0 Vastus Medialis default | 1 Biceps femoris | 2 Gluteaus Maximus : '))
        muscle_name = musles_names[muscle_code]
    except:
        muscle_code = 0  # defaultRic
        muscle_name = musles_names[muscle_code]

    # athlete_name = input(
    # 'Probande Namen geben => Rico (default) | Patrick | Oli |Simon | Sebastian | Jannik : ')
    # if athlete_name in athletes_names:
    #     print(athlete_name + 'selected')
    # else:
    #     athlete_name = 'Rico'
    try:
        Write_the_output = bool(
            input('do you want want to write the output?  => True | False is default) '))
    except:
        Write_the_output = False  # default

    #########################  all athele all trainings extract the effect on one muscle #################
    for athlete_name in athletes_names:

        folder_path = path_to_root_folder + athlete_name
        # get all the trainings files => to extract a muscle from every one
        all_folder_files = fr.get_all_files_in_given_folder(folder_path)

        # add the new table header
        effectOfEveryTraining_OnOneMuscle_table = np.append(
            effectOfEveryTraining_OnOneMuscle_table, [['Athele_Training', 'max(mv)\t', 'rms(mv)']], axis=0)

        #########################  the same athele all trainings extract the effect on one muscle #################
        for file_path in all_folder_files:

            ########## extract a muscle signal from a Trainig file of an athlete #############
            time_to_skip_in_sec = 2  # sec
            SampleRate_to_skip = time_to_skip_in_sec * SampleRate
            # emg_PIN_Signal is a lst of a row muscle signal 
                ## (depends on the muscle code) of a selected athele
            emg_PIN_Signal = fr.P4_P6_P8_Reader(
                file_path, samples_to_skip=SampleRate_to_skip)[muscle_code]

            # another method to get the signal Of pin4 form the extracted file
            # Patrick_Back_Squat_vastusMedialis = get_file_as_matrix(
            #     'EMG Data/Patrick/P4_P6_P8_Extracted/OpenBCI-RAW-Patrick_Back_Squat.txt.mini.txt')[: , 0]

            ####################### apply the filters + rectify #########################
            emg_rectified_BP = f.butter_bandpass_filter(
                emg_PIN_Signal, lowerCutOffFrequancy,
                HigherCutOffFrequancy, SampleRate, 4)
            emg_enveloped = f.envelop(
                emg_rectified_BP, SampleRate, low_pass = 1.3)

            ################# get the max and the rms of the emg signal #######################
            # splitted = np.array_split(emg_enveloped ,3)
            emg_maxs = np.empty((1,0))
            # emg_maxs_indcies = np.empty((1,0), int)
            # zeros_size = 0
            # for s in splitted:
            #     zeros = np.zeros((1, zeros_size))
            #     emg_maxs = np.append(emg_maxs, np.amax(s))
            #     emg_maxs_indcies = np.append(emg_maxs_indcies, np.argmax(np.append(zeros,s)))
            #     zeros_size += s.size
            # emg_enveloped = tails_cutter(emg_enveloped)
            
            emg_max =  0 
            peak_indcies, _ = find_peaks(emg_enveloped, distance = emg_enveloped.size/5 ,
                            height= emg_enveloped.max()/4) # only one peak
            for peak_index in peak_indcies:
                emg_maxs = np.append(emg_maxs, emg_enveloped[peak_index])

            emg_max = mean(emg_maxs)
            emg_rms = emg_rmss(emg_enveloped)

            ############################# build a ['training_name', 'max(mv)\t', 'rms(mv)\t'] row #################
            # get training_name from the relativ file path which equals the given "relativ folder_path/file name"
            # file_path example  ./EMG Data/Rico/athele-training_Name.extention"
            training_name = file_path.split('/')[3].split('-')[2].split('.')[0]

            # add new traning rms in the muscle table
            effectOfEveryTraining_OnOneMuscle_table = np.append(
                effectOfEveryTraining_OnOneMuscle_table, [[training_name, emg_max, emg_rms]], axis=0)

            ############################# plot  all signsls from a mucle varius trainings #####################################
            # calc the x axix from the recorded signal depends on the given sample rate
            time_vec = emg_ploter.time_vector_calc(emg_enveloped, SampleRate)
                                #( X axis , y axis , title , lable)
           
            title = athlete_name + '_' + muscle_name
            figure(title)
            for peak_index in peak_indcies:
                plot(peak_index/250, emg_enveloped[peak_index] ,'x')
            plot(time_vec, emg_enveloped, label= training_name)
            legend()
            savefig(folder_path +'/img/_'+ muscle_name +'_'+ training_name)
            close()  

        
    full_report = np.append( full_report, effectOfEveryTraining_OnOneMuscle_table, axis=0)
    ####################### write the muscle table of rms for every training in a file ###############
    if(Write_the_output):
        path_to_write = path_to_root_folder + muscle_name + '_Full Report.csv'
        fw.table_Writer(path_to_write, full_report)
        print('writen is : ' + path_to_write)

def drawer():  
    athlete_name = input('Probande Namen geben => Rico (default) | Patrick | Oli |Simon | Sebastian | Jannik : ')
    if athlete_name in athletes_names:
        print(athlete_name + 'selected')
    else:
        athlete_name = 'Rico'
    folder_path = path_to_root_folder + athlete_name
    try:
        muscle_code = int(
            input(' Muscle code  => 0 Vastus Medialis default | 1 Biceps femoris | 2 Gluteaus Maximus : '))
    except:
        muscle_code = 0  # default
        muscle_name = musles_names[muscle_code]
    # get all the trainings files => to extract a muscle from every one
    all_folder_files = fr.get_all_files_in_given_folder(folder_path)


    muscle_name = musles_names[muscle_code]
    for file_path in all_folder_files:  
        time_to_skip_in_sec = 2  # sec
        SampleRate_to_skip = time_to_skip_in_sec * SampleRate
        # emg_PIN_Signal is a lst of a row muscle signal 
            ## (depends on the muscle code) of a selected athele
        emg_PIN_Signal = fr.P4_P6_P8_Reader(
            file_path, samples_to_skip=SampleRate_to_skip)[muscle_code]
        


report_generator()

# fw.filterd_file_writer(Rico_vastusMedialis + file_path.split('/')[2], file_path)

# emg_plot(time_vec, emg_filtered_BP, "BP")
# emg_plot(time_vec, emg_rectified_BP_FFT, "BP the Rectified")
# emg_plot(time_vec, emg_enveloped, " BP then fft then rectified")


#f.plot_power(emg_filtered_BP, time_step)

#emg_filtered_BP_fft = f.apply_fft(emg_filtered_BP, time_step)
