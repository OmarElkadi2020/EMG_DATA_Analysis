from datetime import datetime, time, timedelta
import os
import numpy as np

# used to read only the pins 4,6,8


def P4_P6_P8_Reader(path="EMG Data/Patrick/OpenBCI-RAW-Patrick_Back_Squat_Fersenerhöhung.txt",  samples_to_skip=0):

    vastusMedialis = np.array([])
    BicepsFemoris = np.array([])
    GluteausMaximus = np.array([])

    vastusMedialis_row_element = 0.0
    BicepsFemoris_row_element = 0.0
    GluteausMaximus_row_element = 0.0

    with open(path, "r") as filestream:

        for line in filestream:  # get a line of the  in the file

            if (samples_to_skip > 0):
                samples_to_skip = samples_to_skip - 1
                continue

            # split the line to a list of elements
            rowElements = line.strip().split(",")

            # print(LineElements[0])
            #EMG_time = datetime.strptime(LineElements[12], "%H:%M:%S.%f")
            #EMG_time = np.append(EMG_Pin4, EMG_time, axis=0)

            # check if it convertable to float
            # firstChar = rowElements[0][0].strip()
            # if(firstChar in ['%', 'S']):
            #     continue

            # check if the the elements convertable to float
            try:
                vastusMedialis_row_element = float(
                    rowElements[4]) / 1000  # to mV
                BicepsFemoris_row_element = float(
                    rowElements[6]) / 1000  # to mV
                GluteausMaximus_row_element = float(
                    rowElements[8]) / 1000  # to mV

            except:
                vastusMedialis_row_element = 0.0
                BicepsFemoris_row_element = 0.0
                GluteausMaximus_row_element = 0.0
                print('exception')

            vastusMedialis = np.append(
                vastusMedialis, vastusMedialis_row_element)
            BicepsFemoris = np.append(BicepsFemoris, BicepsFemoris_row_element)
            GluteausMaximus = np.append(
                GluteausMaximus, GluteausMaximus_row_element)
        print('files readed and PINS 4,6,8 extracted from' + path)
        matrix = np.array([vastusMedialis, BicepsFemoris, GluteausMaximus])
    return matrix

# used to read only the pins 4,6,8

# row_length is how many elements the files' lins
def get_file_as_matrix(path, row_length):

    arr = np.empty((0, row_length), dtype=float)

    with open(path, "r") as filestream:
        for line in filestream:  # get a line of the stream
            # check if the string floatabele
            try:
                # split the line into element srings and float the elements # ro float is recommended for best performance
                rowElements = np.array(
                    line.strip().split(","), dtype=np.float32)
            except:
                print('exeption2')
                continue

            twoD = [rowElements]
            arr = np.append(arr, twoD, axis=0)

    print(arr, arr.dtype)
    return arr


# #P4_P6_P8_Reader("EMG Data/Patrick/OpenBCI-RAW-Patrick_Back_Squat_Fersenerhöhung.txt")
# get_file_as_matrix(
#     "EMG Data/Patrick/P4_P6_P8_Extracted/OpenBCI-RAW-Patrick_Back_Squat_Fersenerhöhung.txt.mini.txt", 3)



def get_specific_row(arr, row_numb):
    return arr[:, row_numb]
# select some colums and export it into file


def get_all_files_in_given_folder(folder_path):

    Folderslist = os.walk(folder_path)

    # every row contains a folder data
    # Folderslist [0] -> root = rootfolderpath , underfoldersNames => [folder1,folder2 ,...] , FilesNames => [file1,file2,file3,...]
    # Folderslist [1] -> root = rootfolderpath/folder1 , underfoldersNames => [folderA] , FilesNames => [file]
    # Folderslist [2] -> root = rootfolderpath/folder2 , underfoldersNames => [] , FilesNames => [fileX]
    # folder1 and folder2 are a underfolders of folder0
    # folderB a underfolders of folder
    # folder2 contains no folders and one file which is fileX

    for root, underfoldersNames, FilesNames in Folderslist:

        ############### create a list of all files pathes ; file path = given folder path + file name #################

        # get into the list FilesNames and append the root path
        # root path is actually equal to the given folder path because we are in the first loop cycle
        paths_of_all_files_in_the_folder = [
            "{}{}".format(root + '/', filename) for filename in FilesNames]
        break  # to get only the files in the root folder not also in the underfolders

    print(paths_of_all_files_in_the_folder)
    return paths_of_all_files_in_the_folder


#t1 = datetime.now().microsecond

#deltaT = datetime.now().microsecond - t1
#print(deltaT)
