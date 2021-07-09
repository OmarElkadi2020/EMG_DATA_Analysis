import os
import matplotlib.pyplot as plt
import SignalFilter as sf
import numpy as np

# wirte a matrix extracted from a big file to another file
def filterd_file_writer(ExtractedMatrix, path_to_write):
    #transpose it to get every sensor reads in a column
    transposed_Matrix = np.transpose(ExtractedMatrix)
    print(transposed_Matrix.shape)
    np.savetxt(path_to_write + '.mini.txt', transposed_Matrix,
               delimiter=',', newline='\n')


def table_Writer(path_to_write , table):
    np.savetxt(path_to_write, table, delimiter=';', newline='\n', fmt='%s')


# all_folder_files_paths = get_all_files_paths('./EMG Data/Rico')

# for file_path in all_folder_files_paths:
#     extracted_PINS_Matrix = P4_P6_P8_Reader(file_path)
#     filterd_file_writer(extracted_PINS_Matrix, file_path)
