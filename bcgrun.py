import numpy as np
from scipy.io import savemat
import h5py
from bcgunet import bcgunet
import argparse
import platform
import os
import time
import glob

import difflib


import os
import difflib



import os
import difflib


def remove_common_substrings(strings):
    # Split each string by os.sep
    split_strings = [s.split(os.sep) for s in strings]

    # Find the shortest split string in the list
    shortest_split_string = min(split_strings, key=len)

    # Initialize common substrings list
    common_substrings = []

    # Iterate through the elements of the shortest split string
    for i in range(len(shortest_split_string)):
        # Check if the element is common in all split strings
        element = shortest_split_string[i]
        if all(element in string for string in split_strings):
            common_substrings.append(element)

    # Remove the common substrings from each split string
    result = []
    for string in split_strings:
        new_string = [part for part in string if part not in common_substrings]
        result.append(new_string)

    # Join the parts of the strings using os.sep
    return ['_'.join(parts) for parts in result]





def main():
    print('Starting BCGunet.....')
      
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='+', type=str, help='Input mat file')
    parser.add_argument('-o', '--output', default=None, help='Output Path')
    args = parser.parse_args()


    #The following is for preparing the output directory
    #If the user provides an output directory and it does not exist, create it for them
    #If not provided, the default output directory is the folder of the input file



    f_output_dir = args.output

    if f_output_dir is None:
        f_output_dir = os.path.dirname(os.path.abspath(f))
    else:
        os.makedirs(f_output_dir, exist_ok=True)

    #Additionally, if the input file is a folder or contains an asterisk
    # use the glob function to find all input files.

    ffs = args.input
    if os.path.isdir(args.input[0]):
        ffs = glob.glob(os.path.join(args.input[0], '*.mat'))

    elif '*' in args.input[0]:
        ffs = glob.glob(args.input[0])

    short_ffs = remove_common_substrings(ffs)
    for ii in range(len(ffs)):
        f = ffs[ii]
        short_f = short_ffs[ii]

        if len(ffs) > 1:
            f_output = short_f.replace('.mat', '_unet.mat')
        else:
            f_output = os.path.basename(f).replace('.mat', '_unet.mat')
        ff_output = os.path.join(f_output_dir, f_output)

        print(f'Processing {f}.....')
        # to create shorter filename for multiple mat files
        
        t = time.time()
        mat = h5py.File(f, 'r')
        ECG = np.array(mat['ECG']).flatten()
        EEG = np.array(mat['EEG_before_bcg']).T

        # (input_eeg, input_ecg, sfreq=5000, iter_num=5000, winsize_sec=2, lr=1e-3, onecycle=True)
        EEG_unet = bcgunet.run(EEG, ECG)
        result = dict()
        result['EEG_clean'] = EEG_unet



        savemat(ff_output, result, do_compression=True)

        print('Writing output:', ff_output)
        print('Processing time: %d seconds' % (time.time() - t))
    



if __name__ == "__main__":
    main()
    if platform.system() == 'Windows':
        os.system('pause')

