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


def shorten_filename(strings):
    # Initialize a list to store the modified strings
    modified_strings = []

    # Iterate through each string in the list
    for s in strings:
        # Replace the os.sep character with an underscore
        modified_string = s.replace(os.sep, '_')
        modified_strings.append(modified_string)

    # Initialize a list to store the duplicated substrings
    duplicates = []

    # Iterate through each pair of modified strings
    for i in range(len(modified_strings)):
        for j in range(i+1, len(modified_strings)):
            # Use SequenceMatcher to find the longest matching substring
            matcher = difflib.SequenceMatcher(
                None, modified_strings[i], modified_strings[j])
            matches = matcher.get_matching_blocks()

            # Extract the matching substrings from the modified strings
            for match in matches:
                if match.size > 0:
                    substring = modified_strings[i][match.a:match.a+match.size]
                    duplicates.append(substring)

    # Remove duplicates from the list
    duplicates = list(set(duplicates))

    return duplicates


def main():
    print('Starting BCGunet.....')
      
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='+', type=str, help='Input mat file')
    parser.add_argument('-o', '--output', default=None, help='Output Path')
    args = parser.parse_args()

    f_output_dir = args.output

    if f_output_dir is None:
        f_output_dir = os.path.dirname(os.path.abspath(f))
    else:
        os.makedirs(f_output_dir, exist_ok=True)

    ffs = args.input
    if os.path.isdir(args.input[0]):
        ffs = glob.glob(join(args.input[0], '*.mat'))

    elif '*' in args.input[0]:
        ffs = glob.glob(args.input[0])

    short_ffs = shorten_filename(ffs)
    for ii in range(len(ffs)):
        f = ffs[ii]
        print(f'Processing {f}.....')
        # to create shorter filename for multiple mat files
        short_f = short_ffs[ii]
        t = time.time()
        mat = h5py.File(f, 'r')
        ECG = np.array(mat['ECG']).flatten()
        EEG = np.array(mat['EEG_before_bcg']).T

        # (input_eeg, input_ecg, sfreq=5000, iter_num=5000, winsize_sec=2, lr=1e-3, onecycle=True)
        EEG_unet = bcgunet.run(EEG, ECG)
        result = dict()
        result['EEG_clean'] = EEG_unet

        if len(ffs) > 1:
            f_output = short_f.replace('.mat', '_unet.mat')
        else:
            f_output = os.path.basename(f).replace('.mat', '_unet.mat')
        ff_output = os.path.join(f_output_dir, f_output)

        savemat(ff_output, result, do_compression=True)

        print('Writing output:', ff_output)
        print('Processing time: %d seconds' % (time.time() - t))
    



if __name__ == "__main__":
    main()
    if platform.system() == 'Windows':
        os.system('pause')

