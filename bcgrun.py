import numpy as np
from scipy.io import savemat
import h5py
from bcgunet import bcgunet
import argparse
import platform
import os

def main():
      
    parser = argparse.ArgumentParser()
    parser.add_argument('input',  type=str, help='Input mat file')
    parser.add_argument('-o', '--output', default=None, help='Output Path')
    args = parser.parse_args()

    print('Starting BCGunet.....')

    f = h5py.File(args.input, 'r')
    ECG = np.array(f['ECG']).flatten()
    EEG = np.array(f['EEG_before_bcg']).T

    # (input_eeg, input_ecg, sfreq=5000, iter_num=5000, winsize_sec=2, lr=1e-3, onecycle=True)
    EEG_unet = bcgunet.run(EEG, ECG)
    result = dict()
    result['EEG_clean'] = EEG_unet
    savemat(args.input.replace('.mat', '_unet.mat'),
            result, do_compression=True)
    



if __name__ == "__main__":
    main()
    if platform.system() == 'Windows':
        os.system('pause')

