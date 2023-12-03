import numpy as np
from scipy.io import savemat
import h5py
from unet import bcgunet
import argparse
import platform
import os
import time
import glob
import os


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
    return ["_".join(parts) for parts in result]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="+", type=str, help="Input mat file")
    parser.add_argument("-o", "--output", default=None, help="Output Path")
    parser.add_argument(
        "-f", "--frequency", default=5000, type=int, help="Sampling frequency"
    )
    parser.add_argument(
        "-i", "--iteration", default=5000, type=int, help="Number of iterations"
    )
    parser.add_argument(
        "-l", "--learning-rate", default=1e-3, type=float, help="Learning rate"
    )
    parser.add_argument(
        "-w", "--window-size", default=2, type=int, help="Window size (seconds)"
    )
    parser.add_argument(
        "-noc",
        "--no-one-cycle",
        action="store_true",
        help="Disable one cycle scheduler",
    )
    parser.add_argument(
        "--ecg", default="ECG", type=str, help="Variable name for ECG (input)"
    )
    parser.add_argument(
        "--bce",
        default="EEG_before_bcg",
        type=str,
        help="Variable name for BCG corropted EEG (input)",
    )
    parser.add_argument(
        "--eeg",
        default="EEG_clean",
        type=str,
        help="Variable name for clean EEG (output)",
    )
    args = parser.parse_args()

    print(f"Settings: {args}")
    print("Starting BCGunet.....")

    # if the input file is a folder or contains an asterisk
    # use the glob function to find all input files.

    ffs = args.input
    if os.path.isdir(args.input[0]):
        ffs = glob.glob(os.path.join(args.input[0], "*.mat"))

    elif "*" in args.input[0]:
        ffs = glob.glob(args.input[0])

    short_ffs = remove_common_substrings(ffs)
    print(f"Total files: {len(ffs)}")
    for ii in range(len(ffs)):
        f = ffs[ii]
        short_f = short_ffs[ii]

        # The following is for preparing the output directory
        # If the user provides an output directory and it does not exist, create it for them
        # If not provided, the default output directory is the folder of the input file

        f_output_dir = args.output

        if f_output_dir is None:
            f_output_dir = os.path.dirname(os.path.abspath(f))
        else:
            os.makedirs(f_output_dir, exist_ok=True)

        if len(ffs) > 1:
            f_output = short_f.replace(".mat", "_unet.mat")
        else:
            f_output = os.path.basename(f).replace(".mat", "_unet.mat")
        ff_output = os.path.join(f_output_dir, f_output)

        print(f"{ii + 1}: Processing {f}.....")
        # to create shorter filename for multiple mat files

        t = time.time()
        mat = h5py.File(f, "r")
        ECG = np.array(mat[args.ecg]).flatten()
        EEG = np.array(mat[args.bce]).T

        # (input_eeg, input_ecg, sfreq=5000, iter_num=5000, winsize_sec=2, lr=1e-3, onecycle=True)
        EEG_unet = bcgunet.run(
            EEG,
            ECG,
            iter_num=args.iteration,
            winsize_sec=args.window_size,
            lr=args.learning_rate,
            onecycle=not args.no_one_cycle,
            sfreq=args.frequency,
        )
        result = dict()
        result[args.eeg] = EEG_unet

        savemat(ff_output, result, do_compression=True)

        print("Writing output:", ff_output)
        print("Processing time: %d seconds" % (time.time() - t))


if __name__ == "__main__":
    main()
    if platform.system() == "Windows":
        os.system("pause")