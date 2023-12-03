import numpy as np
from scipy.io import savemat
import h5py
from unet import bcgunet
import platform
import os
import time
import os
import gradio as gr
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("agg")

dir = os.path.dirname(os.path.realpath(__file__)) + "/tmp"
os.makedirs(dir, exist_ok=True)


def run(
    files: list[bytes],
    sfreq: int,
    lr: float,
    winsec: int,
    iters: int,
    onecycle: bool,
    ecg: str,
    bce: str,
    eeg: str,
) -> tuple[list[str], str]:
    task = os.path.join(dir, str(int(time.time())))
    os.makedirs(task)

    outputs = []

    for i, file in enumerate(files):
        input = os.path.join(task, str(i) + ".mat")
        with open(input, "wb") as o:
            o.write(file)

        output = os.path.join(task, str(i) + "_clean.mat")

        mat = h5py.File(input, "r")
        ECG = np.array(mat[ecg]).flatten()
        EEG = np.array(mat[bce]).T

        EEG_unet = bcgunet.run(
            EEG,
            ECG,
            iter_num=iters,
            winsize_sec=winsec,
            lr=lr,
            onecycle=onecycle,
            sfreq=sfreq,
        )
        result = dict()
        result[eeg] = EEG_unet

        savemat(output, result, do_compression=True)
        outputs.append(output)

        if i == 0:
            double_sfreq = sfreq * 2
            plt.figure(figsize=(12, 6), dpi=300)
            plt.plot(EEG[19, :double_sfreq], "b.-", label="Orig EEG")
            plt.plot(EEG_unet[19, :double_sfreq], "g.-", label="U-Net")
            plt.legend()
            plt.title("BCG Unet")
            plt.xlabel("Time (samples)")
            plot = os.path.join(task, str(i) + ".png")
            plt.savefig(plot)

    return outputs, plot

matlab_file_spec = """
ECG Data: The raw ECG data collected inside the MRI scanner.
This array should contain the raw electrocardiogram (ECG) data in shape of (n_points, 1).

EEG Data: The raw EEG data collected inside the MRI scanner.
Each row corresponds to an EEG channel, and each column corresponds to a time point. We expect 31 EEG channels. The shape of the array should be (n_points, 31).

P.S. n_points = sampling frequency * recording time (in seconds)
"""

def main():
    app = gr.Interface(
        title="BCG Unet",
        description="BCGunet: Suppressing BCG artifacts on EEG collected inside an MRI scanner\n\n" + matlab_file_spec,
        fn=run,
        inputs=[
            gr.File(
                label="Input Files (.mat)",
                type="binary",
                file_types=["mat"],
                file_count=["multiple", "directory"],
                info="Upload MATLAB .mat files. You can specify in-file variable names in the options below.",
            ),
            gr.Slider(
                label="Sampling Frequency",
                minimum=100,
                maximum=10000,
                step=100,
                value=5000,
                info="The sampling frequency of the EEG and ECG data. (Hz)",
            ),
            gr.Slider(
                label="Learning Rate",
                minimum=1e-5,
                maximum=1e-1,
                step=1e-5,
                value=1e-3,
                info="Adjusts the step size at each iteration of model training.",
            ),
            gr.Slider(
                label="Window Size (seconds)",
                minimum=1,
                maximum=10,
                step=1,
                value=2,
                info="Sets the time window for analyzing the EEG signal.",
            ),
            gr.Slider(
                label="Number of Iterations",
                minimum=1000,
                maximum=10000,
                step=1000,
                value=5000,
                info="The total number of passes for training.",
            ),
            gr.Checkbox(
                label="One Cycle Scheduler",
                value=True,
                info="Enable the learning rate policy that cycles the learning rate between two boundaries.",
            ),
            gr.Textbox(
                label="Variable name for ECG (input)",
                placeholder="Enter the variable name for ECG data",
                value="ECG",
                info="Specify the variable name in your .mat file that corresponds to the ECG data.",
            ),
            gr.Textbox(
                label="Variable name for BCG-corrupted EEG (input)",
                placeholder="Enter the variable name for BCG-corrupted EEG data",
                value="EEG_before_bcg",
                info="Specify the variable name in your .mat file that corresponds to the BCG-corrupted EEG data.",
            ),
            gr.Textbox(
                label="Variable name for clean EEG (output)",
                placeholder="Enter the variable name for clean EEG data",
                value="EEG_clean",
                info="Specify the variable name in your .mat file that corresponds to the clean EEG data.",
            ),
        ],
        outputs=[
            gr.File(label="Output File", file_count="multiple"),
            gr.Image(label="Output Image", type="filepath"),
        ],
        allow_flagging="never",
    )

    app.launch()


if __name__ == "__main__":
    main()
    if platform.system() == "Windows":
        os.system("pause")
