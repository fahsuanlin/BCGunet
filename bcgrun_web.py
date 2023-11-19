import numpy as np
from scipy.io import savemat
import h5py
from unet import bcgunet
import platform
import os
import time
import os
import gradio as gr

dir = os.path.dirname(os.path.realpath(__file__)) + "/tmp"
os.makedirs(dir, exist_ok=True)


def run(
    files: list[bytes], lr: float, winsec: int, iters: int, onecycle: bool
) -> list[str]:
    task = os.path.join(dir, str(int(time.time())))
    os.makedirs(task)

    outputs = []

    for i, file in enumerate(files):
        input = os.path.join(task, str(i) + ".mat")
        with open(input, "wb") as o:
            o.write(file)

        output = os.path.join(task, str(i) + "_clean.mat")

        mat = h5py.File(input, "r")
        ECG = np.array(mat["ECG"]).flatten()
        EEG = np.array(mat["EEG_before_bcg"]).T

        EEG_unet = bcgunet.run(
            EEG,
            ECG,
            iter_num=iters,
            winsize_sec=winsec,
            lr=lr,
            onecycle=onecycle,
        )
        result = dict()
        result["EEG_clean"] = EEG_unet

        savemat(output, result, do_compression=True)
        outputs.append(output)

    return outputs

matlab_file_spec = """
ECG Data (input_ecg): 1D Numeric Array.
This array should contain the raw electrocardiogram (ECG) data in shape of (n_points,). The sampling frequency should be 5000 Hz.

EEG Data (input_eeg): 2D Numeric Array.
Each row corresponds to an EEG channel, and each column corresponds to a time point. We expect 64 EEG channels and 5000 Hz sampling frequency. The shape of the array should be (64, n_points).
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
        outputs=[gr.File(label="Output File", file_count="multiple")],
        allow_flagging="never",
    )

    app.launch()


if __name__ == "__main__":
    main()
    if platform.system() == "Windows":
        os.system("pause")
