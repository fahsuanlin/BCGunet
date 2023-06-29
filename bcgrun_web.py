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
        )
        result = dict()
        result[eeg] = EEG_unet

        savemat(output, result, do_compression=True)
        outputs.append(output)

        if i == 0:
            plt.figure(figsize=(12, 6), dpi=300)
            plt.plot(EEG[19, :10000], "b.-", label="Orig EEG")
            plt.plot(EEG_unet[19, :10000], "g.-", label="U-Net")
            plt.legend()
            plt.title("BCG Unet")
            plt.xlabel("Time (samples)")
            plot = os.path.join(task, str(i) + ".png")
            plt.savefig(plot)

    return outputs, plot


def main():
    app = gr.Interface(
        title="BCG Unet",
        description="BCGunet: Suppressing BCG artifacts on EEG collected inside an MRI scanner",
        fn=run,
        inputs=[
            gr.File(
                label="Input Files (.mat)",
                type="binary",
                file_types=["mat"],
                file_count=["multiple", "directory"],
            ),
            gr.Slider(
                label="Learning Rate", minimum=1e-5, maximum=1e-1, step=1e-5, value=1e-3
            ),
            gr.Slider(
                label="Window Size (seconds)", minimum=1, maximum=10, step=1, value=2
            ),
            gr.Slider(
                label="Number of Iterations",
                minimum=1000,
                maximum=10000,
                step=1000,
                value=5000,
            ),
            gr.Checkbox(
                label="One Cycle Scheduler",
                value=True,
            ),
            gr.Textbox(
                label="Variable name for ECG (input)",
                value="ECG",
            ),
            gr.Textbox(
                label="Variable name for BCG corropted EEG (input)",
                value="EEG_before_bcg",
            ),
            gr.Textbox(
                label="Variable name for clean EEG (output)",
                value="EEG_clean",
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
