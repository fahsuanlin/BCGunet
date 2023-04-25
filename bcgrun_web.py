import numpy as np
from scipy.io import savemat
import h5py
from bcgunet import bcgunet
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
        ],
        outputs=[gr.File(label="Output File", file_count="multiple")],
        allow_flagging="never",
    )

    app.launch()


if __name__ == "__main__":
    main()
    if platform.system() == "Windows":
        os.system("pause")
