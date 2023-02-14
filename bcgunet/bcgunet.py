
from os.path import *
import numpy as np
import random
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import tqdm
from scipy.signal import butter, sosfilt
from .unet import UNet1d


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False,
                 btype='band', output='sos')
    y = sosfilt(sos, data)
    return y


def norm(ecg):
    min1, max1 = np.percentile(ecg, [1, 99])
    ecg[ecg > max1] = max1
    ecg[ecg < min1] = min1
    ecg = (ecg - min1)/(max1-min1)
    return ecg



def run(input_eeg,
        input_ecg=None,
        sfreq=5000,
        iter_num=5000,
        winsize_sec=2,
        lr=1e-3,
        onecycle=True):
    
    window = winsize_sec * sfreq
    eeg_raw = input_eeg
    eeg_channel = eeg_raw.shape[0]
    
    eeg_filtered = eeg_raw * 0
    t = time.time()
    for ii in range(eeg_channel):
        eeg_filtered[ii, ...] = butter_bandpass_filter(
            eeg_raw[ii, :], 0.5, sfreq*0.4, sfreq)

    baseline = eeg_raw - eeg_filtered


    if input_ecg is None:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=1)
        ecg = norm(pca.fit_transform(eeg_filtered.T)[:, 0].flatten())
    else:
        ecg = norm(input_ecg.flatten())


    torch.cuda.empty_cache()
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    NET = UNet1d(n_channels=1, n_classes=eeg_channel, nfilter=8).to(device)
    optimizer = torch.optim.Adam(NET.parameters(), lr=lr)
    optimizer.zero_grad()
    maxlen = ecg.size
    if onecycle:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=iter_num)
    else:
        #constant learning rate
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)

    loss_list = []

    #randomly get  windows in ECG signal
    
    index_all = (np.random.random_sample(iter_num)*(maxlen-window)).astype(int)

    pbar = tqdm.tqdm(index_all)
    count = 0
    for index in pbar:
        count += 1
        ECG = ecg[index:(index + window)]
        EEG = eeg_filtered[:, index:(index + window)]
        ECG_d = torch.from_numpy(ECG[None, ...][None, ...]).to(device).float()
        EEG_d = torch.from_numpy(EEG[None, ...]).to(device).float()

        # step 3: forward path of UNET
        logits = NET(ECG_d)
        loss = nn.MSELoss()(logits, EEG_d)
        loss_list.append(loss.item())


        # Step 5: Perform back-propagation
        loss.backward() #accumulate the gradients
        optimizer.step() #Update network weights according to the optimizer
        optimizer.zero_grad() #empty the gradients
        scheduler.step()
        
        if count % 50 == 0:
            pbar.set_description(f"Loss {np.mean(loss_list):.3f}, lr: {optimizer.param_groups[0]['lr']:.5f}")
            loss_list = []


    EEG = eeg_filtered
    #ECG = norm(butter_bandpass_filter(data['ECG'], 0.5, 20, sfreq))
    ECG = ecg
    ECG_d = torch.from_numpy(ECG[None, ...][None, ...]).to(device).float()
    EEG_d = torch.from_numpy(EEG[None, ...]).to(device).float()
    with torch.no_grad():
        logits = NET(ECG_d)
    BCG_pred = logits.cpu().detach().numpy()[0, ...]

    neweeg = EEG - BCG_pred + baseline

    return neweeg


def morlet_psd(signal, sample_rate=5000, freq=10, wavelet='morl'):
    import pywt

    # Define the wavelet and scales to be used

    scales = np.arange(sample_rate)
    freqs = pywt.scale2frequency('morl', scales) * sample_rate
    indx = np.argmin(abs(freqs - freq))
    
    scale = scales[indx]

    #scale = pywt.frequency2scale('morl', freq/sample_rate)

    # Calculate the wavelet coefficients
    coeffs, freq = pywt.cwt(signal, scale, wavelet, 1/sample_rate)
    # Calculate the power (magnitude squared) of the coefficients
    power = np.abs(coeffs)**2

    # Average the power across time to get the power spectral density
    psd = np.mean(power, axis=1)

    return psd


def get_psd(eeg, sfreq=5000, freq=10):
    psd = []
    for ii in tqdm.tqdm(range(eeg.shape[0])):
        psd.append(morlet_psd(eeg[ii], sample_rate=sfreq, freq=freq))

    return np.array(psd)
