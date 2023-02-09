# BCGunet: Suppressing BCG artifacts on EEG collected inside an MRI scanner

Ballistocardiogram (BCG) is the induced electric potentials caused by heartbeats when the EEG data are collected with a strong ambient magnetic field. This scenario is common in concurrent EEG-MRI acquisitions. One particular application of concurrent EEG-MRI is to delineate the irritative zones responsible for generating inter-ictal spikes (IIS) in medically refractory epilepsy patients. Specifically, EEG is used to detect onsets of spikes and these timing are used to inform functional MRI time series analysis. However, with strong BCG artifacts, the EEG data can be seriously corrupted and thus make spike annotation difficult.

This project aims at using machine learning approaches to suppress BCG artifacts. We will use Unet as the artificual neural network structure to tackle this challenge.

## Data

Data were EEG time series collected inside a 3T MRI scanner (Skyra, Siemens). EEG were sampled by a 32-channel systemm (Brain Products) with electrodes arranged by the international 10-20 standard. EEG were sampled at 5,000 Hz.

### Eyes open/closed in healthy control subjects
Each subject had two sessions of data. One was "eyes-open" and the other was "eyes-closed", where subjects were instructed laying in the MRI without falling sleep but keeping their eyes open and closed, respectively. This is a resting-state recording. 
During the recording, the MRI scanner did not collect any images. No so-called "gradient artifacts" caused by the swithcing of the imaging gradient coils of MRI was present.

## Code
- Data input (Matlab): read EEG data. Each EEG recording has three files with .eeg, .vmrk, and .vhdr file suffix. 
**NOTE** Do not change the file names because data are associated with the file name.

- [Unet basic structure and BCG suppression (Python)] (https://github.com/fahsuanlin/BCGunet/blob/main/bcg_unet/unet1d-simple.ipynb): perform BCG suppression by Unet, including training and testing of data from the same subject.

- Assessment (Matlab): Calculate the alpha-band (10-Hz) power at all EEG electrodes. We expect that stronger alpha-band neural oscillations are found at the parietal lobe of the subject when he/she closed eyes than opened eyes after successful BCG artifact suppression.

- Rendering (Matlab): tools to render EEG data over a scalp.

## External resources.
An RNN-type BCG artifact suppression method can be found [here](https://github.com/jiaangyao/BCGNet)
