from glob import glob
import os
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
import matplotlib.pyplot as plt
import numpy as np
from autoreject import AutoReject #Auto rejecting bad epoch
import pandas as pd
#Hypyp tools
from hypyp import prep 
from hypyp import analyses
from hypyp import stats
from hypyp import viz


"""Testing HyPyP for now"""
#Most of the code is from
#https://github.com/ppsp-team/HyPyP/blob/master/tutorial/getting_started.ipynb
def main():
    #Use T0 epoch as an example
    epoch_path = "/Volumes/NINET/CARTBIND_EEG_data/CARTBIND_UBC/T0_epoch"
    epoch1 = mne.read_epochs(
        fname=f"{epoch_path}/CBN02_0001_REST_EC_T0_UBC_epo.fif",
        preload=True
    )
    epoch2 = mne.read_epochs(
        fname=f"{epoch_path}/CBN02_0002_REST_EC_T0_UBC_epo.fif",
        preload=True
    )
    #Make both epoch the same length
    mne.epochs.equalize_epoch_counts([epoch1, epoch2])
    sampling_rate = epoch1.info['sfreq']
    assert epoch1.info['sfreq'] == epoch2.info['sfreq']
    print('Sampling rate:', sampling_rate)

    #Compute min rank between two epochs
    min_rank = min(len(mne.pick_types(epoch1.info, eeg=True, meg=False, exclude='bads'))-1,
                   len(mne.pick_types(epoch2.info, eeg=True, meg=False, exclude='bads'))-1)
    print("Using min rank: ", min_rank)
    
    #Compute ICA on both epoch
    #ICA is causing matmul issues (div by zero)
    icas = prep.ICA_fit([
        epoch1, epoch2
    ],
        n_components=min_rank,
        method='infomax',
        fit_params=dict(extended=True),
        random_state=42
    )

    #Select the relevant independent components for artefact rejection
    cleaned_epochs_ICA = prep.ICA_choice_comp(icas, [epoch1, epoch2])
    print('ICA correction completed.')

    #Apply Autoreject
    cleaned_epochs_AR, dic_AR = prep.AR_local(
        cleaned_epochs_ICA,
        strategy="union",
        threshold=50.0,
        verbose=True
    )
    print(type(cleaned_epochs_AR))
    participant_1 = cleaned_epochs_AR[0]
    participant_2 = cleaned_epochs_AR[1]

    psd1 = analyses.pow(
        participant_1,
        fmin=7.5,
        fmax=11,
        n_fft=1000,
        n_per_seg=1000,
        epochs_average=True
    )

    # Compute PSD for participant 2 in the Alpha-Low band
    psd2 = analyses.pow(
        participant_2,
        fmin=7.5,
        fmax=11,
        n_fft=1000,
        n_per_seg=1000,
        epochs_average=True
    )  
    

if __name__ == "__main__":
    main()