import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
"""Testing script
"""

#blue amp
mff_path = '/Volumes/NINET/Hyperscan/d01/sing_d01_pB.mff'
#grey amp
raw_path = '/Volumes/NINET/Hyperscan/d01/sing_d01_pA.RAW'

#read files
read_mff = mne.io.read_raw_egi(input_fname=mff_path)
read_RAW = mne.io.read_raw_egi(input_fname=raw_path)

"""
Use only one annotation from the participants
1. Extract DIN1 from both & align
2. Extract rest, sing, stry from one participant & use that as reference for the other participant
3. Start preprocessing
4. Analysis done using HyPyP
"""

mff_annot = read_mff.annotations
raw_annot = read_RAW.annotations

#ignore this (keep only one DIN1 for RAW file)
raw_annot.delete(list(range(1, len(raw_annot))))

mff_start_time = mff_annot.onset[mff_annot.description == 'DIN1']
raw_start_time = raw_annot.onset[raw_annot.description == 'DIN1']

#crop the signal
read_mff.crop(tmin=mff_start_time[0])
read_RAW.crop(tmin=raw_start_time[0])

#create a dictionary
annot_dic = {
    'rest': '1___',
    'sing': '2___',
    'stry': '3___'
}
onset_dic = {}
for key,value in annot_dic.items():
    mask = (read_mff.annotations.description == value)
    onset_dic[key] = read_mff.annotations.onset[mask]

print(onset_dic)
