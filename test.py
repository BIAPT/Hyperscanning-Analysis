import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
"""Testing script
"""

#blue amp
mff_path = '/Volumes/Hyperscan/Hyperscanning/d02/test_d02_pB.mff'
#grey amp
raw_path = '/Volumes/Hyperscan/Hyperscanning/d02/test_d02_pA.raw'

#read files
read_mff = mne.io.read_raw_egi(input_fname=mff_path)
read_raw = mne.io.read_raw_egi(input_fname=raw_path)

"""
Use only one annotation from the participants
1. Extract DIN1 from both & align
2. Extract rest, sing, stry from one participant & use that as reference for the other participant
3. Start preprocessing
4. Analysis done using HyPyP
"""

mff_annot = read_mff.annotations
raw_annot = read_raw.annotations

mff_din_time = mff_annot.onset[mff_annot.description == 'DIN1']
raw_din_time = raw_annot.onset[raw_annot.description == 'DIN1']
print(mff_din_time)
print(raw_din_time)

#crop the signal
read_mff.crop(tmin=mff_din_time[0], tmax=mff_din_time[1])
read_raw.crop(tmin=raw_din_time[0], tmax=raw_din_time[1])

#create a dictionary
annot_dic = {
    'rest': '1___',
    'sing': '2___',
    'stry': '3___'
}
onset_dic = {}

#create onset dictionary later used in hypyp
for key,value in annot_dic.items():
    mask = (read_mff.annotations.description == value)
    onset_dic[key] = read_mff.annotations.onset[mask]

print(onset_dic)
data = read_mff.get_data()
read_raw.plot(block=True)


from hypyp import analyses
"""Testing PLV"""
# analyses.pair_connectivity(
#     ,
# )
"""
Functional connectivity between x and y 
y constant & choose random point in x & swap first and second (spectral property)
Destropy time coreelation
Real & random correlation
"""

#test opening .raw + this works now

#older (grey) amp doesn't use 1__ ask marks instead it just uses the actual string as the marker
#maybe use the .raw as the reference & crop based on that for the .mff file
#custom template (intergenerational_singing) must be used to get this marker