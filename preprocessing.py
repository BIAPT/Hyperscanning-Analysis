from glob import glob
import os
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
import matplotlib.pyplot as plt
import numpy as np
from autoreject import AutoReject #Auto rejecting bad epoch
import autoreject_bads as ar #custom autoreject
from hypyp import prep
import pandas as pd

"""
Preprocessing steps
GOAL:
Works on custom nested folder structure
Change the annotation logic
Call autoreject_bads_ch internally 
Important!!!
Will there be an HEOG VEOG channel or no?
"""

#Epoching
#Out of place operation, doesn't modify the input signal
#returns epoch
def generate_epoch(signal, report, duration=10):
    #Not done yet
    signal_epoch = mne.make_fixed_length_epochs(
        signal, 
        duration=duration, #10 second slice
        preload=True,
        reject_by_annotation=True)
    signal_epoch.pick_types(eeg=True)
    fig_epoch = signal_epoch.compute_psd(picks='eeg', fmax=50).plot(show=False)
    plt.title("PSD of Epochs")
    report.add_figure(
        fig=fig_epoch,
        title="Epochs PSD",
        image_format='PNG'
    )
    plt.close(fig_epoch)
    return signal_epoch

#Perform autoreject (Test diff param)
def reject_epoch(epochs, report, n_interpolate=[1, 4, 8, 12], consensus=[0.1, 0.2, 0.3, 0.4], cv=10): #Keep playing with these params
    picks = mne.pick_types(epochs.info, eeg=True, eog=False, exclude="bads")
    thresh_method = "bayesian_optimization"
    autoreject = AutoReject(n_interpolate=n_interpolate,
                            consensus=consensus,
                            cv=cv,
                            picks=picks,
                            thresh_method=thresh_method,
                            n_jobs=-1,
                            random_state=42,
                            verbose=False)
    ar_epoch, reject_log = autoreject.fit_transform(epochs, return_log=True)
    if any(reject_log.bad_epochs):
        fig_bad = epochs[reject_log.bad_epochs].plot(show=False, scalings=dict(eeg=100e-6))
        report.add_figure(
            fig=fig_bad, title='Autoreject bad epochs',
            image_format='PNG'
        )
        plt.close(fig_bad)
    fig_clean = ar_epoch.compute_psd(picks='eeg', fmax=50).plot(show=False)
    plt.title("PSD of epochs post ar")
    report.add_figure(
        fig=fig_clean,
        title="Post-Autoreject",
        image_format='PNG'
    )
    plt.close(fig_clean)
    #Rejection log
    bad_epochs_mask = reject_log.bad_epochs
    total_epochs = len(bad_epochs_mask)
    dropped_count = int(sum(bad_epochs_mask))
    dropped_pct = (dropped_count / total_epochs) * 100
    retained_count = total_epochs - dropped_count

    #summary in HTML format
    summary_html = f"""
    <div style="font-family: sans-serif; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
        <h3 style="margin-top: 0;">Epoch Rejection Summary</h3>
        <table style="width: 100%; border-collapse: collapse; text-align: left;">
            <tr style="border-bottom: 1px solid #ddd;">
                <th style="padding: 8px;">Total Epochs Processed</th>
                <td style="padding: 8px;">{total_epochs}</td>
            </tr>
            <tr style="border-bottom: 1px solid #ddd;">
                <th style="padding: 8px;">Epochs Retained</th>
                <td style="padding: 8px;"><b>{retained_count}</b> ({100 - dropped_pct:.1f}%)</td>
            </tr>
            <tr style="border-bottom: 1px solid #ddd;">
                <th style="padding: 8px;">Epochs Dropped</th>
                <td style="padding: 8px; color: #d9534f;"><b>{dropped_count}</b> ({dropped_pct:.1f}%)</td>
            </tr>
        </table>
    </div>
    """

    report.add_html(
        html=summary_html,
        title="Epoch Summary",
        tags=("summary", "autoreject")
    )

    #Convert AutoReject Thresholds to DataFrame
    thresh_data = [
        {"Channel": ch, "Threshold (µV)": round(val * 1e6, 2)}
        for ch, val in autoreject.threshes_.items()
    ]
    df_thresh = pd.DataFrame(thresh_data)

    #Visualizing
    #AutoReject Rejection Matrix Plot
    #Color-codes good, bad (dropped), and interpolated channels per epoch
    fig_reject = reject_log.plot(orientation='horizontal', show_names=1, show=False) #
    fig_reject.set_size_inches(8, 12)
    report.add_figure(
        fig=fig_reject,
        title="AutoReject Epoch Matrix (Bad & Interpolated Channels)",
        tags=("epochs", "autoreject", "visualization")
    )
    plt.close(fig_reject)

    #Channel Threshold Barplot
    fig_thresh, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df_thresh["Channel"], df_thresh["Threshold (µV)"], color="#2b5c8f")
    ax.set_ylabel("Peak-to-Peak Threshold (µV)")
    ax.set_title("Peak-to-Peak Thresholds per Channel")
    ax.set_xticklabels(df_thresh["Channel"], rotation=90, fontsize=8)
    plt.tight_layout()

    report.add_figure(
        fig=fig_thresh,
        title="Channel Threshold Distribution",
        tags=("channels", "thresholds", "visualization")
    )
    plt.close(fig_thresh)

    #Channel Threshold Table
    thresh_html = f"""
    <h3>Calculated Channel Thresholds</h3>
    {df_thresh.to_html(index=False, classes='table table-striped', justify='left')}
    """
    report.add_html(
        html=thresh_html,
        title="Threshold Data Table",
        tags=("channels", "thresholds", "table")
    )
    return ar_epoch

#h_trans_bandwidth=5
def filter_signal(raw, report, l_freq=0.5, h_freq=45.0, notch=60, downsample=250):
    # raw_filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq).notch_filter(freqs=notch)
    #Try this one
    raw_filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, h_trans_bandwidth=5.0).notch_filter(freqs=notch)
    filter_ds = raw_filtered.resample(downsample)
    #Report downsampling
    filter_ds.compute_psd(fmax=50).plot()
    plt.show(block=True)
    #For manual bad channel selection
    filter_ds.plot(block=True)
    report.add_raw(
        raw=filter_ds,
        title="Filter First & Downsample",
        psd=True
    )
    return filter_ds

#Mutates the signal
def set_reference(signal, report, ref='average'):
    misc_ch_names = [signal.ch_names[i] for i in mne.pick_types(signal.info, misc=True)]
    signal.drop_channels(misc_ch_names)
    # if misc_ch_names != []:
    #     signal.set_eeg_reference(ref_channels=misc_ch_names)
    #     signal.set_channel_types({misc_ch_names[0]: 'eeg'})
    # else:
    #     signal.add_reference_channels(ref_channels='CPz')
    #     signal.set_channel_types({'CPz': 'eeg'})
    signal.set_eeg_reference(ref_channels=ref)
    #Report reref
    report.add_raw(
        raw=signal,
        title=f"{ref} reference",
        psd=True
    )

#For EEG where HEOG and VEOG are present
def ica_remove_HVEOG(signal, report):
    signal.set_channel_types({'VEOG': 'eog', 'HEOG': 'eog'})
    eeg_picks = mne.pick_types(signal.info, eeg=True, meg=False, exclude='bads')
    n_eeg_channels = len(eeg_picks)
    ica = ICA(n_components=n_eeg_channels-1, 
                  random_state=97, 
                  max_iter="auto",
                  method='infomax', 
                  fit_params=dict(extended=True))
        
    ica_train = signal.copy()
    ica.fit(ica_train, picks='eeg')
    eog_indices, eog_scores = ica.find_bads_eog(ica_train, ch_name=['VEOG', 'HEOG'])
    ica.exclude = eog_indices
    cleaned_data = signal.copy()
    ica.apply(cleaned_data)

    #Report ICA
    report.add_ica(
        ica=ica,
        title="ICA EOG",
        inst=signal,
        eog_scores=eog_scores,
        n_jobs=None
    )
    fig = cleaned_data.compute_psd(picks='eeg',fmax=50).plot(show=False)
    plt.title("PSD after ocular artifact removed")
    report.add_figure(
        fig=fig,
        title="PSD post ICA",
        image_format='PNG'
    )
    plt.close(fig)
    return cleaned_data

#Use ICA_Label to remove artifacts
#Input signal here is filtered at 0.5~45hz
def remove_artifacts(signal, report):
    ica = ICA(n_components=None, 
              random_state=42, 
              max_iter="auto",
              method='infomax', 
              fit_params=dict(extended=True))
    
    ica_train = signal.copy().filter(l_freq=1.0, h_freq=45.0, h_trans_bandwidth=5.0).notch_filter(freqs=60.0)
    ica.fit(ica_train)
    labels = label_components(ica_train, ica, method='iclabel')
    
    target_labels = ['muscle', 'channel_noise', 'heart', 'line_noise']
    exclude_idx = [
        idx for idx, (label, prob) in enumerate(zip(labels['labels'], labels['y_pred_proba']))
        if label in target_labels and prob > 0.80
    ]
    
    ica.exclude = exclude_idx
    cleaned_data = signal.copy()
    ica.apply(cleaned_data)

    report.add_ica(
        ica=ica,
        title="ICA Artifact Removal (Filtered Thresholds)",
        inst=signal,
        n_jobs=None
    )
    
    label_data = [
        {"Component": f"ICA{idx:03d}", "Classification": lbl, "Confidence": f"{prob:.2%}"}
        for idx, (lbl, prob) in enumerate(zip(labels["labels"], labels["y_pred_proba"]))
    ]
    df_labels = pd.DataFrame(label_data)
    
    #Highlight the specific components that met the >80% probability threshold
    def highlight_bads(row):
        if row.name in exclude_idx:  
            return ['background-color: #ffcccc'] * len(row)
        return [''] * len(row)
        
    styled_df = df_labels.style.apply(highlight_bads, axis=1).hide(axis="index").to_html()

    labels_html = f"""
    <div style="font-family: sans-serif; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
        <h3 style="margin-top: 0;">ICLabel Component Classifications</h3>
        <p>Components classified as <b>muscle artifact, channel noise, heart beat,</b> or <b>line noise</b> with >80% probability have been automatically excluded.</p>
        {styled_df}
    </div>
    """
    
    report.add_html(
        html=labels_html,
        title="ICLabel Classifications",
        tags=("ica", "labels", "table")
    )

    #Spectral leakage???
    fig = cleaned_data.compute_psd(picks='eeg', fmax=50).plot(show=False)
    plt.title("PSD post ICA (Filtered)")
    report.add_figure(
        fig=fig,
        title="PSD post ICALabel",
        image_format='PNG'
    )
    plt.close(fig)
    return cleaned_data



#Use this method if there is only EOG present in the EEG recording
def ica_remove_EOG():
    pass

#Mutates the raw object
def crop_signal(raw, samp_freq):
    annotations = raw.annotations
    din_time = annotations.onset[annotations.description == 'DIN1']
    raw.crop(tmin=din_time[0], tmax=din_time[1])

#Mutates the raw object
def set_montage(raw, report, type='standard_1020'):
    # montage = mne.channels.make_standard_montage(type)
    montage = mne.channels.make_standard_montage('GSN-HydroCel-64_1.0') #double check this tmr
    #To make this work for everything this needs to change
    # mapping ={ #This is only for CARTBIND dataset
    #     'FPz':'Fpz'
    # }
    if 'VREF' in raw.ch_names:
        raw.set_channel_types({'VREF': 'misc'})
    # raw.rename_channels(mapping=mapping)
    raw.set_montage(montage=montage, on_missing='warn')
    fig = raw.plot_sensors(show_names=True)
    report.add_figure(
        fig=fig,
        title=f"{type} system plot",
        image_format='PNG'
    )
    plt.close(fig)

def save_result(obj, save_path, title, overwrite):
    if isinstance(obj, mne.Report):
        obj.save(fname=f"{save_path}/{title}", overwrite=overwrite)
    else:
        obj.save(f"{save_path}/{title}", overwrite=overwrite)

def main(input_dir, output_dir, sampling_freq, downsample_freq, montage):
    path = input_dir
    save_path = output_dir
    dropped = []
    folders = glob(f"{path}/*")
    epoch_num = []
    file_paths = []
    id_dic = {}
    #get the file paths
    for file in sorted(folders):
        if os.path.basename(file) == 'outputs':
            continue
        #here ID refers to dyad ID
        ID = os.path.basename(file).split("_")[1]
        print(ID)
        if not os.path.basename(file).endswith(('.mff', '.raw')): #did not know .mff was a dir format
            continue
        file_paths.append(file)

        #Handle two files at a time here (How to handle pA & pB differently?)
        #+ change the folder structure in the README (has changed)
        basename = os.path.basename(file)
        #remove suffix 
        file_name = os.path.splitext(basename)[0]
        patient_ID = file_name.split("_")[-1]
        #MNE Report
        report_title = f"Hyperscan_{ID}_{patient_ID}"
        report = mne.Report(title=report_title)
        id_dic[patient_ID] = report


    """Make sure the files are not empty"""
    # if file == []:
    #     print(f"=======File is empty: Skipping {ID}=======")
    #     continue
    # else:
    #     print(f"=======Processing {ID} Time {patient_ID}=======")

    """
    Filter logic +
    Artifact + other methods that change the raw signal
    should come here
    """   
    raw_file = file_paths[0]
    mff_file = file_paths[1]

    raw = mne.io.read_raw_egi(raw_file, preload=False)
    mff = mne.io.read_raw_egi(mff_file, preload=False)
    
    crop_signal(raw=raw, samp_freq=sampling_freq)
    crop_signal(raw=mff, samp_freq=sampling_freq)
    raw.load_data()
    mff.load_data()
    print(id_dic)
    set_montage(raw=raw, report=id_dic['pA'], type=montage)
    set_montage(raw=mff, report=id_dic['pB'], type=montage)

    filter_raw = filter_signal(raw=raw, report=id_dic['pA'], downsample=downsample_freq)
    filter_mff = filter_signal(raw=mff, report=id_dic['pB'], downsample=downsample_freq)
    print(id_dic)

    # cleaned_signal = ica_remove_HVEOG(signal=filtered_signal, report=report) should the hypyp ICA come here?

    """Failed Cleaning???"""
    # if cleaned_signal == None:
    #     dropped.append(f"{ID}_{patient_ID}")
    #     continue

    #Drop all bad channels (not needed for feature generations)
    filter_raw.drop_channels(raw.info['bads'])
    filter_mff.drop_channels(mff.info['bads'])

    
    save_result(obj=filter_raw, save_path=f"{save_path}", 
                title=f"{ID}_pA_filtered_eeg.fif", overwrite=True)
    save_result(obj=filter_mff, save_path=f"{save_path}", 
                    title=f"{ID}_pB_filtered_eeg.fif", overwrite=True)
    
    set_reference(signal=filter_raw, report=id_dic['pA'])
    set_reference(signal=filter_mff, report=id_dic['pB'])

    epochs_raw = generate_epoch(signal=filter_raw, report=id_dic['pA'], duration=2)
    epochs_mff = generate_epoch(signal=filter_mff, report=id_dic['pB'], duration=2)

    """Hypyp ICA + Autoreject come here"""
    #make sure the epoch count is the same
    assert epochs_raw.info['sfreq'] == epochs_mff.info['sfreq']
    #redorder the channels
    ch_order = sorted(epochs_mff.ch_names)
    epochs_raw.reorder_channels(ch_order)
    epochs_mff.reorder_channels(ch_order)

    min_rank = min(len(mne.pick_types(epochs_raw.info, eeg=True, meg=False, exclude='bads'))-1,
                len(mne.pick_types(epochs_mff.info, eeg=True, meg=False, exclude='bads'))-1)
    print("Using min rank: ", min_rank)
    icas = prep.ICA_fit([
        epochs_raw, epochs_mff
    ],
        n_components=min_rank,
        method='infomax',
        fit_params=dict(extended=True),
        random_state=42
    )

    #Select the relevant independent components for artefact rejection
    cleaned_epochs_ICA = prep.ICA_choice_comp(icas, [epochs_raw, epochs_mff])
    print('ICA correction completed.')

    #save mne Report
    subjects = ["pA", "pB"]
    for idx, subj in enumerate(subjects):
        ica = icas[idx]
        epochs_clean = cleaned_epochs_ICA[idx]

        # Plot ICA properties / topographies of excluded components
        if len(ica.exclude) > 0:
            fig_ica_exclude = ica.plot_components(
                picks=ica.exclude, show=False, title=f"Excluded ICs ({subj})"
            )
            id_dic[subj].add_figure(
                fig=fig_ica_exclude,
                title="ICA Excluded Components",
                caption=f"Excluded IC indices: {ica.exclude}",
                image_format="png",
            )
            plt.close(fig_ica_exclude)

    #Apply Autoreject
    cleaned_epochs_AR, dic_AR = prep.AR_local(
        cleaned_epochs_ICA,
        strategy="union",
        threshold=50.0,
        verbose=True
    )
    #save autoreject report
    print(dic_AR.keys())
    for idx, subj in enumerate(subjects):
        if subj == 'pA':
            sub = 'S1'
        else:
            sub = 'S2'
        epochs_ar = cleaned_epochs_AR[idx]
        ar_log = dic_AR[sub]  # Autoreject log/dictionary for this participant

        # Add Epoch drop log figure (shows which epochs/channels were dropped by AR)
        fig_drop = epochs_ar.plot_drop_log(show=False)
        id_dic[subj].add_figure(
            fig=fig_drop,
            title="Autoreject Drop Log",
            caption=f"Percentage of dropped epochs: {epochs_ar.drop_log_stats():.1f}%",
            image_format="png",
        )
        plt.close(fig_drop)

        # Optional: Plot sample clean epoch traces
        fig_epochs = epochs_ar.plot(
            n_epochs=5, n_channels=20, show=False, title=f"Cleaned Epochs ({subj})"
        )
        id_dic[subj].add_figure(
            fig=fig_epochs,
            title="Post-Autoreject Signal Traces",
            caption="First 5 epochs following ICA and Autoreject cleanup",
            image_format="png",
        )
        plt.close(fig_epochs)

    print(type(cleaned_epochs_AR))
    participant_1 = cleaned_epochs_AR[0]
    participant_2 = cleaned_epochs_AR[1]
    print(type(cleaned_epochs_AR[0]))


    # ar_epochs = reject_epoch(epochs=epochs, report=report)
    # #Save epoch
    save_result(obj=participant_1, save_path=f"{save_path}", 
                title=f"{ID}_pA_epo.fif", overwrite=True)
    save_result(obj=participant_2, save_path=f"{save_path}", 
                    title=f"{ID}_pB_epo.fif", overwrite=True)

    #check if dir exists
    if not os.path.isdir(f'{save_path}/Reports'):
        os.makedirs(f'{save_path}/Reports', exist_ok=True)
    #Save mne Report
    save_result(obj=id_dic['pA'], save_path=f"{save_path}/Reports", 
                title=f"{ID}_pA_report.html", overwrite=True)
    save_result(obj=id_dic['pB'], save_path=f"{save_path}/Reports", 
                    title=f"{ID}_pB_report.html", overwrite=True)
            
if __name__ =="__main__":
    main()
