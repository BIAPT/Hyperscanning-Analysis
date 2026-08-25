from glob import glob
import os
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
import matplotlib.pyplot as plt
import numpy as np
from autoreject import AutoReject #Auto rejecting bad epoch
import pandas as pd

"""
Preprocessing steps
GOAL:
Make it so that this preprocessing pipeline works for BIDS structures
Currently only work for CARTBIND (Nested Structure)
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
def crop_signal(raw, samp_freq=5000):
    print(f"Raw information: {raw}")
    events, event_id = mne.events_from_annotations(raw)
    print(f"Events and corresponding id: {event_id}: {events}")
    """
    First row gives us the point where the recording started
    Second row gives us the position where the event occurs
    Third row gives us the position where the event ends
    """
    print(events.shape)
    event_start_val = event_id.get("Comment/Start Eyes Closed ")
    event_end_val = event_id.get("Comment/Stop Eyes Closed")

    """Early exit"""
    if event_end_val is None:
        event_end_val = event_id.get("Comment/Start Eyes Open")
    if event_end_val is None:
        return None

    #6db to 3db

    start_idx = events[:,2].tolist().index(event_start_val)
    end_idx = events[:,2].tolist().index(event_end_val)
    event_start = events[start_idx, 0]
    event_end = events[end_idx, 0]

    print(f"Event begins: {event_start}")
    print(f"Event ends: {event_end}")

    """Trim the eeg signal"""
    samp_freq = samp_freq
    t_start = event_start/samp_freq
    t_end = event_end/samp_freq
    raw.crop(tmin=t_start, tmax=t_end)

#Mutates the raw object
def set_montage(raw, report, type='standard_1020'):
    montage = mne.channels.make_standard_montage(type)
    #To make this work for everything this needs to change
    mapping ={ #This is only for CARTBIND dataset
        'FPz':'Fpz'
    }
    raw.rename_channels(mapping=mapping)
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
    for folder in sorted(folders)[7:8]:
        ID = os.path.basename(folder).split("_")[1]
        print(ID)
        subfolders = sorted(glob(f"{folder}/*"))
        #Subfolders ex. T0, T1, T2
        print(subfolders)
        for subfolder in subfolders:
            time = os.path.basename(subfolder) #T0/T1/T2
            #MNE Report
            report_title = f"UBC_CARTBIND_{ID}_{time}"
            report = mne.Report(title=report_title)
            file = glob(f"{subfolder}/CBN02_{ID}_REST_EC_{time}_UBC.vhdr")
            if file == []:
                print(f"=======File is empty: Skipping {ID}=======")
                continue
            else:
                print(f"=======Processing {ID} Time {time}=======")
            """
            Filter logic +
            Artifact + other methods that change the raw signal
            should come here
            """    
            file = glob(file[0])
            raw = mne.io.read_raw_brainvision(file[0], preload=False)
            #crop_signal may be different for different EEG recordings
            crop_signal(raw=raw, samp_freq=sampling_freq)
            raw.load_data()
            set_montage(raw=raw, report=report, type=montage)
            filtered_signal = filter_signal(raw=raw, report=report, downsample=downsample_freq)
            cleaned_signal = ica_remove_HVEOG(signal=filtered_signal, report=report)
            # cleaned_signal = remove_artifacts(signal=cleaned_signal, report=report)
            if cleaned_signal == None:
                dropped.append(f"{ID}_{time}")
                continue
            #Drop all bad channels (not needed for feature generations)
            cleaned_signal.drop_channels(raw.info['bads'])
            save_result(obj=cleaned_signal, save_path=f"{save_path}/{time}_filtered_eeg", 
                        title=f"CBN02_{ID}_REST_EC_{time}_UBC_filtered_eeg.fif", overwrite=True)
            
            set_reference(signal=cleaned_signal, report=report)
            epochs = generate_epoch(signal=cleaned_signal, report=report)
            ar_epochs = reject_epoch(epochs=epochs, report=report)
            report.save(fname=f"{save_path}/Reports/{time}/{report_title}_report.html", overwrite=True)
            #Save epoch
            save_result(obj=ar_epochs, save_path=f"{save_path}/{time}_epoch", 
                        title=f"CBN02_{ID}_REST_EC_{time}_UBC_epo.fif", overwrite=True)
            #Save mne Report
            save_result(obj=report, save_path=f"{save_path}/Reports/{time}", 
                        title=f"{report_title}_report.html", overwrite=True)

if __name__ =="__main__":
    main()