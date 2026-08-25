import numpy as np
import mne
import matplotlib.pyplot as plt
from collections import Counter

"""
For the report, the following should be marked
1. What channels are marked bad + total #
2. percentage of region thrown out (ch in specific region. ex. frontal)
"""
def autoreject_bads_ch(signal, report, zscore_thresh, n_fft=2048, fmax=45.0):
    #mne picks
    picks = mne.pick_types(signal.info, eeg=True, exclude='bads')

    #computing psd
    spectrum = signal.compute_psd(fmax=fmax, picks='eeg', n_fft=n_fft)
    psd, freq = spectrum.get_data(return_freqs=True)
    ch_names = spectrum.ch_names
    
    #specific window of frequencies of interest
    psd_scaled = np.array(psd*1e12)[:,24:]
    freq = freq[24:]
    mean_ch_power = np.mean(psd_scaled, axis=0) #scalar val per channel (only eeg)
    median_ch_power = np.median(psd_scaled, axis=0) #scalar val per channel (only eeg)

    #calculate std & zscore
    std_power = np.std(psd_scaled, axis=0)
    z_scores = np.abs(psd_scaled - mean_ch_power)/(std_power + 1e-12)
    bad_indices = np.where(z_scores > zscore_thresh)[0]

    #calculate the % of bad ch in the observed window
    length = len(freq) 
    counts = Counter(bad_indices)
    bads = [int(item) for item, count in counts.items() if count/length > 0.40]
    bad_ch = [ch_names[bad] for bad in bads]

    #detect bad contact via peak-to-peak calculation
    data = signal.get_data(picks='eeg')
    min_data = np.min(data, axis=1)
    max_data = np.max(data, axis=1)

    #lower bound
    bad_lower_bound = np.where((max_data - min_data) <= 5e-6)[0] #np.where() returns a tuple (unpack the tuple)
    bad_upper_bound = np.where((max_data - min_data) >= 500e-6)[0]
    bad_bounds_idx = np.concatenate((bad_lower_bound, bad_upper_bound), axis=0)
    bad_bound_ch = [ch_names[b] for b in bad_bounds_idx]

    bad_ch_names = list(set(bad_bound_ch + bad_ch))
    ratio = len(bad_ch_names)/len(ch_names)
    if ratio > 0.3:
        alert_html = f"""
        <div style="padding: 15px; margin: 10px 0; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; color: #721c24;">
            <h3 style="margin-top:0;"> Skipping autoreject_bads_ch: Excessive Bad Channels</h3>
            <p><b>Bad Channel Ratio:</b> {ratio:.1%} ({len(bad_ch_names)} / {len(ch_names)} channels marked bad)</p>
            <p><b>Threshold:</b> 30.0%</p>
            <p><b>Bad Channels:</b> {', '.join(bad_ch_names)}</p>
        </div>
        """
        #report in mne.Report
        report.add_html(
            html=alert_html,
            title="Bad Channel Threshold Exceeded",
            section="Quality Control",
        )
    elif bad_ch_names:
        #mark bad
        signal.info['bads'] = list(set(signal.info["bads"] + bad_ch_names))


    #report everything in mne.Report
    fig, ax = plt.subplots(figsize=(8, 4))
    # Plot normal channels in gray, flagged bad channels in orange/red
    print(ch_names)
    print("Bad ch: ", signal.info['bads'])
    print(len(ch_names))
    print("Shape of the psd: ", psd_scaled.shape)
    for i, ch_name in enumerate(ch_names):
        if ch_name in bad_ch_names:
            ax.plot(
                freq,
                psd_scaled[i],
                color="orange",
                alpha=0.6,
                linewidth=1,
                label="Bad Channel" if i == 0 else "",
            )
        else:
            ax.plot(freq, psd_scaled[i], color="gray", alpha=0.2, linewidth=0.8)

    # Plot Global Reference Baselines
    ax.plot(
        freq,
        mean_ch_power,
        color="red",
        linewidth=2,
        label="Global Mean PSD",
    )
    ax.plot(
        freq,
        median_ch_power,
        color="blue",
        linewidth=2,
        linestyle="--",
        label="Global Median PSD",
    )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density")
    ax.set_title(f"PSD Spectrum Baseline (Flagged {len(bad_ch_names)} channels)")
    ax.grid(True)

    # Clean up duplicate legend labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    # Pass plot directly into MNE Report
    if report is not None:
        report.add_figure(
            fig=fig,
            title="Auto-Reject Bad Channels (PSD Analysis)",
            caption=f"Rejected Channels: {', '.join(bad_ch_names) if bad_ch_names else 'None'}",
            section="Quality Control",
        )
    plt.close(fig)