import sys
sys.path.append(r"C:\Users\au682737\OneDrive - Aarhus universitet\PhD\MNE_projects")
import mne
from MNE_ASSR import *
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog
from TMSiFileFormats.file_readers import Poly5Reader
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import matplotlib.patches as mpatchespip
from scipy.io import loadmat
from scipy.signal import find_peaks, periodogram
from scipy.stats import ttest_ind
from mne.stats import fdr_correction
from mne.stats import permutation_cluster_test
#from sklearn.decomposition import PCA
import platform
from mne.time_frequency import psd_array_multitaper
from scipy.signal.windows import tukey
from mne.time_frequency import tfr_array_morlet
import scipy.stats as stats
from statsmodels.stats.multitest import fdrcorrection

def check_triggers(trig_points, expected_blocks=12, block_size=1000, threshold_factor=5):
    #plot the recorded triggers for inspection
    
    trig_points = np.asarray(trig_points).flatten()
    n_triggers = len(trig_points)
    intervals = np.diff(trig_points)

    # Detect where blocks start (long ISIs)
    median_interval = np.median(intervals)
    block_start_indices = np.where(intervals > threshold_factor * median_interval)[0] + 1

    # Include first and last as boundaries
    block_boundaries = np.concatenate(([0], block_start_indices, [n_triggers]))
    block_counts = np.diff(block_boundaries)

    # ✅ Consistency check
    total_count = np.sum(block_counts)
    if total_count != n_triggers:
        print(f"⚠️ Warning: mismatch in counts! {total_count=} vs {n_triggers=}")
    else:
        print(f"✔️ Total triggers accounted for: {total_count} (matches input)")

    # Identify blocks with missing triggers
    missing_blocks = [i+1 for i, count in enumerate(block_counts) if count < block_size]

    # Plot intervals and block markers
    plt.figure(figsize=(10,4))
    plt.plot(intervals, marker='o', linestyle='-', alpha=0.7)
    plt.title("Intervals Between Triggers")
    plt.xlabel("Trigger index")
    plt.ylabel("Interval (samples)")
    plt.grid(True)
    for idx in block_start_indices:
        plt.axvline(idx, color='r', linestyle='--', alpha=0.6)
    plt.show()

    # Print block summary
    print(f"\nDetected {len(block_counts)} blocks:")
    for i, count in enumerate(block_counts, 1):
        print(f"  Block {i}: {count} triggers")

    if missing_blocks:
        print(f"\n⚠️ Blocks with missing triggers: {missing_blocks}")
    else:
        print("\nAll blocks appear complete.")

    return {
        "intervals": intervals,
        "block_start_indices": block_start_indices,
        "block_boundaries": block_boundaries,
        "block_counts": block_counts,
        "missing_blocks": missing_blocks
    }

def triggers_from_mat(variable_names):
    #Read the jittered triggers from a .mat file 
    root = tk.Tk()
    root.withdraw()  # hide root window
    filename = filedialog.askopenfilename(
        title='Select data file',
        filetypes=(('MATLAB files', '*.mat'), ('All files', '*.*'))
    )
    if not filename:
        raise ValueError("No file selected.")

    data = loadmat(filename, variable_names=variable_names)
    
    return data

def reconstruct_triggers(ref_trig, rec_trig, fs_ref=48000, fs_rec=1000):
    """
    Reconstruct full-length trigger array in recorded time base.

    Parameters
    ----------
    ref_trig : array-like
        Reference triggers (samples at fs_ref)
    rec_trig : array-like
        Recorded triggers (samples at fs_rec)
    fs_ref : int
        Sampling rate of reference triggers
    fs_rec : int
        Sampling rate of recorded triggers
    """
    ref_trig = np.array(ref_trig).squeeze().astype(float)
    rec_trig = np.array(rec_trig).squeeze().astype(float)

    n_ref = len(ref_trig)
    n_rec = len(rec_trig)
    rec_trig_full = np.zeros(n_ref, dtype=float)

    # Determine step: how often recorded triggers appear
    step = round(n_ref / n_rec)

    # Place recorded triggers at every 'step' position
    anchor_positions = np.arange(0, n_ref, step)
    anchor_positions = anchor_positions[:n_rec]  # make sure lengths match
    rec_trig_full[anchor_positions] = rec_trig

    # Fill in missing triggers iteratively
    last_anchor_idx = anchor_positions[0]
    last_anchor_val = rec_trig_full[last_anchor_idx]

    for i in range(last_anchor_idx + 1, n_ref):
        if i in anchor_positions:
            # Update anchor
            last_anchor_val = rec_trig_full[i]
            last_anchor_idx = i
        else:
            # Fill using sample difference from reference
            delta_ref = ref_trig[i] - ref_trig[i - 1]
            rec_trig_full[i] = last_anchor_val + delta_ref * fs_rec / fs_ref
            last_anchor_val = rec_trig_full[i]

    return rec_trig_full.reshape(-1, 1)


def shuffle_events():
    # This carries the logic to shuffle events but needs to be generalized
    # --- Recreate the exact structure you build ---
    chunk = np.concatenate([
        np.tile(np.arange(1, 5), 250),   # A: 1..10 repeated 100 times  -> length 1000
        np.tile(np.arange(11, 15), 250),  # B: 11..20 repeated 100 times -> length 1000
        np.tile(np.arange(21, 25), 250),  # C: 21..30 repeated 100 times -> length 1000
    ])
    event_order = np.tile(chunk, 4)      # repeat the chunk 10× -> total length 30000
    event_order = event_order.copy()      # safe copy

    # --- Masks for group positions (these are the positions that are allowed for each group) ---
    A_mask = (event_order >= 1)  & (event_order <= 10)
    B_mask = (event_order >= 11) & (event_order <= 20)
    C_mask = (event_order >= 21) & (event_order <= 30)

    # Sanity: counts of positions per group
    assert A_mask.sum() == 4 * 1000  # 10 values * 1000 each = 10000
    assert B_mask.sum() == 4 * 1000
    assert C_mask.sum() == 4 * 1000

    # --- Create the values to place into each group's positions (1000 of each value) ---
    vals_A = np.repeat(np.arange(1, 5), 1000)   # 1..10 each repeated 1000 times -> len 10000
    vals_B = np.repeat(np.arange(11, 15), 1000)  # 11..20 -> len 10000
    vals_C = np.repeat(np.arange(21, 25), 1000)  # 21..30 -> len 10000

    # --- Shuffle each group's values and assign them into the group's available positions ---
    rng = np.random.default_rng()  # seed optional for reproducibility
    rng.shuffle(vals_A)
    rng.shuffle(vals_B)
    rng.shuffle(vals_C)

    shuffled = event_order.copy()
    shuffled[A_mask] = vals_A
    shuffled[B_mask] = vals_B
    shuffled[C_mask] = vals_C

    # --- Quick checks ---
    unique, counts = np.unique(shuffled, return_counts=True)
    print("Counts per value (should be 1000 each):")
    print(dict(zip(unique, counts)))

    # show distribution of a sample value across chunks (to prove per-chunk counts are NOT preserved)
    chunk_len = len(chunk)               # 3000
    per_chunk_counts_for_1 = [np.sum(shuffled[i*chunk_len:(i+1)*chunk_len] == 1) for i in range(10)]
    print("Per-chunk counts for value 1 (10 chunks, sum -> 1000):")
    print(per_chunk_counts_for_1, "sum=", sum(per_chunk_counts_for_1))

    print(len(shuffled))  # Check length
    print(shuffled[:20])  # Preview first 20 labels
    return shuffled

def create_rnd_trigs(trig_points, block_size):
    # creates random trigers within a stimuli block, for significance comparisons


    trig_points = np.asarray(trig_points)
    n_triggers = len(trig_points)
    min_idx = np.min(trig_points)
    max_idx = np.max(trig_points)

    # Determine how many blocks
    n_blocks = int(np.ceil((max_idx - min_idx + 1) / block_size))

    random_triggers = []

    for b in range(n_blocks):
        # Define block boundaries
        block_start = min_idx + b * block_size
        block_end = min(block_start + block_size, max_idx + 1)

        # Get triggers within this block
        block_trigs = trig_points[(trig_points >= block_start) & (trig_points < block_end)]
        n_block_trigs = len(block_trigs)

        if n_block_trigs > 0:
            # Generate random triggers in this block, unique and uniform
            rand_trigs = np.random.choice(
                np.arange(block_start, block_end),
                size=n_block_trigs,
                replace=True
            )
            random_triggers.extend(rand_trigs)

    random_triggers = np.sort(random_triggers)
    return np.array(random_triggers)


# Plotting functions


def Spectrum_significance(raws,idx,events, titles,t_min, t_max, channel, baseline=None,reject=None):
    evokeds = []
    wa_list = []
    spectra = [] # Store spectra for each evoked
    freq_axes = []
    channels = channel
    N_realizations = 100
    events= events[0]
    idx = idx[0]



    n_pad = 124
    plot_fmax = 120
    tukey_alpha = 0.4  # Tukey taper fraction
    epoch_sig = mne.Epochs(raws[0], events, event_id=idx,
                            tmin=t_min, tmax=t_max,
                            baseline=baseline, reject=reject,picks=channels)
    epoch_sig.load_data().crop(tmin=t_min, tmax=t_max)  
    wa_sig = variance_weighted_average(epoch_sig)
    wa_sig = wa_sig.copy().shift_time(t_min, relative=False)

    wave = wa_sig.get_data(picks=channels[0]).squeeze()
    tuk_win = tukey(100, alpha=tukey_alpha)
    tuk_win = np.hstack((tuk_win, np.zeros((100))))
    fft_tukey_sig = np.fft.rfft(wave[0:len(tuk_win)] * tuk_win, n=len(tuk_win))
    psd_tukey_sig = np.abs(fft_tukey_sig)
    freqs_tuk_sig = np.fft.rfftfreq(len(tuk_win), 1 / wa_sig.info['sfreq'])


    for i in range(N_realizations):
        channels = channel
        
        events_rnd = copy.deepcopy(events)


        ii = np.argwhere(np.isin(events[:, 2], idx))

        for j in ii:
            events_rnd[j,0] = events[j,0] + np.random.randint(-200,200)

        values, counts = np.unique(events_rnd[:,0], return_counts=True)
        repeated_values = values[counts > 1]
        repeated_counts = counts[counts > 1]
        print(repeated_values, repeated_counts)
        
        while len(values) != 83200:

            for j in repeated_values:
                x = np.argwhere(events_rnd[:,0]==j)
                for n in range(len(x)):
                    events_rnd[x[n],0] += n*3

            values, counts = np.unique(events_rnd[:,0], return_counts=True)
            repeated_values = values[counts > 1]

        epoch = mne.Epochs(raws[0], events_rnd, event_id=idx,
                            tmin=t_min, tmax=t_max,
                            baseline=baseline, reject=reject,picks=channels)
        
        epoch.load_data().crop(tmin=t_min, tmax=t_max)

        

        # Check fallback channel
        if not all(ch in epoch.ch_names for ch in channels):
            channel = ["FpzT", "FpzB"]
            print("Fallback channels:", channel)

        # Weighted average for time-domain storage
        wa = variance_weighted_average(epoch).pick(channel)
        wa = wa.copy().shift_time(t_min, relative=False)
        evokeds.append(wa)
        wa_list.append(wa)

        # Crop and get data for spectral analysis
        x = wa.copy()
        wave = x.get_data(picks=channel[0]).squeeze()
        #wave = wave - np.mean(wave)

        # --- FFT with Tukey window ---
        tuk_win = tukey(100, alpha=tukey_alpha)
        tuk_win = np.hstack((tuk_win, np.zeros((100))))
        fft_tukey = np.fft.rfft(wave[0:len(tuk_win)] * tuk_win, n=len(tuk_win))
        psd_tukey = np.abs(fft_tukey)
        freqs_tuk = np.fft.rfftfreq(len(tuk_win), 1 / x.info['sfreq'])

        # Store spectrum
        spectra.append(psd_tukey)
        freq_axes.append(freqs_tuk)

        
    # Convert list of randomized spectra to an array for easier comparison
    # Shape: (N_realizations, n_freqs)
    rand_spectra = np.array(spectra)

    # Ensure the randomized spectra have the same length as psd_tukey_sig
    min_len = min(psd_tukey_sig.shape[0], rand_spectra.shape[1])
    psd_tukey_sig_crop = psd_tukey_sig[:min_len]
    rand_spectra_crop = rand_spectra[:, :min_len]

    # Perform a t-test at each frequency point
    t_vals, p_vals = stats.ttest_1samp(rand_spectra_crop, psd_tukey_sig_crop, axis=0)
    

    # t_vals -> t-statistic for each frequency
    # p_vals -> p-value for each frequency
    # Crop to the same length if needed
    min_len = min(psd_tukey_sig.shape[0], rand_spectra.shape[1])
    psd_tukey_sig_crop = psd_tukey_sig[:min_len]
    freqs_crop = freqs_tuk_sig[:min_len]
    p_vals_crop = p_vals[:min_len]

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,6), sharex=True)

    # Top: PSD spectrum
    ax1.plot(freqs_crop, psd_tukey_sig_crop, color='b', label='Original PSD')
    ax1.plot(freqs_crop,rand_spectra_crop[1,:],color = 'r', label='Noise spectrum')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Weighted-Average Spectrum')
    ax1.legend()
    ax1.grid(True)

    # Bottom: p-values
    # Bottom: p-values
    print(p_vals_crop)
    ax2.plot(freqs_crop, p_vals_crop, label='-log10(p-value)')
    ax2.axhline(.05, linestyle='--', label='p=0.05 threshold')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('-log10(p-value)')
    ax2.set_title('Significance vs Randomized Spectra')
    ax2.legend()
    ax2.grid(True)

    # Set frequency range
    ax2.set_xlim(0, 120)

    plt.tight_layout()
    plt.show()

    
    return rand_spectra_crop, psd_tukey_sig_crop, freqs_crop


def spectrum_significance(raws, idx, events, titles, t_min, t_max, channel, baseline=None, reject=None, N_realizations=20):
    """
    Compute weighted-average (WA) spectrum and generate noise/null spectra
    by randomly flipping the polarity of half the epochs for each realization.
    
    Parameters
    ----------
    raws : list of Raw objects
    idx : list of event IDs (may contain multiple IDs)
    events : list of event arrays
    titles : list of titles (not used in this function)
    t_min, t_max : float, epoch window
    channel : list of channel names
    baseline : tuple or None
    reject : dict or None
    N_realizations : int, number of noise realizations
    
    Returns
    -------
    psd_sig : array, spectrum of the original WA
    freqs_sig : array, frequency axis
    noise_spectra : array, shape (N_realizations, n_freqs), null spectra
    """
    events = events[0]
    idx = idx[0]

    tukey_alpha = 0.4

    # Epoch and compute true signal WA
    epoch_sig = mne.Epochs(raws[0], events, event_id=idx,
                           tmin=t_min, tmax=t_max,
                           baseline=baseline, reject=reject, picks=channel)
    epoch_sig.load_data().crop(tmin=t_min, tmax=t_max)
    
    wa_sig = variance_weighted_average(epoch_sig).copy().shift_time(t_min, relative=False)
    wave_sig = wa_sig.get_data(picks=channel[0]).squeeze()

    # Prepare Tukey window
    tuk_win = np.hstack((tukey(100, alpha=tukey_alpha), np.zeros(100)))
    fft_sig = np.fft.rfft(wave_sig[0:len(tuk_win)] * tuk_win, n=len(tuk_win))
    psd_sig = np.abs(fft_sig)
    freqs_sig = np.fft.rfftfreq(len(tuk_win), 1 / wa_sig.info['sfreq'])

    # Store original epoch data for in-place manipulation
    epoch_data_orig = epoch_sig.get_data()
    n_epochs = epoch_data_orig.shape[0]

    noise_spectra = []

    for i in range(N_realizations):
        print(f"Noise realization {i+1}/{N_realizations}")

        # Generate a new ±1 vector for this iteration
        signs = np.array([1]*(n_epochs//2) + [-1]*(n_epochs - n_epochs//2))
        np.random.shuffle(signs)

        # Backup original data
        epoch_data_backup = epoch_sig._data.copy()

        # Apply ±1 signs in-place
        epoch_sig._data = epoch_data_orig * signs[:, None, None]

        # Compute WA of flipped epochs
        wa_noise = variance_weighted_average(epoch_sig).copy().shift_time(t_min, relative=False)

        # Compute FFT
        wave_noise = wa_noise.get_data(picks=channel[0]).squeeze()
        fft_noise = np.fft.rfft(wave_noise[0:len(tuk_win)] * tuk_win, n=len(tuk_win))
        psd_noise = np.abs(fft_noise)
        noise_spectra.append(psd_noise)

        # Restore original data
        epoch_sig._data = epoch_data_backup

    noise_spectra = np.array(noise_spectra)

    return psd_sig, freqs_sig, noise_spectra


def plot_spectrum_sig(noise_spectra, true_spectrum, freqs, alpha=0.05):
    """
    Plots the true spectrum with noise distribution and highlights frequencies
    that are significant after FDR correction.

    Parameters
    ----------
    noise_spectra : array, shape (N_realizations, n_freqs)
        Spectra estimated from noise.
    true_spectrum : array, shape (n_freqs,)
        The spectrum to test.
    freqs : array, shape (n_freqs,)
        Frequency axis.
    alpha : float
        Significance level for FDR correction (default 0.05).
    """

    # Compute empirical p-values at each frequency
    # One-sided test: true_spectrum > noise
   # Crop to 0–120 Hz
    freq_cutoff_idx = 24  # corresponds to 120 Hz
    true_spectrum_crop = true_spectrum[:freq_cutoff_idx + 1]        # include index 24
    noise_spectra_crop = noise_spectra[:, :freq_cutoff_idx + 1]
    freqs_crop = freqs[:freq_cutoff_idx + 1]
    # Calculate p-values only for these frequencies
    p_vals = np.mean(noise_spectra_crop >= true_spectrum_crop[None, :], axis=0)

    # Apply FDR correction
    reject, p_fdr = fdrcorrection(p_vals, alpha=alpha)

    # Compute noise thresholds for plotting
    lower_05 = np.percentile(noise_spectra_crop, 5, axis=0)
    upper_05 = np.percentile(noise_spectra_crop, 95, axis=0)
    print(p_fdr)
    print(p_vals)
    plt.figure(figsize=(10, 4))

    # Noise null band
    plt.fill_between(
        freqs_crop,
        lower_05,
        upper_05,
        alpha=0.3,
        label='Noise 5–95% range'
    )

    # Noise threshold (optional visual)
    plt.plot(
        freqs_crop,
        upper_05,
        linestyle='--',
        linewidth=1.5,
        label='Noise p = 0.05 threshold'
    )

    # True spectrum
    plt.plot(
        freqs_crop,
        true_spectrum_crop,
        linewidth=2,
        label='True spectrum'
    )

    # Highlight significant frequencies after FDR correction
    plt.scatter(
        freqs_crop[reject],
        true_spectrum_crop[reject],
        s=20,
        color='red',
        zorder=3,
        label=f'Significant (FDR p < {alpha})'
    )

    plt.xlim(0, 120)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Spectrum vs Noise Distribution (FDR-corrected),40 Hz')
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    

    plt.tight_layout()
    plt.show()

def ITPC(raws, idx, events, t_min, t_max, channel,
         baseline=None, reject=None,
         tukey_alpha=0.4, n_win=100,
         N_realizations=100, jitter=200):


    events = events[0]
    idx = idx[0]
    channels = channel

    # ---------- Helper: compute ITPC from events ----------
     # ---------- Helper: compute ITPC from events ----------
    def _compute_itpc(events_in):

        epochs = mne.Epochs(
            raws[0], events_in, event_id=idx,
            tmin=t_min, tmax=t_max,
            baseline=baseline, reject=reject,
            picks=channels
        ).load_data()

        # shape: (n_trials, n_times)
        data = epochs.get_data(picks=channels[0])[:, 0, :]
        sfreq = epochs.info['sfreq']

        tuk_win = tukey(n_win, alpha=tukey_alpha)
        tuk_win = np.hstack((tuk_win, np.zeros(n_win)))
        n_fft = len(tuk_win)

        # FFT
        fft_epochs = np.fft.rfft(
            data[:, :n_fft] * tuk_win,
            n=n_fft,
            axis=1
        )

        # ITPC
        fft_mean = np.mean(fft_epochs, axis=0)
        itpc = np.abs(fft_mean) / np.mean(np.abs(fft_epochs), axis=0)
        mean_phase = np.angle(fft_mean)
        freqs = np.fft.rfftfreq(n_fft, 1 / sfreq)

        # ----------------- Cosine similarity -----------------
        # Compute cosine similarity across trials for each frequency
        # We'll use the complex vectors as 2D real vectors [real, imag]
        n_freqs = fft_epochs.shape[1]
        cs_vector = np.zeros(n_freqs)

        for f in range(n_freqs):
            # shape: (n_trials, 2) -> [Re, Im]
            vecs = np.vstack((fft_epochs[:, f].real, fft_epochs[:, f].imag)).T
            # Compute mean vector
            mean_vec = np.mean(vecs, axis=0)
            # Compute cosine similarity of each trial to mean vector
            cos_sims = []
            for v in vecs:
                norm_v = np.linalg.norm(v)
                norm_m = np.linalg.norm(mean_vec)
                if norm_v == 0 or norm_m == 0:
                    cos_sims.append(0)
                else:
                    cos_sims.append(np.dot(v, mean_vec) / (norm_v * norm_m))
            cs_vector[f] = np.mean(cos_sims)

        return itpc, mean_phase, freqs, cs_vector
    
    # ---------- True ITPC ----------
    itpc_true, phase_true, freqs, cs_true = _compute_itpc(events)

    # ---------- Null distribution via trigger shuffling ----------
    itpc_null = []
    cs_null = []

    for _ in range(N_realizations):

        events_rnd = copy.deepcopy(events)

        ii = np.where(np.isin(events[:, 2], idx))[0]
        for j in ii:
            events_rnd[j, 0] += np.random.randint(-jitter, jitter)

        # (Optional but recommended: ensure unique sample indices)
        values, counts = np.unique(events_rnd[:, 0], return_counts=True)
        while np.any(counts > 1):
            for v in values[counts > 1]:
                dup = np.where(events_rnd[:, 0] == v)[0]
                for k in range(len(dup)):
                    events_rnd[dup[k], 0] += k + 1
            values, counts = np.unique(events_rnd[:, 0], return_counts=True)

        itpc_rnd, _, _ , cs_vector_null= _compute_itpc(events_rnd)
        itpc_null.append(itpc_rnd)
        cs_null.append(cs_vector_null)

    itpc_null = np.array(itpc_null)  # (N_realizations, n_freqs)

    return itpc_true, phase_true, itpc_null, freqs, cs_true, cs_null

def make_title(event_ids, event_id_map):
    if isinstance(event_ids, (list, tuple)):
        titles = [event_id_map[eid] for eid in event_ids]
        return f"avg({', '.join(titles)})"
    else:
        return event_id_map[event_ids]

def plot_evokeds_1(raws, idx, events,event_id, t_min, t_max, channel,
                   baseline=None, reject=None):

    evokeds = {}
    wa_list = []
    spectra = {}
    freq_axes = {}
    channels = channel

    plot_fmax = 170
    tukey_alpha = 0.4  # Tukey taper fraction

    titles = []
    # ---- main loop ----
    for i, event, new_raw in zip(idx, events, raws):

        # create title from annotation
        title = make_title(i, event_id)
        titles.append(title)
        epoch = mne.Epochs(
            new_raw,
            event,
            event_id=i,
            tmin=t_min,
            tmax=t_max,
            baseline=baseline,
            reject=reject,
            preload=True
        )

        epoch.crop(tmin=t_min, tmax=t_max)
        # Check fallback channel
        if not all(ch in epoch.ch_names for ch in channels):
            channel = ["FpzT", "FpzB"]
            print("Fallback channels:", channel)

        # Weighted average for time-domain storage
        wa = variance_weighted_average(epoch).pick(channel)
        wa = wa.copy().shift_time(t_min, relative=False)
        evokeds[title] = wa
        wa_list.append(wa)

        # Crop and get data for spectral analysis
        x = wa.copy()
        wave = x.get_data(picks=channel[0]).squeeze()
        #wave = wave - np.mean(wave)

        # --- FFT with Tukey window ---
        tuk_win = tukey(100, alpha=tukey_alpha)
        tuk_win = np.hstack((tuk_win, np.zeros((100))))
        fft_tukey = np.fft.rfft(wave[0:len(tuk_win)] * tuk_win, n=len(tuk_win))
        psd_tukey = np.abs(fft_tukey)
        freqs_tuk = np.fft.rfftfreq(len(tuk_win), 1 / x.info['sfreq'])

        # Store spectrum
        spectra[title] = psd_tukey
        freq_axes[title] = freqs_tuk

    # --- PLOT ALL ESTIMATES TOGETHER ---
    plt.figure(figsize=(10, 6))
    for title in titles:
        plt.plot(freq_axes[title], spectra[title], label=title)
    
    plt.xlim(1, plot_fmax)
    plt.yscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title(f"Spectral Estimates ({t_max*1000:.0f} ms window)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    if "random trigger" not in spectra:
        print("Warning: 'random trigger' not found. Skipping SNR plot.")
    else:
        plt.figure(figsize=(10, 6))
        for title in titles:
            plt.plot(freq_axes[title], 20*np.log10(spectra[title]/spectra["random trigger"]), label=title)
        
        plt.xlim(1, plot_fmax)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title(f"Spectral SNR ({t_max*1000:.0f} ms window)")
        plt.legend()
        plt.tight_layout()
        plt.show()

   
    
    # Now compute differences relative to last evoked
    #difs = {
    #    titles[i]: mne.combine_evoked([wa_list[i], wa_list[-1]], weights=[1, -1])
    #    for i in range(len(wa_list) - 1)
    #}
    rename_map = {
    "FpzT": "AFz",
    "FpzB": "Fpz"
    }

    # Apply to all evoked objects
    for key in evokeds:
        evokeds[key] = evokeds[key].copy()  # avoid modifying original
        # only rename channels that exist
        valid_renames = {old: new for old, new in rename_map.items() if old in evokeds[key].ch_names}
        evokeds[key].rename_channels(valid_renames)
    
    # Generate a color map with enough colors
    colors = plt.cm.tab20.colors  # 20 distinct colors

    out = mne.viz.plot_compare_evokeds(
    evokeds,
    picks=channel, 
    combine='mean', # or multiple picks if you like
    title="Evoked Responses: Different ACCs",
    show=False,
    colors=colors
    )

    # If only one figure is returned
    if hasattr(out, "axes"):  # single figure
        figs = [out]
    else:
        figs = out  # it's already a list of figures

    # Modify all figures
    """for fig in figs:
        ax = fig.axes[0]
        ax.axvline(x=0.08,c=colors[1],ls='--')
        ax.axvline(x=0.08*2,c=colors[1],ls='--')
        ax.axvline(x=0.08*3,c=colors[1],ls='--')
        ax.axvline(x=0.1,c=colors[0],ls='--')
        ax.axvline(x=0.1*2,c=colors[0],ls='--')
        ax.axvline(x=0.1*3,c=colors[0],ls='--')
        ax.legend(loc='lower right')
    """
    plt.show()
   
    return wa_list,spectra

def compare_two_with_shifted_sum(idx,events,titles,raws,channel,t_min,t_max,baseline,reject,offset_ms,):
            """
            Compare exactly two conditions and an artificial evoked:
            idx[1] + time-shifted(idx[1])

            Parameters
            ----------
            idx : list or array, length == 2
            offset_ms : float
                Time shift applied to idx[1] in milliseconds
            """

            assert len(idx) == 2, "idx must contain exactly two entries"

            evokeds = {}
            wa_list = []
            spectra = {}
            freq_axes = {}

            plot_fmax = 120
            tukey_alpha = 0.4

            channels = channel

            # =====================
            # Compute real evokeds
            # =====================
            for i, event, title, new_raw in zip(idx, events, titles, raws):
                channel = channels

                epoch = mne.Epochs(
                    new_raw,
                    event,
                    event_id=i,
                    tmin=t_min,
                    tmax=t_max,
                    baseline=baseline,
                    reject=reject,
                    preload=True,
                ).crop(tmin=t_min, tmax=t_max)

                # Fallback channels
                if not all(ch in epoch.ch_names for ch in channel):
                    channel = ["FpzT", "FpzB"]
                    print("Fallback channels:", channel)

                wa = variance_weighted_average(epoch).pick(channel)
                wa = wa.copy().shift_time(t_min, relative=False)

                evokeds[title] = wa
                wa_list.append(wa)

                # ---- Spectral analysis ----
                x = wa.copy()
                wave = x.get_data(picks=channel[0]).squeeze()

                tuk_win = tukey(100, alpha=tukey_alpha)
                tuk_win = np.hstack((tuk_win, np.zeros(100)))

                fft_tukey = np.fft.rfft(
                    wave[: len(tuk_win)] * tuk_win,
                    n=len(tuk_win),
                )
                psd_tukey = np.abs(fft_tukey)
                freqs_tuk = np.fft.rfftfreq(
                    len(tuk_win), 1 / x.info["sfreq"]
                )

                spectra[title] = psd_tukey
                freq_axes[title] = freqs_tuk

            # ==========================
            # Create shifted-sum evoked
            # ==========================
            base_evoked = wa_list[1]
            sfreq = base_evoked.info["sfreq"]

            # ms → samples
            offset_samp = int(np.round(offset_ms / 1000 * sfreq))

            data = base_evoked.data
            n_ch, n_t = data.shape

            shifted_data = np.zeros_like(data)

            if offset_samp > 0:
                # Shift forward, drop tail
                shifted_data[:, offset_samp:] = data[:, : n_t - offset_samp]

            elif offset_samp < 0:
                # Shift backward, drop head
                shifted_data[:, : n_t + offset_samp] = data[:, -offset_samp:]

            # else: offset_samp == 0 → all zeros already wrong, so copy
            else:
                shifted_data = data.copy()

            shifted = base_evoked.copy()
            shifted.data = shifted_data
            

            summed = mne.combine_evoked(
                [base_evoked, shifted],
                weights=[1, 1])

            sum_title = f"{titles[1]} + shifted({offset_ms:.0f} ms)"
            evokeds[sum_title] = summed
            wa_list.append(summed)

            # ---- Spectrum of summed evoked ----
            x = summed.copy()
            wave = x.get_data(picks=channel[0]).squeeze()

            tuk_win = tukey(100, alpha=tukey_alpha)
            tuk_win = np.hstack((tuk_win, np.zeros(100)))

            fft_tukey = np.fft.rfft(
                wave[: len(tuk_win)] * tuk_win,
                n=len(tuk_win),
            )
            spectra[sum_title] = np.abs(fft_tukey)
            freq_axes[sum_title] = np.fft.rfftfreq(
                len(tuk_win), 1 / sfreq
            )

            # =====================
            # Plot spectra
            # =====================
            plt.figure(figsize=(10, 6))
            for key in spectra:
                plt.plot(freq_axes[key], spectra[key], label=key)

            plt.xlim(1, plot_fmax)
            plt.yscale("log")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Power")
            plt.title(f"Spectral Estimates ({t_max*1000:.0f} ms window)")
            plt.legend()
            plt.tight_layout()
            plt.show()

            # =====================
            # Rename channels
            # =====================
            rename_map = {"FpzT": "AFz", "FpzB": "Fpz"}

            for key in evokeds:
                valid = {
                    old: new
                    for old, new in rename_map.items()
                    if old in evokeds[key].ch_names
                }
                evokeds[key] = evokeds[key].copy().rename_channels(valid)

            # =====================
            # Plot evoked potentials
            # =====================
            colors = plt.cm.tab20.colors

            out = mne.viz.plot_compare_evokeds(
                evokeds,
                picks=channel,
                combine="mean",
                title="Evoked Responses: Two Conditions + Shifted Sum",
                colors=colors,
                show=False,
            )

            plt.show()

            return wa_list, spectra




def plot_evokeds_2(raws, idx, events, titles, t_min, t_max, channel,
                    baseline=None, reject=None):
    """
    Computes weighted-average evoked and per-trial Blackman-Tukey PSD
    for each condition.

    Parameters
    ----------
    raws : list of Raw objects
    idx : list of int
        Event IDs per condition
    events : list of ndarray
        Each element is the event array for the corresponding condition
    titles : list of str
        Condition names
    t_min, t_max : float
        Epoch time window
    channel : list of str
        Channels to use
    baseline : tuple or None
        Baseline correction
    reject : dict or None
        Epoch rejection criteria

    Returns
    -------
    evokeds : dict
        Weighted-average evoked per condition
    spectra : dict
        Mean PSD per condition
    freq_axes : dict
        Frequency axis per condition
    """

    evokeds = {}
    spectra = {}
    freq_axes = {}
    plot_fmax = 170
    tukey_alpha = 0.4
    max_lag = 150  # lag window length (samples), can tune

    for i, event, title, new_raw in zip(idx, events, titles, raws):
        # Epoch data
        epoch = mne.Epochs(new_raw, event, event_id=i,
                           tmin=t_min, tmax=t_max,
                           baseline=baseline, reject=reject,
                           picks=channel)
        epoch.load_data().crop(tmin=0, tmax=t_max)
        # Fallback channel
        channel_use = channel
        if not all(ch in epoch.ch_names for ch in channel_use):
            channel_use = ["FpzT", "FpzB"]
            print("Fallback channels:", channel_use)

        # ----- TIME DOMAIN: weighted average -----
        wa = variance_weighted_average(epoch).pick(channel_use)
        wa = wa.copy().shift_time(t_min, relative=False)
        evokeds[title] = wa
        # ----- PER-TRIAL BLACKMAN-TUKEY PSD -----
        data = epoch.get_data(picks=channel_use[0])  # (n_trials, n_times)
        sfreq = epoch.info['sfreq']
        data = data[:, 0, :] 
        print(data.shape)
        n_trials, n_times = data.shape

        print(n_trials)

        psd_trials = []

        # Lag window (Tukey in lag domain)
        lag_win = tukey(4/3 * max_lag + 1, alpha=tukey_alpha)

        for k in range(n_trials):
            x = data[k, :]
            x = x - np.mean(x)

            # autocorrelation (biased)
            rxx_full = np.correlate(x, x, mode='full') / n_times
            mid = len(rxx_full) // 2
            rxx = rxx_full[mid - max_lag: mid + max_lag + 1]

            # apply lag window
            rxx_win = rxx * lag_win

            # FFT -> PSD
            psd = np.real(np.fft.rfft(rxx_win, n=2 * max_lag + 1))
            psd_trials.append(psd)

        psd_trials = np.array(psd_trials)
        psd_mean = psd_trials.mean(axis=0)

        freqs = np.fft.rfftfreq(2 * max_lag + 1, d=1 / sfreq)

        spectra[title] = psd_mean
        freq_axes[title] = freqs

    # --- PLOT ALL ESTIMATES TOGETHER ---
    plt.figure(figsize=(10, 6))
    for title in titles:
        plt.plot(freq_axes[title], spectra[title], label=title)
    
    plt.xlim(1, plot_fmax)
    plt.yscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title(f"Spectral Estimates ({t_max*1000:.0f} ms window)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    if "random trigger" not in spectra:
        print("Warning: 'random trigger' not found. Skipping SNR plot.")
    else:
        plt.figure(figsize=(10, 6))
        for title in titles:
            plt.plot(freq_axes[title], 20*np.log10(spectra[title]/spectra["random trigger"]), label=title)
        
        plt.xlim(1, plot_fmax)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title(f"Spectral SNR ({t_max*1000:.0f} ms window)")
        plt.legend()
        plt.tight_layout()
        plt.show()

   
    
    # Now compute differences relative to last evoked
    #difs = {
    #    titles[i]: mne.combine_evoked([wa_list[i], wa_list[-1]], weights=[1, -1])
    #    for i in range(len(wa_list) - 1)
    #}
    rename_map = {
    "FpzT": "AFz",
    "FpzB": "Fpz"
    }

    # Apply to all evoked objects
    for key in evokeds:
        evokeds[key] = evokeds[key].copy()  # avoid modifying original
        # only rename channels that exist
        valid_renames = {old: new for old, new in rename_map.items() if old in evokeds[key].ch_names}
        evokeds[key].rename_channels(valid_renames)
    
    # Generate a color map with enough colors
    colors = plt.cm.tab20.colors  # 20 distinct colors

    out = mne.viz.plot_compare_evokeds(
    evokeds,
    picks=channel, 
    combine='mean', # or multiple picks if you like
    title="Evoked Responses: Different ACCs",
    show=False,
    colors=colors
    )

    # If only one figure is returned
    if hasattr(out, "axes"):  # single figure
        figs = [out]
    else:
        figs = out  # it's already a list of figures

    # Modify all figures
    """for fig in figs:
        ax = fig.axes[0]
        ax.axvline(x=0.08,c=colors[1],ls='--')
        ax.axvline(x=0.08*2,c=colors[1],ls='--')
        ax.axvline(x=0.08*3,c=colors[1],ls='--')
        ax.axvline(x=0.1,c=colors[0],ls='--')
        ax.axvline(x=0.1*2,c=colors[0],ls='--')
        ax.axvline(x=0.1*3,c=colors[0],ls='--')
        ax.legend(loc='lower right')
    """
    plt.show()
   
    return wa_list,spectra

def MLR_peak_delays(wa_list, channel=None):
    """
    Detect Na, Pb, Nb using derivative zero-crossings (corrected for np.diff shift):
    - Na: first negative peak zero-crossing (~20-40 ms)
    - Pb, Nb: next two zero-crossings after Na

    Parameters
    ----------
    wa_list : list of mne.EvokedArray
        List of evoked responses.
    channel : str or int, optional
        Channel to use. Defaults to first channel.

    Returns
    -------
    latencies : np.ndarray
        Nx3 array of Na, Pb, Nb peak times (s)
    """
    latencies = []

    for evoked in wa_list:
        # Pick channel
        if channel is None:
            data = evoked.get_data(picks=0).squeeze()
        else:
            data = evoked.get_data(picks=channel).squeeze()
        
        times = evoked.times
        deriv = np.diff(data)

        # Zero-crossings in derivative
        zero_crossings = np.where(np.diff(np.sign(deriv)))[0] + 1  # <-- add 1 sample correction

        # Select Na: first negative peak in 20-40 ms
        window = (times[zero_crossings] >= 0.018) & (times[zero_crossings] <= 0.03)
        if np.any(window):
            Na_idx = zero_crossings[window][np.argmin(data[zero_crossings[window]])]
            Na_lat = times[Na_idx]
        else:
            Na_lat = np.nan
            Na_idx = None

        # Pb and Nb: next two zero-crossings after Na
        if Na_idx is not None:
            post_Na = zero_crossings[zero_crossings > Na_idx]
            if len(post_Na) >= 2:
                Pb_lat = times[post_Na[0]]
                Nb_lat = times[post_Na[1]]
            elif len(post_Na) == 1:
                Pb_lat = times[post_Na[0]]
                Nb_lat = np.nan
            else:
                Pb_lat = Nb_lat = np.nan
        else:
            Pb_lat = Nb_lat = np.nan

        latencies.append([Na_lat, Pb_lat, Nb_lat])

    return np.array(latencies)

def evoked_peak_delays(wa_list, latencies, channel=None):
    """
    Plot EvokedArray signals with vertical lines at Na, Pb, Nb peaks,
    using the same color as the evoked trace.

    Parameters
    ----------
    wa_list : list of mne.EvokedArray
        Evoked responses.
    latencies : np.ndarray
        Nx3 array of Na, Pb, Nb peak times (s)
    channel : str or int, optional
        Channel to plot. Defaults to first channel.
    """
    plt.figure(figsize=(10,6))

    # Generate a color cycle from matplotlib
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, evoked in enumerate(wa_list):
        color = color_cycle[i % len(color_cycle)]  # wrap around if more evokeds than colors

        if channel is None:
            data = evoked.get_data(picks=0).squeeze()
            ch_name = evoked.ch_names[0]
        else:
            data = evoked.get_data(picks=channel).squeeze()
            ch_name = channel

        times = evoked.times
        plt.plot(times, data, label=f'Evoked {i}', color=color)

        # Plot Na/Pb/Nb vertical lines with same color
        for peak_lat in latencies[i]:
            if not np.isnan(peak_lat):
                plt.axvline(x=peak_lat, color=color, linestyle='--')

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.title(f'Evoked Responses ({ch_name}) with Na, Pb, Nb Peaks')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_latency_vs_stimulus(x_vals, latencies):
    """
    Generate two scatter plots:
    1) Na, Pb, Nb latencies vs. 1/stimulus
    2) Inter-peak delays (Na-Pb, Pb-Nb, Na-Nb) vs. 1/stimulus
    
    Parameters
    ----------
    x_vals : np.ndarray
        1/stimulus values.
    latencies : np.ndarray
        Nx3 array of Na, Pb, Nb latencies (s).
    """
    # --- Plot 1: Absolute latencies ---
    plt.figure(figsize=(8,5))
    labels = ['Na', 'Pa', 'Nb']
    markers = ['o', 's', '^']
    colors = ['r', 'g', 'b']

    for i in range(3):
        plt.scatter(x_vals, latencies[:, i], label=labels[i], color=colors[i], marker=markers[i], s=80)

    plt.xlabel('Frequency change duration (ms)')
    plt.ylabel('Latency (s)')
    plt.title('MLR peak Latencies vs duration of frequency change')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Plot 2: Inter-peak delays ---
    plt.figure(figsize=(8,5))
    delays = [
        latencies[:, 0] - latencies[:, 0],  # Na-Pb
        latencies[:, 1] - latencies[:, 0],  # Pb-Nb
        latencies[:, 2] - latencies[:, 0]   # Na-Nb
    ]
    delay_labels = ['Na-Na', 'Pa-Na', 'Nb-Na']

    for delay, label, marker, color in zip(delays, delay_labels, markers, colors):
        plt.scatter(x_vals, delay, label=label, marker=marker, color=color, s=80)

    plt.xlabel('Frequency change duration (ms)')
    plt.ylabel('Inter-Peak Delay (s)')
    plt.title('Inter-Peak Delays vs duration of frequency change')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_spectra(raws, idx, events, titles, t_min, t_max, channels,
                 baseline=None, reject=None):
    """
    Compute and plot several spectral estimates of AEP/MLR responses:
    1. Direct FFT
    2. Tukey-windowed FFT
    3. Multitaper PSD
    4. Autocorrelation-based FFT
    5. Morlet Wavelet (time-averaged power)

    One plot is generated per (raw, event, title).
    """

    # PARAMETERS
    n_pad = (t_max)*1000             # zero-padding for FFT resolution
    plot_fmax = 120           # max freq in plots
    tukey_alpha = 0.25        # taper fraction for Tukey window
    wav_nfreqs = 80           # number of Morlet frequencies

    for i, event, title, new_raw in zip(idx, events, titles, raws):

        # --- Load epoch ---
        epoch = mne.Epochs(
            new_raw, event, event_id=i,
            tmin=t_min, tmax=t_max,
            baseline=baseline, reject=reject
        )
        epoch.load_data().crop(tmin=t_min, tmax=t_max)

        # fallback channel check
        if not all(ch in epoch.ch_names for ch in channels):
            channels = ["FpzT", "FpzB"]
            print("Fallback channels:", channels)

        # Weighted average
        wa = epoch.average(method="median")   # or your own function
        wa = wa.pick(channels)
        wa = wa.copy().shift_time(t_min, relative=False)

        # extract waveform
        x = wa.copy().crop(tmin=0, tmax=t_max)
        wave = x.get_data(picks=channels[0]).squeeze()
        wave = wave - np.mean(wave)
        sfreq = x.info['sfreq']

        n_pad = 75 #round(t_max*1000)

        # --- 1) Direct FFT ---
        fft_direct = np.fft.rfft(wave, n=n_pad)
        freqs = np.fft.rfftfreq(n_pad, 1/sfreq)
        psd_direct = np.abs(fft_direct)**2

        # --- 2) Tukey-window FFT ---
        tuk_win = tukey(len(wave), alpha=tukey_alpha)
        fft_tukey = np.fft.rfft(wave * tuk_win, n=n_pad)
        psd_tukey = np.abs(fft_tukey)**2
        
        freqs_per, psd_per = periodogram(wave, fs=1000, window='tukey', nfft=n_pad)
        psd_per /= psd_per.max()

        # --- 3) Multitaper ---
        psd_mt, freqs_mt = psd_array_multitaper(
            wave[np.newaxis, :],
            sfreq=sfreq,
            fmin=0, fmax=plot_fmax,
            bandwidth=2.7,
            adaptive=True, low_bias=True,
            normalization='full',
            verbose=False
        )
        psd_mt = psd_mt.squeeze()
        psd_mt_interp = np.interp(freqs, freqs_mt, psd_mt)

        psd_mt1, freqs_mt1 = psd_array_multitaper(
            wave[np.newaxis, :],
            sfreq=sfreq,
            fmin=0, fmax=plot_fmax,
            bandwidth=2.6,
            adaptive=True, low_bias=False,
            normalization='full',
            verbose=False
        )
        psd_mt1 = psd_mt1.squeeze()
        psd_mt_interp1 = np.interp(freqs, freqs_mt1, psd_mt1)

        # --- 4) Autocorrelation FFT ---
        ac = np.correlate(wave, wave, mode='full')
        ac = ac[len(ac)//2:]  # keep causal side
        fft_ac = np.fft.rfft(ac, n=n_pad)
        psd_ac = np.abs(fft_ac)

        # --- 5) Morlet Wavelet ---
        freqs_wav = np.linspace(1, plot_fmax, wav_nfreqs)
        n_cycles = freqs_wav * 0.05
        data_wav = wave[np.newaxis, np.newaxis, :]

        power = tfr_array_morlet(
            data_wav, sfreq,
            freqs=freqs_wav,
            n_cycles=n_cycles,
            output="power",
            zero_mean=True
        ).squeeze()  # → (nfreq, ntimes)

        psd_wav = power.mean(axis=-1)
        psd_wav_interp = np.interp(freqs, freqs_wav, psd_wav)


        
        # --- 6) Welch method ---
        freqs_welch1, psd_welch1 = welch(wave, fs=sfreq, nperseg=75, noverlap=32, nfft=len(wave))
        psd_welch1 /= psd_welch1.max()  # normalize

        # --- 7) Welch method ---
        freqs_welch2, psd_welch2 = welch(wave, fs=sfreq, nperseg=75, noverlap=32, nfft=n_pad)
        psd_welch2 /= psd_welch2.max()  # normalize




        # Normalize all PSD estimates to their respective maxima
        psd_direct  = psd_direct  / np.max(psd_direct)
        psd_tukey   = psd_tukey   / np.max(psd_tukey)
        psd_mt_interp = psd_mt_interp / np.max(psd_mt_interp)
        psd_mt_interp1 = psd_mt_interp1 / np.max(psd_mt_interp1)
        psd_ac      = psd_ac      / np.max(psd_ac)
        psd_wav_interp = psd_wav_interp / np.max(psd_wav_interp)


        

        # --- PLOT ALL 5 METHODS ---
        plt.figure(figsize=(9, 5))

        plt.plot(freqs, psd_direct, label='Direct FFT', color='C0')
        plt.plot(freqs, psd_tukey, label='Tukey FFT', color='C1')
        plt.plot(freqs, psd_mt_interp, label='Multitaper', color='C2')
        plt.plot(freqs, psd_mt_interp1, label='Multitaper bias', color='C4')
        plt.plot(freqs, psd_ac, label='Autocorr FFT', color='C3')
        #plt.plot(freqs, psd_wav_interp, label='Morlet Wavelet', color='C4')
        plt.plot(freqs_welch1, psd_welch1, label='Welch1', color='C5')
        plt.plot(freqs_welch2, psd_welch2, label='Welch2', color='C6')
        plt.plot(freqs_per, psd_per, label='periodogram', color ='C7')

        plt.xlim(1, plot_fmax)
        plt.yscale('log')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        plt.title(f"Spectrum Methods – {title} ({t_max*1000:.0f} ms)")
        plt.legend()
        plt.tight_layout()
        plt.show()



def plot_evokeds_scalp(raws,idx, events, titles,t_min, t_max, baseline=None,reject=None):
    evokeds_list = []
    wa_list = []
    

    for i, event, title,new_raw in zip(idx, events, titles,raws):
        print(f"Processing event ID {i}: {title}")

        # Epoching
        epoch = mne.Epochs(
            new_raw, event, event_id=i,
            tmin=t_min, tmax=t_max,
            baseline=baseline,
            reject=reject
        )
        epoch.load_data().crop(tmin=t_min, tmax=t_max)

        # Weighted average
        wa = variance_weighted_average(epoch)
        wa.comment = title
        evokeds_list.append(wa)
        wa_list.append(wa)

        # --- FFT spectrum of first 75 ms for all channels ---
        try:
            x = wa.copy()
            wave = x.get_data()  # shape (n_channels, n_times)
            sfreq = x.info['sfreq']
            tukey_alpha = 0.4
            tuk_win = np.hstack((tukey(100, alpha=tukey_alpha), np.zeros(100)))
            fft = np.fft.rfft(wave[0:len(tuk_win)] * tuk_win, n=len(tuk_win))
            fft = np.abs(fft)
            freqs= np.fft.rfftfreq(len(tuk_win), 1 / sfreq)

            # Option 1: Plot the average spectrum across channels
            plt.figure()
            plt.plot(freqs, fft.mean(axis=0))
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude")
            plt.title(f"Mean Spectrum (0–75 ms) - {title}")
            plt.xlim(1, 120)
            plt.show()

            # Option 2: Plot a topomap of spectral amplitude at a target frequency (e.g., 40 Hz)
            target_freq = 40  # or whatever is relevant
            idx_f = np.argmin(np.abs(freqs - target_freq))
            amp_at_f = fft[:, idx_f]

            fig, ax = plt.subplots()
            mne.viz.plot_topomap(
                amp_at_f, x.info, axes=ax, show=False,
                cmap='viridis', contours=0
            )
            ax.set_title(f"{title} – Spectral amplitude at {freqs[idx_f]:.1f} Hz (0–75 ms)")
            plt.show()

        except Exception as e:
            print(f"Skipping spectrum plot for {title}: {e}")

    # --- Compute difference waves (vs. last) ---
    difs = {
        titles[i]: mne.combine_evoked([wa_list[i], wa_list[-1]], weights=[1, -1])
        for i in range(len(wa_list) - 1)
    }

    # --- TOPOGRAPHIC PLOTS (time-domain evokeds) ---
    print("\nPlotting topographies for all conditions...")
    mne.viz.plot_evoked_topo(
        evokeds_list,
        title="Evoked Topographies for All Conditions",
        background_color='white'
    )

    if difs:
        diff_list = list(difs.values())
        for i, d in enumerate(diff_list):
            d.comment = f"Diff: {titles[i]} - {titles[-1]}"
        print("\nPlotting difference topographies vs. last condition...")
        mne.viz.plot_evoked_topo(
            diff_list,
            title="Difference Topographies vs. Last Condition",
            background_color='white'
        )

    return wa_list


def ERP_superimposition(wa, isi, T, comp_ind ,sim_label, fs=1000, jit=0.03, N=2000, plot=True, plot_jitter=False):
    """
    Simulate overlapping ERPs using the first ERP from a matrix of evoked ERPs,
    with optional jitter in ISI, repeated N times, and optionally plot the mean result.

    Parameters
    ----------
    wa : array_like or list
        ERP matrix (n_evokeds x n_times) or list of MNE Evoked objects.
        Only the first ERP will be used.
    isi : float
        Inter-stimulus interval in seconds.
    T : float
        Total duration of simulated time series in seconds.
    fs : float
        Sampling frequency in Hz (default 1000 Hz).
    jit : float
        Maximum jitter to apply to each ISI (±jit seconds). Default 0 (no jitter).
    N : int
        Number of repetitions of the simulation to average. Default 1.
    plot : bool
        If True, plot the resulting mean summed ERP.
    plot_jitter : bool
        If True, plot the accumulated jitter for each simulation.

    Returns
    -------
    sim_times : np.ndarray
        Time vector for the simulated series.
    sim_mean : np.ndarray
        Mean summed ERP across N simulations.
    jitter_matrix : np.ndarray
        Array of jittered ERP onsets for each simulation (N x n_erp).
    """
    # Extract first ERP
    if isinstance(wa, list):
        erp = wa[0].data.mean(axis=0)
    else:
        erp = wa[0]


    dt = 1 / fs
    n_samples = int(np.round(T * fs))
    sim_times = np.arange(n_samples) * dt

    erp_length = len(erp)
    n_erp = int(np.floor(T / isi))

    if (n_erp -20) <= 1000:
        ind = n_erp - 20
        print(n_erp - 20)

    # Store all simulations
    all_sim = np.zeros((N, n_samples))
    jitter_matrix = np.zeros((N, n_erp))  # store jittered onset times
    onset_ind = np.zeros((N,1))
    for sim in range(N):
        sim_signal = np.zeros(n_samples)
        current_time = 0.0


        for i in range(n_erp):
            # Generate uniform jitter
            jitter = np.random.uniform(-jit, jit)
            onset_time = max(0.0, current_time + jitter)  # prevent negative
            start_idx = int(np.round(onset_time * fs))
            end_idx = start_idx + erp_length
            if end_idx > n_samples:
                if start_idx > n_samples:
                    break
                erp_slice = erp[:n_samples - start_idx]
                end_idx = n_samples
            else:
                erp_slice = erp
            sim_signal[start_idx:end_idx] += erp_slice
            
            # Store jittered onset for debugging
            jitter_matrix[sim, i] = onset_time
            if i == ind:
                onset_ind[sim] = start_idx 
            # Increment nominal ISI
            current_time += (isi + jitter)

        all_sim[sim, :] = sim_signal
    # --- Extract ERP-aligned segments around each stored onset index ---
    erp_segments = np.zeros((N, len(erp)))

    for i in range(N):
        idx = int(onset_ind[i])
        if np.isnan(idx) or idx + len(erp) > n_samples:
            # if onset is missing or too close to end → pad with zeros
            erp_segments[i, :] = np.zeros(len(erp))
        else:
            erp_segments[i, :] = all_sim[i, idx:idx + len(erp)]
    mean_segment = erp_segments.mean(axis=0)
    if isinstance(wa, list):
        erp_ref = wa[comp_ind].data.mean(axis=0)
    else:
        erp_ref = wa[comp_ind]

    plt.figure(figsize=(8, 4))
    plt.plot(mean_segment-np.mean(mean_segment), label='Mean simulated segment', color='purple', lw=1.5)
    plt.plot(erp_ref-np.mean(erp_ref), label=sim_label, color='gray', lw=1.5, alpha=0.7)
    plt.title("Comparison: simulated mean segment vs" + sim_label)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude (µV)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Compute mean across N simulations
    sim_mean = all_sim.mean(axis=0)
    
    # --- Plot mean waveform ---
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(sim_times, sim_mean, color='purple', lw=1.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (µV)')
        plt.title(f'Mean simulated overlapping ERP (ISI={isi*1000:.0f} ms ±{jit*1000:.0f} ms, T={T}s, N={N})')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # --- Plot jittered onset times ---
    if plot_jitter:
        plt.figure(figsize=(8, 4))
        for sim in range(N):
            plt.plot(jitter_matrix[sim, :], np.ones(n_erp)*sim, 'o', alpha=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Simulation #')
        plt.title('Accumulated jittered ERP onsets')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return sim_times, sim_mean, jitter_matrix, onset_ind

def MLR_delayed_sum(evokeds,fm):
    #create a delayed sum of MLR waveforms
    
    evoked1 = evokeds[0]
    # Delay in seconds
    delays_ms = [0.0, 1/(4*fm), 3/(4*fm)]
    sfreq = evoked1.info['sfreq']  # sampling frequency
    n_channels, n_times = evoked1.data.shape

    # Determine maximum delay in samples to calculate output size
    max_delay_samples = int(np.ceil(max(delays_ms) * sfreq))
    total_length = n_times + max_delay_samples

    # Initialize the accumulator
    summed_data = np.zeros((n_channels, total_length))

    # Shift and sum each evoked
    for evoked, delay_sec in zip(evokeds, delays_ms):
        delay_samples = int(round(delay_sec * sfreq))
        shifted_data = np.zeros((n_channels, total_length))
        shifted_data[:, delay_samples:delay_samples + n_times] = evoked.data
        summed_data += shifted_data

    # Create new EvokedArray with correct info and time alignment
    # Shift times so 0 corresponds to original time 0 of first evoked
    new_times = evoked1.times[0] - delays_ms[0] + np.arange(total_length) / sfreq
    evoked_sum = mne.EvokedArray(summed_data, evoked1.info.copy(), tmin=new_times[0])

    # Plot result
    evoked_sum.plot(picks='M1', titles='Summed with Delays')
    x = evoked_sum.copy().crop(tmin=0.00, tmax=0.3)
    x.plot('M1')
    wave = evoked_sum.get_data(picks='M1')

    n_pad = 512
    wave_padded = np.pad(wave, ((0, 0), (0, n_pad - wave.shape[1])), mode='constant')

    # FFT
    fft = np.fft.rfft(wave_padded)
    freqs = np.fft.rfftfreq(n_pad, 1 / x.info['sfreq'])

    # Plot
    plt.plot(freqs, np.abs(fft[0]))  # e.g., channel 0
    plt.xlim(1, 100)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Interpolated Spectrum (Zero-padded)")
    plt.show()
    return evoked_sum.crop(0,0.3)

def plot_erp_var(wa, picks=None, window_ms=40, figsize=(12,6)):
    """
    Plot the variance between multiple MNE Evoked objects using a moving window.

    Parameters
    ----------
    wa : list of mne.Evoked
        List of Evoked objects
    picks : list or None
        Channels to include. Default: None (all channels)
    window_ms : float
        Window width in milliseconds for computing variance
    figsize : tuple
        Figure size
    """
    # Stack ERP data across ERPs
    erp_data = []
    for evoked in wa:
        if picks is None:
            pick_ch = np.arange(evoked.data.shape[0])
        else:
            pick_ch = picks
        erp_data.append(evoked.data[pick_ch, :])  # shape: (n_channels, n_times)

    erp_data = np.array(erp_data)  # shape: (n_erps, n_channels, n_times)

    times = wa[0].times
    sfreq = wa[0].info['sfreq']
    window_samples = int(window_ms / 1000 * sfreq)

    # Compute variance across ERPs within a moving window
    var_window = []
    for t in range(erp_data.shape[2] - window_samples):
        window_slice = erp_data[:, :, t:t+window_samples]  # shape: (n_erps, n_channels, window_samples)
        # Compute mean across window samples for each ERP and channel
        window_mean = window_slice.mean(axis=2)  # shape: (n_erps, n_channels)
        # Variance across ERPs, then mean across channels
        var_across_erps = window_mean.var(axis=0).mean()  # scalar
        var_window.append(var_across_erps)

    var_window = np.array(var_window)
    times_var = times[:len(var_window)] + (window_samples / 2) / sfreq

    # Plot
    plt.figure(figsize=figsize)
    plt.plot(times_var, var_window, color='purple', lw=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Variance across ERPs')
    plt.title(f'Variance across ERPs (window {window_ms} ms)')
    plt.grid(True)
    plt.show()

def plot_erp_derivative(wa, picks=None, smooth_sigma=0, figsize=(12,8)):
    """
    Plot derivatives of multiple MNE Evoked objects and SEM width.

    Parameters
    ----------
    wa : list of mne.Evoked
        List of Evoked objects
    picks : list or None
        Channels to plot. Default: None (all channels)
    smooth_sigma : float
        Gaussian smoothing sigma. Default: 0 (no smoothing)
    figsize : tuple
        Figure size
    """
    all_derivatives = []

    for evoked in wa:
        if picks is None:
            pick_ch = np.arange(evoked.data.shape[0])
        else:
            pick_ch = picks
        
        data = evoked.data[pick_ch, :]  # shape: (n_channels, n_times)
        deriv = np.diff(data, axis=1)
        
        if smooth_sigma > 0:
            deriv = gaussian_filter1d(deriv, sigma=smooth_sigma, axis=1)
        
        all_derivatives.append(deriv)

    # Concatenate all derivatives across evokeds and channels
    all_derivatives = np.vstack(all_derivatives)  # shape: (n_evokeds * n_channels, n_times-1)
    
    times = wa[0].times[1:]  # derivative aligns with interval between samples

    # Compute mean and SEM
    mean_deriv = all_derivatives.mean(axis=0)
    sem_deriv = all_derivatives.std(axis=0) / np.sqrt(all_derivatives.shape[0])

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # --- Derivative plot ---
    ax = axes[0]
    for d in all_derivatives:
        ax.plot(times, d, color='gray', alpha=0.3)
    ax.plot(times, mean_deriv, color='k', lw=2, label='Mean derivative')
    ax.fill_between(times, mean_deriv - sem_deriv, mean_deriv + sem_deriv,
                    color='blue', alpha=0.3, label='±1 SEM')
    ax.set_ylabel('d(ERP)/dt')
    ax.set_title('Derivative of all 25ms ISI Evokeds')
    ax.legend()
    ax.grid(True)

    # --- SEM width plot ---
    ax = axes[1]
    ax.plot(times, sem_deriv, color='red', lw=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('SEM width')
    ax.set_title('Width of SEM across all Evokeds')
    ax.grid(True)

    plt.tight_layout()
    plt.show()

def evoked_MLR_baselined(wa, window, picks=None, smooth_sigma=0, refine_width=0.01,
                         plot_channels=True, verbose=True):
    """
    Baseline and align evoked ERPs using the positive derivative max (Nb→P1 slope),
    refined around the group-level median derivative peak.
    Uses three zero crossings (two before, one after) for baseline correction.
    """

    tmin, tmax = window
    n_evokeds = len(wa)
    peak_times = []

    # --- Stage 1: Initial coarse detection ---
    for i, evoked in enumerate(wa):
        if picks is None:
            data = evoked.data
        else:
            data = evoked.data[picks, :]

        erp = data.mean(axis=0)
        deriv = np.diff(erp)
        if smooth_sigma > 0:
            deriv = gaussian_filter1d(deriv, sigma=smooth_sigma)
        deriv_times = evoked.times[1:]

        mask = (deriv_times >= tmin) & (deriv_times <= tmax)
        idx_window = np.where(mask)[0]
        if len(idx_window) == 0:
            raise ValueError(f"No samples found in window {window} for evoked {i}")

        idx_peak = idx_window[np.argmax(deriv[idx_window])]
        peak_times.append(deriv_times[idx_peak])

    # --- Stage 2: Median-centered refinement ---
    t_median = np.median(peak_times)
    if verbose:
        print(f"Median peak time across evokeds: {t_median:.4f} s")

    wa_baselined = []
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, evoked in enumerate(wa):
        if picks is None:
            data = evoked.data
        else:
            data = evoked.data[picks, :]

        erp = data.mean(axis=0)
        times = evoked.times

        deriv = np.diff(erp)
        if smooth_sigma > 0:
            deriv = gaussian_filter1d(deriv, sigma=smooth_sigma)
        deriv_times = times[1:]

        # Refined window around median
        t1 = t_median - refine_width
        t2 = t_median + refine_width
        mask = (deriv_times >= t1) & (deriv_times <= t2)
        idx_window = np.where(mask)[0]
        if len(idx_window) == 0:
            if verbose:
                print(f"Evoked {i}: No data in refined window.")
            wa_baselined.append(evoked.copy())
            continue

        idx_peak = idx_window[np.argmax(deriv[idx_window])]

        # Find zero crossings
        sign_change = np.sign(deriv)
        zero_crossings = np.where(np.diff(sign_change))[0]

        left_zc = zero_crossings[zero_crossings < idx_peak]
        right_zc = zero_crossings[zero_crossings > idx_peak]

        # --- UPDATED SECTION: two before + one after ---
        if len(left_zc) < 2 or len(right_zc) == 0:
            if verbose:
                print(f"Evoked {i}: Not enough zero crossings near refined peak.")
            wa_baselined.append(evoked.copy())
            continue

        i0, i1 = left_zc[-1], left_zc[-2]
        i2 = right_zc[0]
        zc_indices = [i0, i1, i2]

        baseline_value = np.mean(erp[zc_indices])

        evoked_bl = evoked.copy()
        evoked_bl.data = evoked_bl.data - baseline_value
        wa_baselined.append(evoked_bl)

        # --- Diagnostic plotting ---
        if plot_channels:
            for ch_data in evoked_bl.data:
                ax.plot(times, ch_data, alpha=0.25)
        else:
            ax.plot(times, erp - baseline_value, alpha=0.6, label=f'ERP {i+1}')

        ax.scatter(times[zc_indices], erp[zc_indices] - baseline_value,
                   color='red', s=40, zorder=5)
        ax.axvline(deriv_times[idx_peak], color='blue', ls='--', alpha=0.6)
        ax.axvspan(times[i0], times[i2], color='orange', alpha=0.1)

        if verbose:
            print(f"Evoked {i}: Peak={deriv_times[idx_peak]:.3f}s, "
                  f"ZCs=({times[i0]:.3f}, {times[i1]:.3f}, {times[i2]:.3f}), "
                  f"Baseline={baseline_value:.3e}")

    # --- Plot 1: Diagnostic ---
    ax.set_title('Derivative-based baseline alignment (median-refined, 3 ZCs)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (µV)')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Plot 2: Clean baselined ERPs ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for evoked_bl in wa_baselined:
        if plot_channels:
            for ch_data in evoked_bl.data:
                ax2.plot(evoked_bl.times, ch_data, alpha=0.3)
        else:
            ax2.plot(evoked_bl.times, evoked_bl.data.mean(axis=0), lw=1.5, alpha=0.7)
    ax2.axvline(t_median, color='blue', ls='--', alpha=0.5, label='Median Peak')
    ax2.set_title('Baselined Evokeds (aligned to median derivative peak, 3 ZCs)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude (µV)')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.show()
    labels = ['600ms', '400ms','200ms', '100ms', '33ms', 'avg', '100 ms']
    wa_baselined_dict = dict(zip(labels, wa_baselined))
    mne.viz.plot_compare_evokeds(
    wa_baselined_dict,
    picks=['Fpz','AFz'], 
    combine='mean', # or multiple picks if you like
    title="Evoked Responses: ACC for various ISIs, N=2000(8000)",
    show=False
    )
    plt.show()

    return wa_baselined

def ASSR(idx,new_raw,events,channels,plot):

    ASSR30 = mne.Epochs(new_raw, events, event_id = idx, tmin=0, tmax=4,baseline=None)
    ASSR30.load_data()
    ASSR30wa = variance_weighted_average(ASSR30)
    ASSR30_powspec= fft(ASSR30wa)
    if plot == True:
        ASSR30_powspec.plot(picks=channels)
    return ASSR30_powspec

def ASSR_MLR_scalp(assr_event_ids, assr_freqs, fft_event_ids, events, fft_titles, fft_window, new_raw, t_min, reject, plot=False, freq_lim=(1,110), inset_size=0.07):
    # --- Step 1: ASSR amplitudes ---
    ASSR_matrices = []
    for idx, f_target in zip(assr_event_ids, assr_freqs):
        pow_spec = ASSR(idx, plot)
        analysis = ASSR_Analysis(pow_spec, f_target, deltaf_low=8, deltaf_high=8, power=False,tagging_frequency_tollerance=0.05)
        ASSR_matrices.append(analysis)

    ch_names = ASSR_matrices[0].index.tolist()
    n_channels = len(ch_names)
    amp_matrix = np.array([[analysis.loc[ch, 'signal'] for analysis in ASSR_matrices] 
                           for ch in ch_names])
    noise_matrix = np.array([[analysis.loc[ch, 'noise'] for analysis in ASSR_matrices]
                             for ch in ch_names])
    
    # --- Step 2: FFT spectra ---
    fft_matrices = []
    for i, title in zip(fft_event_ids, fft_titles):
        epoch = mne.Epochs(new_raw, events=events, event_id=i,
                           tmin=t_min, tmax=fft_window,
                           baseline=None, reject=reject)
        epoch.load_data().crop(tmin=t_min, tmax=fft_window)
        wa = variance_weighted_average(epoch)
        if not wa.info.get_montage():
            montage = mne.channels.make_standard_montage('standard_1020')
            new_raw.set_montage(montage)
            wa.set_montage(montage)
        data = wa.get_data()
        sfreq = wa.info['sfreq']
        fft = np.fft.rfft(data, axis=1)
        freqs = np.fft.rfftfreq(data.shape[1], 1/sfreq)
        mask = (freqs >= freq_lim[0]) & (freqs <= freq_lim[1])
        fft_matrices.append(np.abs(fft[:, mask]))
    freqs_plot = freqs[mask]

    # --- Step 3: scalp layout ---
    if new_raw.get_montage():
        layout = mne.channels.find_layout(new_raw.info)
        pos2d = layout.pos[:, :2]
        layout_names = layout.names
        xcoords, ycoords, valid_idx = [], [], []
        for i, ch in enumerate(ch_names):
            if ch in layout_names:
                idx_pos = layout_names.index(ch)
                xcoords.append(pos2d[idx_pos, 0])
                ycoords.append(pos2d[idx_pos, 1])
                valid_idx.append(i)
        xcoords, ycoords = np.array(xcoords), np.array(ycoords)
    else:
        n_cols = int(np.ceil(np.sqrt(n_channels)))
        n_rows = int(np.ceil(n_channels / n_cols))
        xcoords = np.tile(np.linspace(0.2, 0.8, n_cols), n_rows)[:n_channels]
        ycoords = np.repeat(np.linspace(0.2, 0.8, n_rows), n_cols)[:n_channels]
        valid_idx = list(range(n_channels))

    xmin, xmax = xcoords.min(), xcoords.max()
    ymin, ymax = ycoords.min(), ycoords.max()

    # --- Step 3.5: compute y-axis limits robustly based on plotted data ---
    # FFT limits
    all_fft_vals = np.hstack([fm[:-5, :].flatten() for fm in fft_matrices])
    all_fft_vals = all_fft_vals[np.isfinite(all_fft_vals)]  # remove NaN/Inf

    print("\nAll FFT values considered for y-axis limits:")
    print(all_fft_vals)

    if all_fft_vals.size == 0:
        fft_ymin, fft_ymax = 0, 1e-20
    else:
        # Use percentiles to avoid extreme outliers
        fft_ymin, fft_ymax = np.percentile(all_fft_vals, [1, 99])
        print(fft_ymin,fft_ymax)
        fft_pad = 0.0 * (fft_ymax - fft_ymin)
        fft_ymin, fft_ymax = fft_ymin - fft_pad, fft_ymax + fft_pad
        print(f"\nFFT y-axis limits after padding: ymin={fft_ymin:.3e}, ymax={fft_ymax:.3e}")

    # ASSR limits
    all_assr_vals = amp_matrix.flatten()
    all_assr_vals = all_assr_vals[np.isfinite(all_assr_vals)]  # remove NaN/Inf

    print("\nAll ASSR values considered for y-axis limits:")
    print(all_assr_vals)

    if all_assr_vals.size == 0:
        assr_ymin, assr_ymax = 0, 1e-20
    else:
        assr_ymin, assr_ymax = np.percentile(all_assr_vals, [1, 99])
        assr_pad = 0.05 * (assr_ymax - assr_ymin)
        assr_ymin, assr_ymax = assr_ymin - assr_pad, assr_ymax + assr_pad
        print(f"\nASSR y-axis limits after padding: ymin={assr_ymin:.3e}, ymax={assr_ymax:.3e}")



    # --- Step 4: create figure ---
    fig = plt.figure(figsize=(12, 12))
    ax_main = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax_main.set_xticks([]); ax_main.set_yticks([])
    head = plt.Circle((0.5, 0.5), 0.45, transform=fig.transFigure,
                      fill=False, lw=1.0, color='k', alpha=0.4)
    fig.patches.append(head)

    axes_to_ch = {}

    # --- Step 5: plot insets ---
    for i, ch_idx in enumerate(valid_idx):
        # Normalize coordinates
        left = (xcoords[i] - xmin) / (xmax - xmin)
        bottom = (ycoords[i] - ymin) / (ymax - ymin)
        w = h = inset_size

        # Keep insets inside figure
        left = np.clip(left, w/2, 1 - w/2)
        bottom = np.clip(bottom, h/2, 1 - h/2)

        # Create axes
        ax_in = fig.add_axes([left - w/2, bottom - h/2, w, h])
        ax2 = ax_in.twinx()

        # --- Plot FFT spectra (blue, left axis) ---
        for fft_mat in fft_matrices:
            line_fft, = ax_in.plot(freqs_plot, fft_mat[ch_idx, :],
                                color='tab:blue', lw=1, alpha=0.8, zorder=1, picker=5)
            axes_to_ch[line_fft] = ch_idx  # map line to channel for interactive click

        # Set log scale for FFT magnitude axis
        ax_in.set_yscale("log")

        ax_in.set_ylim(fft_ymin, fft_ymax)
        ax_in.set_autoscale_on(False)
        ax_in.set_xticks([]); ax_in.set_yticks([])


        # --- Plot ASSR amplitudes (red, right axis) ---
        line_assr, = ax2.plot(assr_freqs, amp_matrix[ch_idx, :],
                            marker='o', linestyle='-', color='red', lw=1, zorder=2, picker=5)
        axes_to_ch[line_assr] = ch_idx  # map line to channel

        # Noise floor dotted line (not pickable)
        ax2.plot(assr_freqs, noise_matrix[ch_idx, :],
                linestyle=':', color='red', linewidth=0.8, alpha=0.7)
        ax2.set_yscale("log")
        ax2.set_ylim(assr_ymin, assr_ymax)
        ax2.set_autoscale_on(False)
        ax2.set_xticks([]); ax2.set_yticks([])

        ax_in.set_title(ch_names[ch_idx], fontsize=6)


    # --- Step 6: interactive viewer ---
    def on_pick(event):
        artist = event.artist
        if artist not in axes_to_ch:
            return
        ch_idx_clicked = axes_to_ch[artist]

        fig_full, ax_full = plt.subplots(figsize=(6, 4))
        ax2_full = ax_full.twinx()

        for fft_mat in fft_matrices:
            ax_full.plot(freqs_plot, fft_mat[ch_idx_clicked, :], color='tab:blue', alpha=0.8)
        ax_full.set_ylim(fft_ymin, fft_ymax)
        ax_full.tick_params(axis='y', colors='tab:blue')
        ax_full.spines['left'].set_color('tab:blue')

        ax2_full.plot(assr_freqs, amp_matrix[ch_idx_clicked, :],
                      marker='o', linestyle='-', color='red', label='ASSR amplitude')
        ax2_full.plot(assr_freqs, noise_matrix[ch_idx_clicked, :],
                      linestyle=':', color='red', linewidth=1, alpha=0.8, label='Noise floor')
        
        ax2_full.set_yscale("log")
        ax2_full.set_ylim(assr_ymin, assr_ymax)
        ax2_full.tick_params(axis='y', colors='red')
        ax2_full.spines['right'].set_color('red')
        
        ax_full.set_yscale('log')
        ax_full.set_xlabel("Frequency (Hz)")
        ax_full.set_ylabel("MLR FFT amplitude", color='tab:blue')
        ax2_full.set_ylabel("ASSR amplitude", color='red')

        fig_full.suptitle(f"{ch_names[ch_idx_clicked]} — ASSR (red) + MLR FFT (blue)", fontsize=12)
        ax2_full.legend(fontsize=8, loc='upper right')

        plt.tight_layout()
        plt.show()

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()

    return amp_matrix, fft_matrices, freqs_plot, ch_names

def plot_electrodes(raws,idx, events, titles, t_min, t_max, selected_channels,
                    baseline=None, reject=None, fft_plot=False):

    evokeds_list = []
    wa_list = []

    # -------- PREPROCESS EACH CONDITION --------
    for i, event, title, new_raw in zip(idx, events, titles,raws):
        print(f"Processing event ID {i}: {title}")

        # Epoch
        epoch = mne.Epochs(
            new_raw, event, event_id=i,
            tmin=t_min, tmax=t_max,
            baseline=baseline,
            reject=reject,
            preload=True
        )
        epoch.crop(tmin=t_min, tmax=t_max)

        # Weighted average
        wa = variance_weighted_average(epoch)
        wa.comment = title
        wa_list.append(wa)

        # Optional FFT removed from per-condition loop: will compute after

    # -------- CHANNEL VALIDATION --------
    available_ch = wa_list[0].ch_names
    selected_channels = [ch for ch in selected_channels if ch in available_ch]

    if not selected_channels:
        raise ValueError("None of the specified electrodes exist in data.")

    print(f"Plotting electrodes: {selected_channels}")

    # -------- SUBPLOT LAYOUT --------
    n_ch = len(selected_channels)
    n_cols = int(np.ceil(np.sqrt(n_ch)))
    n_rows = int(np.ceil(n_ch / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = np.array(axes).reshape(-1)

    times = wa_list[0].times

    # -------- COMPUTE COMMON Y-LIMITS --------
    y_min = np.inf
    y_max = -np.inf
    for wa in wa_list:
        for ch in selected_channels:
            ch_idx = available_ch.index(ch)
            y_min = min(y_min, wa.data[ch_idx].min())
            y_max = max(y_max, wa.data[ch_idx].max())
    y_pad = 0.05 * (y_max - y_min)
    y_min -= y_pad
    y_max += y_pad

    # -------- PLOT EACH ELECTRODE --------
    cmap = plt.get_cmap("tab10")  # good qualitative colormap
    colors = cmap.colors[:len(wa_list)]
    for ax, ch in zip(axes, selected_channels):
        ch_idx = available_ch.index(ch)

        for wa, col in zip(wa_list, colors):
            ax.plot(times, wa.data[ch_idx], label=wa.comment, linewidth=0.8)

        ax.axvline(0, linestyle='--', linewidth=0.8, color='black')
        ax.set_title(ch, fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (µV)")
        ax.grid(True)
        ax.set_ylim(y_min, y_max)

    # hide unused axes
    for ax in axes[n_ch:]:
        ax.axis("off")

    # Legend moved to north-east corner
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=1)

    fig.suptitle("Evoked Responses by Electrode", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # -------- FFT plots for selected channels (if requested) --------
    if fft_plot:
        try:
            tukey_alpha = 0.5
            tuk_len = 100
            tuk_win = np.hstack((tukey(tuk_len, alpha=tukey_alpha), np.zeros(tuk_len)))
            nfft = len(tuk_win)
            sfreq = wa_list[0].info['sfreq']
            freqs_fft = np.fft.rfftfreq(nfft, 1.0 / sfreq)
            plot_fmax = 120

            # layout for FFT subplots mirrors time-domain layout
            n_ch = len(selected_channels)
            n_cols = int(np.ceil(np.sqrt(n_ch)))
            n_rows = int(np.ceil(n_ch / n_cols))
            fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
            axes2 = np.array(axes2).reshape(-1)

            for ax, ch in zip(axes2, selected_channels):
                ch_idx = available_ch.index(ch)

                for wa, col in zip(wa_list, colors):
                    wave = wa.get_data()[ch_idx].squeeze()
                    fft_ch = np.abs(np.fft.rfft(wave[:len(tuk_win)] * tuk_win, n=nfft))
                    ax.plot(freqs_fft, fft_ch, label=wa.comment, color=col, linewidth=0.8)

                ax.set_title(ch, fontsize=10)
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Amplitude')
                ax.set_xlim(1, plot_fmax)
                ax.set_yscale('log')
                ax.grid(True)

            # hide unused axes
            for ax in axes2[n_ch:]:
                ax.axis('off')

            # unified legend
            handles, labels = axes2[0].get_legend_handles_labels()
            fig2.legend(handles, labels, loc='upper right', ncol=1)
            fig2.suptitle('FFT: Selected Channels')
            fig2.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()
        except Exception as e:
            print(f"FFT plotting failed: {e}")

    return wa_list


def plot_evokeds_scalp_spectra(new_raw,idx, events, titles,t_min,t_max, baseline=None, reject=None,
    freq_lim=(1, 170),
    inset_size=0.080,
    montage_name='standard_1020'
):
    """
    Plot evoked spectra (first 75 ms) as topographic inset spectra.
    Click any channel inset to open full-size spectrum.
    Also export FFT amplitudes of all electrodes in a matrix.

    Parameters
    ----------
    idx : list of event ids
    events : list of event definitions
    titles : list of condition names
    window : float
    t_spec : tuple, time window for FFT (default 0–0.075 s)
    freq_lim : tuple, frequency limits for FFT plots
    inset_size : float, relative size of each inset axes
    montage_name : str, standard montage if missing

    Returns
    -------
    wa_list : list of evoked objects
    fft_matrices : list of np.ndarray
        List of FFT amplitude matrices per condition (n_channels x n_freqs)
    freqs_plot : np.ndarray
        Frequency vector used for the FFT plots
    """
    evokeds_list = []
    wa_list = []
    fft_matrices = []

    # Build evokeds
    for i, event, title in zip(idx, events, titles):
        print(f"Processing event ID {i}: {title}")
        epoch = mne.Epochs(
            new_raw, event, event_id=i,
            tmin=t_min, tmax=t_max,
            baseline=baseline,
            reject=reject
        )
        epoch.load_data().crop(tmin=t_min, tmax=t_max)
        wa = variance_weighted_average(epoch)
        wa.comment = title
        evokeds_list.append(wa)
        wa_list.append(wa)

    # Apply montage if missing
    info = wa_list[0].info
    if not info.get_montage():
        print(f"No montage found. Applying '{montage_name}' montage.")
        montage = mne.channels.make_standard_montage(montage_name)
        new_raw.set_montage(montage)
        for w in wa_list:
            w.set_montage(montage)

    # Loop through conditions
    for wa, title in zip(wa_list, titles):
        x = wa.copy().crop(tmin=t_min, tmax=t_min+0.2)
        data = x.get_data()  # (n_channels, n_times)
        sfreq = x.info['sfreq']
        n_ch, n_times = data.shape

        # --- Tukey window (YOUR METHOD) ---
        tukey_alpha = 0.4
        tuk_len = min(150, n_times)  # safety
        tuk_win = np.hstack((
            tukey(tuk_len, alpha=tukey_alpha),
            np.zeros(n_times - tuk_len)
        ))

        # Apply window to ALL channels
        data_win = data * tuk_win[None, :]

        # FFT per channel
        fft = np.fft.rfft(data_win, axis=1)
        freqs = np.fft.rfftfreq(n_times, 1.0 / sfreq)
        amp = np.abs(fft)

        # Frequency mask
        mask = (freqs >= freq_lim[0]) & (freqs <= freq_lim[1])
        freqs_plot = freqs[mask]
        amp_plot = amp[:, mask]

        # Store amplitude matrix
        fft_matrices.append(amp_plot)

        # Layout for electrode positions
        layout = mne.channels.find_layout(x.info)
        pos2d = layout.pos[:, :2]
        ch_names = layout.names

        layout_order = [ch_names.index(ch) for ch in x.ch_names if ch in ch_names]
        pos2d = pos2d[layout_order]
        amp_plot = amp_plot[:len(pos2d)]

        xcoords = pos2d[:, 0]
        ycoords = pos2d[:, 1]
        xmin, xmax = xcoords.min(), xcoords.max()
        ymin, ymax = ycoords.min(), ycoords.max()

        # Figure
        fig = plt.figure(figsize=(10, 10))
        fig.suptitle(f"{title} — Channel spectra (0–75 ms)", fontsize=14)

        ax_main = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax_main.set_xticks([]); ax_main.set_yticks([])
        head = plt.Circle((0.5, 0.5), 0.45, transform=fig.transFigure,
                          fill=False, lw=1.0, color='k', alpha=0.4)
        fig.patches.append(head)

        # Store mapping from axes to channel index
        axes_to_ch = {}

        for ch_idx, ch_name in enumerate(x.ch_names):
            if ch_name not in ch_names:
                continue
            li = ch_names.index(ch_name)
            px, py = pos2d[ch_idx, 0], pos2d[ch_idx, 1]
            left = (px - xmin) / (xmax - xmin)
            bottom = (py - ymin) / (ymax - ymin)
            w = inset_size; h = inset_size

            ax_in = fig.add_axes([left - w/2, bottom - h/2, w, h])
            line, = ax_in.plot(freqs_plot, amp_plot[ch_idx, :], picker=5)  # picker enables click

            ax_in.set_yscale('log')   

            ax_in.set_xticks([]); ax_in.set_yticks([])
            ax_in.set_title(ch_name, fontsize=6)


            axes_to_ch[ax_in] = ch_idx

        # Click callback
        def on_pick(event):
            ax_clicked = event.artist.axes
            if ax_clicked not in axes_to_ch:
                return
            ch_idx_clicked = axes_to_ch[ax_clicked]

            fig_full = plt.figure()
            ax_full = fig_full.add_subplot(111)
            ax_full.plot(freqs_plot, amp_plot[ch_idx_clicked, :])
            ax_full.set_xlabel("Frequency (Hz)")
            ax_full.set_ylabel("Amplitude")
            ax_full.set_title(f"{x.ch_names[ch_idx_clicked]} full spectrum")
            plt.show()

        fig.canvas.mpl_connect('pick_event', on_pick)
        plt.show()

    return wa_list, fft_matrices, freqs_plot


# not a good function
def run_decomposition(raw, events, event_id, tmin, tmax,
                      baseline=None, reject=None, n_components=None,
                      method='ica', random_state=42):
    """
    Epoch raw data, run ICA or PCA, plot topomaps of all components,
    then allow user to select a component to see its waveform per channel
    in a topographic layout (like plot_evoked_topo).

    Returns the decomposition object, sources, and the evoked.
    """

    # ---- Epoch raw data ----
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                        baseline=baseline, reject=reject, preload=True)
    epochs.load_data().crop(tmin=tmin, tmax=tmax)

    # ---- Variance-weighted evoked ----
    evoked = variance_weighted_average(epochs)
    data = evoked.get_data()  # n_channels x n_times
    n_channels, n_times = data.shape

    if n_components is None or n_components > n_channels:
        n_components = n_channels

    # ---- Decomposition ----
    if method.lower() == 'ica':
        decomposition = mne.preprocessing.ICA(n_components=n_components, random_state=random_state)
        decomposition.fit(epochs)
        sources = decomposition.get_sources(evoked).get_data()
        mixing_matrix = decomposition.get_components()

        # Topomap of all ICA components
        fig, axes = plt.subplots(1, n_components, figsize=(2.5 * n_components, 3))
        if n_components == 1:
            axes = [axes]
        for c, ax in enumerate(axes):
            mne.viz.plot_topomap(mixing_matrix[:, c], evoked.info,
                                 axes=ax, show=False, cmap="RdBu_r")
            ax.set_title(f"ICA {c}")
        plt.suptitle("ICA Component Topographies", fontsize=14)
        plt.tight_layout()
        plt.show()

    elif method.lower() == 'pca':
        decomposition = PCA(n_components=n_components, random_state=random_state)
        sources = decomposition.fit_transform(data.T).T
        components_matrix = decomposition.components_

        # Topomap of all PCA components
        fig, axes = plt.subplots(1, n_components, figsize=(2.5 * n_components, 3))
        if n_components == 1:
            axes = [axes]
        for c, ax in enumerate(axes):
            mne.viz.plot_topomap(components_matrix[c, :], evoked.info,
                                 axes=ax, show=False, cmap="RdBu_r")
            ax.set_title(f"PCA {c}")
        plt.suptitle("PCA Component Topographies", fontsize=14)
        plt.tight_layout()
        plt.show()

    else:
        raise ValueError("Invalid method. Use 'ica' or 'pca'.")

    # ---- Prompt user to select component ----
    comp_idx = int(input(f"Enter component index to visualize (0-{n_components-1}): "))

    # ---- Reconstruct the selected component for all channels ----
    if method.lower() == 'ica':
        reconstructed = np.outer(mixing_matrix[:, comp_idx], sources[comp_idx])
    else:  # PCA
        reconstructed = np.outer(components_matrix[comp_idx, :], sources[comp_idx])

    # ---- Convert reconstructed component to EvokedArray for plotting like topo ----
    comp_evokeds = []
    for ch_idx in range(n_channels):
        ev = mne.EvokedArray(reconstructed[ch_idx:ch_idx+1, :],
                             mne.create_info([evoked.ch_names[ch_idx]],
                                             sfreq=evoked.info['sfreq'],
                                             ch_types='eeg'),
                             tmin=evoked.tmin)
        comp_evokeds.append(ev)

    # Plot using MNE topo layout
    mne.viz.plot_evoked_topo(
        comp_evokeds,
        title=f"{method.upper()} Component {comp_idx} Waveforms per Channel",
        background_color='white'
    )

    return decomposition, sources, evoked



#Significance functions, not finalized

def ERP_t_test(wa, channel, alpha=0.05, fdr=False):
    """
    Perform a two-sample t-test at each time point between real and random-trigger evoked responses,
    with optional FDR correction.

    Parameters
    ----------
    wa : list of mne.Evoked
        List of Evoked objects. Assumes first half are real evokeds, second half random-trigger evokeds.
    channel : str
        Channel name to analyze (default: 'M1').
    alpha : float
        Significance threshold for plotting (default: 0.05).
    fdr : bool
        Whether to apply FDR correction across time points (default: True).

    Returns
    -------
    t_vals : np.ndarray
        T-values per time point.
    p_vals_corr : np.ndarray
        Corrected (or raw) p-values per time point.
    reject : np.ndarray
        Boolean array indicating significant time points after correction.
    """

    # --- Split evokeds ---
    n_total = len(wa)
    assert n_total % 2 == 0, "wa must contain an even number of evokeds (half real, half random)."
    n_half = n_total // 2
    real_evokeds = wa[:n_half]
    rand_evokeds = wa[n_half:]

    # --- Extract channel data ---
    ch_idx = wa[0].ch_names.index(channel)
    real_data = np.stack([ev.data[ch_idx] for ev in real_evokeds])   # (n_real, n_times)
    rand_data = np.stack([ev.data[ch_idx] for ev in rand_evokeds])   # (n_rand, n_times)
    times = wa[0].times

    # --- Compute t-test per time point ---
    t_vals, p_vals = ttest_ind(real_data, rand_data, axis=0, equal_var=False)

    # --- Apply FDR correction ---
    if fdr:
        reject, p_vals_corr = fdr_correction(p_vals, alpha=alpha)
    else:
        reject, p_vals_corr = p_vals < alpha, p_vals

    # --- Plot results ---
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                           gridspec_kw={'height_ratios': [2, 1]})

    # --- Plot evoked responses ---
    real_mean, real_sem = real_data.mean(0), real_data.std(0) / np.sqrt(real_data.shape[0])
    rand_mean, rand_sem = rand_data.mean(0), rand_data.std(0) / np.sqrt(rand_data.shape[0])

    ax[0].plot(times, real_mean, color='C0', lw=2, label='Real evoked')
    ax[0].fill_between(times, real_mean - real_sem, real_mean + real_sem, color='C0', alpha=0.2)
    ax[0].plot(times, rand_mean, color='r', lw=2, label='Random-trigger evoked')
    ax[0].fill_between(times, rand_mean - rand_sem, rand_mean + rand_sem, color='r', alpha=0.2)

    # Highlight significant time points (after correction)
    ax[0].fill_between(
        times, ax[0].get_ylim()[0], ax[0].get_ylim()[1],
        where=reject, color='gray', alpha=0.25,
        label='Significant (FDR)' if fdr else f'p < {alpha}',
        transform=ax[0].get_xaxis_transform()
    )

    ax[0].set_ylabel('Amplitude (µV)')
    ax[0].set_title(f'{channel}: Real vs Random-trigger Evoked (t-test per time point)')
    ax[0].legend(loc='upper right')
    ax[0].grid(True)

    # --- Plot p-values ---
    ax[1].plot(times, p_vals_corr, color='k', lw=1)
    ax[1].axhline(alpha, color='r', linestyle='--', label=f'α = {alpha}')
    ax[1].set_yscale('log')
    ax[1].set_ylabel('p-value' + (' (FDR corrected)' if fdr else ''))
    ax[1].set_xlabel('Time (s)')
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    return t_vals, p_vals_corr, reject


def ERP_permutation_test_pointwise(wa, channel, alpha=0.05, n_permutations=1000):
        """
        Perform a pointwise permutation test between real and random-trigger evoked responses.

        Parameters
        ----------
        wa : list of mne.Evoked
            List of Evoked objects. Assumes the first half are real evokeds,
            the second half random-trigger evokeds.
        channel : str
            Channel name to analyze.
        alpha : float
            Significance threshold (default 0.05).
        n_permutations : int
            Number of permutations for the test (default 1000).

        Returns
        -------
        t_obs : np.ndarray
            Observed t-values for each time point.
        p_values : np.ndarray
            Uncorrected permutation-based p-values per time point.
        p_fdr : np.ndarray
            FDR-corrected p-values per time point.
        """

        # --- Split the evokeds ---
        n_total = len(wa)
        assert n_total % 2 == 0, "wa must contain an even number of evokeds (half real, half random)."
        n_half = n_total // 2
        real_evokeds = wa[:n_half]
        rand_evokeds = wa[n_half:]

        # --- Extract data for chosen channel ---
        ch_idx = wa[0].ch_names.index(channel)
        real_data = np.stack([ev.data[ch_idx] for ev in real_evokeds])   # shape (n_real, n_times)
        rand_data = np.stack([ev.data[ch_idx] for ev in rand_evokeds])   # shape (n_rand, n_times)
        times = wa[0].times

        n1, n2 = real_data.shape[0], rand_data.shape[0]
        pooled = np.vstack([real_data, rand_data])
        n_total = pooled.shape[0]
        n_times = pooled.shape[1]
        rng = np.random.default_rng()

        # --- Observed difference ---
        m1 = real_data.mean(axis=0)
        m2 = rand_data.mean(axis=0)
        s1 = real_data.var(axis=0, ddof=1)
        s2 = rand_data.var(axis=0, ddof=1)
        denom = np.sqrt(s1/n1 + s2/n2)
        denom[denom == 0] = np.inf
        t_obs = (m1 - m2) / denom

        # --- Build permutation null distribution ---
        perm_t = np.zeros((n_permutations, n_times))
        for i in range(n_permutations):
            perm_idx = rng.permutation(n_total)
            grp1 = pooled[perm_idx[:n1]]
            grp2 = pooled[perm_idx[n1:]]

            m1p = grp1.mean(axis=0)
            m2p = grp2.mean(axis=0)
            s1p = grp1.var(axis=0, ddof=1)
            s2p = grp2.var(axis=0, ddof=1)
            denom_p = np.sqrt(s1p/n1 + s2p/n2)
            denom_p[denom_p == 0] = np.inf
            perm_t[i] = (m1p - m2p) / denom_p

        # --- Two-tailed p-values per timepoint ---
        abs_perm = np.abs(perm_t)
        abs_tobs = np.abs(t_obs)[np.newaxis, :]
        p_values = (np.sum(abs_perm >= abs_tobs, axis=0) + 1) / (n_permutations + 1)

        # --- FDR correction ---
        _, p_fdr = fdr_correction(p_values, alpha=alpha)

        # --- Plot results ---
        plt.figure(figsize=(10, 5))

        # Mean ± SEM for both groups
        real_mean, real_sem = real_data.mean(0), real_data.std(0) / np.sqrt(n1)
        rand_mean, rand_sem = rand_data.mean(0), rand_data.std(0) / np.sqrt(n2)

        plt.plot(times, real_mean, color='C0', lw=2, label='Real evoked')
        plt.fill_between(times, real_mean - real_sem, real_mean + real_sem, color='C0', alpha=0.2)

        plt.plot(times, rand_mean, color='r', lw=2, label='Random-trigger evoked')
        plt.fill_between(times, rand_mean - rand_sem, rand_mean + rand_sem, color='r', alpha=0.2)

        # Highlight significant time points
        sig_mask = p_fdr < alpha
        plt.fill_between(times, plt.gca().get_ylim()[0], plt.gca().get_ylim()[1],
                        where=sig_mask, color='gray', alpha=0.2,
                        transform=plt.gca().get_xaxis_transform(),
                        label='Significant (FDR < α)')

        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (µV)')
        plt.title(f'{channel}: Real vs Random-trigger Evoked (Pointwise Permutation Test)')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return t_obs, p_values, p_fdr


def ERP_permutation_test(wa, channel, alpha=0.05, n_permutations=1000):


    """
    Perform a permutation-based cluster test between real and random-trigger evoked responses.

    Parameters
    ----------
    wa : list of mne.Evoked
        List of Evoked objects. Assumes the first half are real evokeds, the second half random-trigger evokeds.
    channel : str
        Channel name to analyze.
    alpha : float
        Significance threshold for clusters.
    n_permutations : int
        Number of permutations for the test (default 5000).

    Returns
    -------
    T_obs : np.ndarray
        Observed test statistic (t-values) per time point.
    clusters : list
        List of boolean arrays representing significant clusters.
    cluster_p_values : np.ndarray
        P-values associated with each cluster.
    """

    # --- Split the evokeds ---
    n_total = len(wa)
    assert n_total % 2 == 0, "wa must contain an even number of evokeds (half real, half random)."
    n_half = n_total // 2
    real_evokeds = wa[:n_half]
    rand_evokeds = wa[n_half:]

    # --- Extract data for chosen channel ---
    ch_idx = wa[0].ch_names.index(channel)
    real_data = np.stack([ev.data[ch_idx] for ev in real_evokeds])   # shape (n_real, n_times)
    rand_data = np.stack([ev.data[ch_idx] for ev in rand_evokeds])   # shape (n_rand, n_times)
    times = wa[0].times

    # --- Run permutation cluster test ---
    X = [real_data, rand_data]  # two groups
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
        X, n_permutations=n_permutations, tail=0, n_jobs=1
    )

    # --- Plot the results ---
    plt.figure(figsize=(10, 5))

    # Plot mean ± SEM for both groups
    real_mean, real_sem = real_data.mean(0), real_data.std(0) / np.sqrt(len(real_data))
    rand_mean, rand_sem = rand_data.mean(0), rand_data.std(0) / np.sqrt(len(rand_data))

    plt.plot(times, real_mean, color='C0', lw=2, label='Real evoked')
    plt.fill_between(times, real_mean - real_sem, real_mean + real_sem, color='C0', alpha=0.2)

    plt.plot(times, rand_mean, color='r', lw=2, label='Random-trigger evoked')
    plt.fill_between(times, rand_mean - rand_sem, rand_mean + rand_sem, color='r', alpha=0.2)

    # --- Highlight significant clusters ---
    for c, p_val in zip(clusters, cluster_p_values):
        if p_val < alpha:
            plt.axvspan(times[c][0], times[c][-1], color='gray', alpha=0.3,
                        label=f'Significant cluster (p={p_val:.3f})')

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.title(f'{channel}: Real vs Random-trigger Evoked (Permutation Cluster Test)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return T_obs, clusters, cluster_p_values


def select_stimuli_ui(raw_list, all_events, all_event_ids, event_to_raw):
    """
    UI to pick stimuli (multi-select). Uses the provided event_to_raw mapping
    to pick the correct raw/events for each stimulus.

    Parameters
    ----------
    raw_list : list
        list of raw objects
    all_events : list
        list of event arrays (one per raw); each event array may be e.g. Nx3 with IDs in column 2
    all_event_ids : dict
        mapping from stim_name -> stim_id (int)
    event_to_raw : dict
        mapping from stim_name -> raw_list index (int)

    Returns
    -------
    titles_selected, raws_selected, events_selected, MLR_ids_selected
    """

    stim_names = list(all_event_ids.keys())

    # Basic validation of mapping keys (optional)
    missing = [s for s in stim_names if s not in event_to_raw]
    if missing:
        raise ValueError(f"event_to_raw is missing entries for: {missing}")

    selected_indices = []

    def on_submit():
        nonlocal selected_indices
        selected_indices = listbox.curselection()
        root.quit()
        root.destroy()

    root = tk.Tk()
    root.title("Select stimuli")
    root.geometry("420x800")

    tk.Label(root, text="Select stimuli to include (Ctrl+Click multiple):").pack(pady=(6,0))

    listbox = tk.Listbox(root, selectmode=tk.MULTIPLE, width=60, height=30)
    for stim in stim_names:
        # show stim name and assigned raw index for clarity
        display = f"{stim} (raw_idx={event_to_raw.get(stim)})"
        listbox.insert(tk.END, display)
    listbox.pack(padx=10, pady=8)

    tk.Button(root, text="OK", command=on_submit).pack(pady=(0,8))

    root.mainloop()

    # Build returned lists using the provided mapping
    titles_selected = []
    raws_selected = []
    events_selected = []
    MLR_ids_selected = []

    for i in selected_indices:
        stim_name = stim_names[i]
        raw_idx = event_to_raw[stim_name]  # will KeyError earlier if missing
        titles_selected.append(stim_name)
        raws_selected.append(raw_list[raw_idx])
        events_selected.append(all_events[raw_idx])
        MLR_ids_selected.append(all_event_ids[stim_name])

    return titles_selected, raws_selected, events_selected, MLR_ids_selected

def rename_ins_annotations(raw, suffix):
    """Rename INS-related annotation labels to include suffix (_A or _E)."""
    desc = raw.annotations.description.copy()

    mapping = {
        'ins_100_D': f'ins_100_D_{suffix}',
        'ins_100_U': f'ins_100_U_{suffix}',
        'ins_600_D': f'ins_600_D_{suffix}',
        'ins_600_U': f'ins_600_U_{suffix}',
    }

    for old, new in mapping.items():
        desc[desc == old] = new

    raw.annotations.description = desc
    return raw



