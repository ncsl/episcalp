from pathlib import Path
import numpy as np
import collections
import scipy as sc

import json

import mne
from mne.time_frequency import psd_array_multitaper
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals
from scipy.integrate import simps
from scipy.fft import rfft, rfftfreq, irfft
import matplotlib.pyplot as plt

from episcalp.utils.utils import compute_sample_points
from sample_code.utils import _standard_channels


def run_bandpass_analysis(
    bids_path,
    band_dict,
    winsize_sec,
    stepsize_sec,
    deriv_path,
    reference,
    threshold: int = 3,
    save_intermed: bool = True,
    verbose: bool = False,
):
    subject = bids_path.subject
    raw = read_raw_bids(bids_path)
    ntimes = raw.n_times
    nchs = len(raw.ch_names)
    ch_names = raw.ch_names
    sfreq = raw.info["sfreq"]

    winsize = winsize_sec * int(sfreq)
    stepsize = stepsize_sec * int(sfreq)

    sample_points = compute_sample_points(ntimes, winsize, stepsize)
    #sample_points = [sample_points[0]]

    band_names = list(band_dict.keys())

    bandpower_dict = dict()
    for ch in ch_names:
        bandpower_dict[ch] = dict()
        for band_name in band_names:
            bandpower_dict[ch][band_name] = []

    mean_ptp_amplitude_dict = collections.defaultdict(list)

    for (start_samp, end_samp) in sample_points:
        data = raw.get_data(start=start_samp, stop=end_samp)
        power_dict = _compute_power_window(data, band_dict, sfreq, ch_names)
        for ch in ch_names:
            for band_name in band_names:
                bandpower_dict[ch][band_name].append(power_dict[band_name][ch])
        ptp_amplitude_dict = _compute_ptp_amplitude_window(data, sfreq, ch_names)
        for ch in ch_names:
            mean_ptp_amplitude_dict[ch].append(ptp_amplitude_dict[ch])

    deriv_dir = deriv_path / "band-power" / f"winsize-{winsize}" / f"stepsize-{stepsize}" / reference / f"sub-{subject}"
    deriv_dir.mkdir(exist_ok=True, parents=True)

    if save_intermed:
        bandpower_fname = Path(f"{bids_path.basename.split('_eeg')[0]}_bandpower.json")
        bandpower_fpath = deriv_dir / bandpower_fname

        with open(bandpower_fpath, 'w+') as fid:
            bandpower_dict["samplepoints"] = sample_points
            json.dump(bandpower_dict, fid, indent=4)
            bandpower_dict.pop("samplepoints")

        ptp_amplitude_fname = Path(bandpower_fname.name.replace("bandpower", "ptpamplitude"))
        ptp_amplitude_fpath = deriv_dir / ptp_amplitude_fname

        with open(ptp_amplitude_fpath, 'w+') as fid:
            json.dump(mean_ptp_amplitude_dict, fid, indent=4)

    outlier_window_dict_ = _determine_outlier_windows(bandpower_dict, threshold)
    [print(f"{key}: {value}") for key, value in outlier_window_dict_.items()]

    outlier_window_dict = _confirm_outlier_window_peaks(outlier_window_dict_, mean_ptp_amplitude_dict)
    [print(f"{key}: {value}") for key, value in outlier_window_dict.items()]

    outlier_windows = _convert_outlier_windows_to_samplepoints(outlier_window_dict, sample_points)

    outlier_fname = Path(f"{bids_path.basename.split('_eeg')[0]}_outliers.json")
    outlier_fpath = deriv_dir / outlier_fname
    with open(outlier_fpath, 'w+') as fid:
        json.dump(outlier_windows, fid, indent=4)

    outlier_windows_full = _fill_chs(outlier_windows)
    outlier_fname = Path(f"{bids_path.basename.split('_eeg')[0]}_outliersfull.json")
    outlier_fpath = deriv_dir / outlier_fname
    with open(outlier_fpath, 'w+') as fid:
        json.dump(outlier_windows_full, fid, indent=4)

    return outlier_window_dict


def _fill_chs(sparse_dict):
    full_dict = dict()
    standard_channels = _standard_channels()
    for ch in standard_channels:
        full_dict[ch] = sparse_dict.get(ch.upper(), [])
    return full_dict


def _compute_power_window(data, band_dict, sfreq, ch_names):
    bandpower_window = collections.defaultdict(dict)
    for band_name, band_values in band_dict.items():
        band_window = _bandpower(data, sfreq, band_values, ch_names)
        bandpower_window[band_name] = band_window
    return bandpower_window


def _compute_ptp_amplitude_window(data, sfreq, ch_names):
    amplitude_dict = dict()
    for idx, ch in enumerate(ch_names):
        ch_data = data[idx, :]
        ptp_amplitude = _ptp_amplitude(ch_data, sfreq)
        amplitude_dict[ch] = ptp_amplitude
    return amplitude_dict


def _bandpower(data, sf, band, ch_names):
    """
    Calculate power in the given frequency band for the provided channels.

    Parameters
    ----------
    data: np.ndarray
        nch x nSamp matrix containing the eeg data
    sf: int
        Sampling frequency
    band: list[int]
        List in the form of [lFreq, hFreq]
    ch_names: list[str]
        List of channel names

    Returns
    -------
    band_dict: dict
        Dictionary where keys are channel names and values are power in the provided band.

    """
    psd, freqs = psd_array_multitaper(data, sf, band[0], band[1], adaptive=True, normalization='full', verbose=0)
    # Calculate frequency resolution
    freq_res = freqs[1] - freqs[0]
    # Specify band range
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    # PSD is a series of discrete values, cannot be directly integrated, so use SIMPS to make a parabolic approximation
    band_power = simps(psd[:,idx_band], dx=freq_res)
    band_dict = {}
    for ind, bp in enumerate(band_power):
        band_dict[ch_names[ind]] = bp
    return band_dict


def _ptp_amplitude(ch_data, sfreq):
    delta_data = _get_delta_activity(ch_data, sfreq)
    min_distance = sfreq / 4
    peaks, peaks_properties = sc.signal.find_peaks(delta_data)
    troughs, troughs_properties = sc.signal.find_peaks(-delta_data)

    peak_amplitudes = delta_data[peaks]
    trough_amplitudes = delta_data[troughs]

    ptp_amplitudes = [abs(p-t) for p, t in zip(peak_amplitudes, trough_amplitudes)]

    if peaks[0] < troughs[0]:
        peak_amplitudes_ = peak_amplitudes[1:]
        trough_amplitudes_ = trough_amplitudes[:-1]
    else:
        peak_amplitudes_ = peak_amplitudes[:-1]
        trough_amplitudes_ = trough_amplitudes[1:]

    ptp_amplitudes_ = [abs(p-t) for p, t in zip(peak_amplitudes_, trough_amplitudes_)]
    ptp_amplitudes.extend(ptp_amplitudes_)

    return ptp_amplitudes


def _get_delta_activity(data, sfreq):
    N = len(data)
    yf = rfft(data)
    xf = rfftfreq(N, 1 / sfreq)
    start = min([i for i, x in enumerate(xf) if x >= 0.5])
    stop = min([i for i, x in enumerate(xf) if x >= 4.0])
    yf[0:start] = 0
    yf[stop+1:len(yf)] = 0

    new_sig = irfft(yf)

    return new_sig


def _determine_outlier_windows(bandpower_dict, threshold):
    outlier_window_dict = collections.defaultdict(list)
    total_delta = []
    [total_delta.extend(bandpower_dict[key]['delta']) for key in bandpower_dict.keys()]
    mean_delta = np.mean(total_delta)
    stdev_delta = np.std(total_delta)
    threshold_delta = mean_delta + threshold * stdev_delta
    for chname in bandpower_dict.keys():
        delta_power = bandpower_dict[chname]['delta']
        [outlier_window_dict[chname].append(idx) for idx, delta in enumerate(delta_power) if delta > threshold_delta]
    return outlier_window_dict


def _confirm_outlier_window_peaks(window_dict, amplitude_dict):
    confirmed_window_dict = collections.defaultdict(list)
    for chname, windows in window_dict.items():
        for window in windows:
            window_peaks_ = amplitude_dict[chname][window]
            if all(wp < 1 for wp in window_peaks_):
                window_peaks = [1e6 * win for win in window_peaks_]
            else:
                window_peaks = window_peaks_.copy()
            mean_amplitude = np.mean(window_peaks)
            std_amplitude = np.std(window_peaks)
            if 50 < mean_amplitude < 100 and std_amplitude < 50:
                confirmed_window_dict[chname].append(window)
    return confirmed_window_dict


def _convert_outlier_windows_to_samplepoints(outlier_window_dict, sample_points):
    sample_point_dict = collections.defaultdict(list)
    for chname, windows in outlier_window_dict.items():
        [sample_point_dict[chname].append(sample_points[win]) for win in windows]
    return sample_point_dict


if __name__ == "__main__":
    root = Path("D:/OneDriveParent/OneDrive - Johns Hopkins/Shared Documents/40Hz-30")
    subjects = get_entity_vals(root, 'subject')
    #subjects = [subjects[0]]
    for subject in subjects:
        bids_entities = {
            "subject": subject,
            "session": "initialvisit",
            "task": "monitor",
            "run": "01",
            "datatype": "eeg",
            "extension": ".vhdr"
        }
        bids_path_ = BIDSPath(root=root, **bids_entities)
        bids_path = bids_path_.match()
        if len(bids_path) == 0:
            bids_entities.update({"session": "awake"})
            bids_path_ = BIDSPath(root=root, **bids_entities)
            bids_path = bids_path_.match()
        bids_path = bids_path[0]
        band_dict = {
            "beta": [13, 30],
            "alpha": [8, 12],
            "theta": [4, 7],
            "delta": [1, 3]
        }
        winsize = 5
        stepsize = 1
        deriv_path = root / "derivatives"
        out_dict = run_bandpass_analysis(bids_path, band_dict, winsize, stepsize, deriv_path, "monopolar",
                                         save_intermed=False)
        #print(out_dict)

