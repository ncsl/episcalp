import collections
from typing import Dict
from mne_bids.path import get_bids_path_from_fname
import numpy as np
from mne import make_fixed_length_epochs
from mne.time_frequency import EpochsTFR
from mne_bids import read_raw_bids, get_entity_vals, get_entities_from_fname, BIDSPath

from episcalp.utils import (
    get_best_matching_montage,
)
from .persyst import compute_spike_df


def _preprocess_epochs_tfr(data):
    """Turn TFR data into a 2D array."""
    assert data.ndim == 4

    # take the average over frequencies
    data = np.mean(data, axis=2)

    # move the epoch ("window") axis to last
    data = np.moveaxis(data, 0, -2)

    # compress the time axis
    data = np.mean(data, axis=-1)

    # convert to dB
    data = 20 * np.log10(data)

    data = np.reshape(data, (data.shape[0], -1))
    return data


def _map_ch_to_type(ch_name):
    """Helper function to map certain EEG patterns to mne-python type channels."""
    from collections import OrderedDict

    ch_type_pattern = OrderedDict(
        [("stim", ("Mark",)), ("misc", ("DC", "NA", "Z")), ("bio", ("X",))]
    )
    for key, kinds in ch_type_pattern.items():
        if any(kind in ch_name for kind in kinds):
            return key
    return "eeg"


def _get_montage_rename_dict(ch_names, montage_ch_names):
    channel_rename = {}
    for ch in ch_names:
        for mch in montage_ch_names:
            if ch.upper() == mch.upper():
                channel_rename[ch] = mch
    return channel_rename


def load_reject_log(bids_path, duration=1, verbose=False):
    """Reload the Autoreject Log from disc.
    
    This assumes the Autoreject object was previously fit and then
    remake the reject log from the Epoched data.
    """
    from autoreject import read_auto_reject
    ar_bids_path = bids_path.copy().update(
        check=False,
        processing="autoreject",
        extension=".h5",
        suffix="desc-thresh0.95_eeg",
    )
    ar = read_auto_reject(ar_bids_path)

    # read in that file
    raw = read_raw_bids(bids_path)
    raw.set_montage("standard_1020")

    # make these into epochs
    epochs = make_fixed_length_epochs(raw, duration=duration)
    epochs.load_data()

    # get the reject log again
    reject_log = ar.get_reject_log(epochs)
    return reject_log


def map_rejectlog_to_deriv(
    deriv_onsets, winsize, rejectlog_events, rejectlog_duration
):
    """Map AR rejection log to derivative windowed analysis.

    Parameters
    ----------
    deriv_onsets : np.ndarray, shape (n_windows,)
        The onset in sample times of each derivative window.
    winsize : int
        The window size in samples of each derivative window.
    rejectlog_events : np.ndarray, shape (n_windows, 3)
        The array of events that were created by ``make_fixed_length_epochs``
        and then filtering by the ``bad_epochs`` of the ``RejectLog``.
    rejectlog_duration : int
        The duration in samples of each autoreject epoch.

    Returns
    -------
    [type]
        [description]
    """
    deriv_offsets = deriv_onsets + winsize

    bad_wins = []
    # loop through all bad epochs
    for onset in rejectlog_events[:, 0]:
        # now get the stop sample point for the bad epoch
        stop = onset + rejectlog_duration

        # find all derivative windows that are entirely
        # within the onset - stop bad epoch window
        #         bad_wins = []
        #         bad_wins.extend(
        #             np.argwhere((deriv_onsets >= onset) & (deriv_onsets <= stop).tolist())
        #         )

        # find all derivative windows that have any part
        # within the bad epoch
        bad_start_idx = np.argwhere((deriv_onsets >= onset) & (deriv_onsets <= stop))
        bad_end_idx = np.argwhere((deriv_offsets >= onset) & (deriv_offsets <= stop))
        bad_idx = np.unique(np.union1d(bad_start_idx, bad_end_idx))
        # bad_idx = np.intersect1d(bad_start_idx, bad_end_idx)
        bad_wins.extend(bad_idx.tolist())
    return bad_wins

def load_persyst_spikes(root, subjects=None, search_str="*.edf", verbose=True):
    """Read all persyst spikes."""
    if subjects is None:
        subjects = get_entity_vals(root, "subject")

    if verbose:
        print(f"Loading data for subjects: {subjects}")

    dataset = collections.defaultdict(list)
    for subject in subjects:
        subj_path = root / f"sub-{subject}"

        # get all files of certain search_str
        fpaths = subj_path.rglob(search_str)

        # now load in all file paths
        for fpath in fpaths:
            bids_path = get_bids_path_from_fname(fpath)
            raw = read_raw_bids(bids_path)

            # extract data
            ch_spike_df = compute_spike_df(raw)
            ch_spike_df["n_secs"] = raw.n_times / raw.info["sfreq"]
            ch_names = raw.ch_names

            dataset["data"].append(ch_spike_df)
            dataset["subject"].append(subject)
            dataset["bids_path"].append(bids_path)
            dataset["ch_names"].append(ch_names)
            dataset["roots"].append(root)
    return dataset


def load_derivative_heatmaps(
    deriv_path, search_str, read_func, subjects=None, verbose=True, **kwargs
) -> Dict:
    """Read all derived spatiotemporal heatmap in a derivative folder.

    Parameters
    ----------
    deriv_path : str | pathlib.Path
        The path to the derivatives. Inside this folder should
        be a folder for each subject, stored in BIDS format.
    search_str : str
        A regex string pattern to search for. This uniquely
        identifies the derivative file. E.g. '*.npy'
    read_func : function
        The reading function to load in one derived dataset.
    subjects : [type], optional
        Specific subjects to load, by default None, which loads all
        subjects found in ``deriv_path``.
    verbose : bool, optional
        verbosity, by default True
    """
    if subjects is None:
        subjects = get_entity_vals(deriv_path, "subject")

    if verbose:
        print(f"Loading data for subjects: {subjects} from {deriv_path}")

    # extract the BIDS root of the dataset
    this_path = deriv_path
    while this_path.parent.name != "derivatives" and this_path.parent.name != "/":
        this_path = this_path.parent
    root = this_path.parent.parent

    dataset = collections.defaultdict(list)
    for subject in subjects:
        subj_path = deriv_path / f"sub-{subject}"

        # get all files of certain search_str
        fpaths = subj_path.glob(search_str)

        # now load in all file paths
        for fpath in fpaths:
            entities = get_entities_from_fname(fpath.name, on_error='ignore')
            bids_path = BIDSPath(
                root=root, subject=entities['subject'],
                session=entities['session'],
                task=entities['task'],
                run=entities['run'],
                suffix='eeg',
                datatype='eeg',
                extension='.edf'
            )

            deriv = read_func(fpath, **kwargs)

            # extract data
            ch_names = deriv.ch_names
            if isinstance(deriv, EpochsTFR):
                deriv_data = deriv.data
                deriv_data = _preprocess_epochs_tfr(deriv_data)
            else:
                if "perturbmatrix" in fpath.name:
                    deriv.normalize()
                deriv_data = deriv.get_data()

            dataset["subject"].append(subject)
            dataset["data"].append(deriv_data)
            dataset["ch_names"].append(ch_names)
            dataset["roots"].append(root)
            dataset["bids_path"].append(bids_path)
    return dataset


def read_scalp_eeg(
    bids_path, reference, rereference=False, resample_sfreq=None, verbose=True
):
    # load in the data via mne-bids
    raw = read_raw_bids(bids_path)

    # should we resample?
    if resample_sfreq is not None:
        raw.resample(sfreq=resample_sfreq)
        print(f"New resampled sfreq: ", raw.info["sfreq"])
    # resample to 256 Hz if larger due to the fact that
    # scalp EEG can't really resolve frequencies > 90 Hz
    if raw.info["sfreq"] > 256:
        raw.resample(sfreq=256)
        print(f"New resampled sfreq: ", raw.info["sfreq"])

    # set channel types for scalp EEG data
    ch_types = {ch_name: _map_ch_to_type(ch_name) for ch_name in raw.ch_names}
    raw = raw.set_channel_types(ch_types)

    # try to get the best matching montage based on string matching of channels
    best_montage, montage_name = get_best_matching_montage(raw.ch_names)
    raw.rename_channels(_get_montage_rename_dict(raw.ch_names, best_montage.ch_names))

    # set montage and then convert all to upper-casing again
    raw.set_montage(best_montage, on_missing="warn")

    # get eeg channels that are not inside montage
    # assign them to bads, (or non-eeg) channels
    montage_chs = [ch.upper() for ch in best_montage.ch_names]
    not_in_montage_chs = [ch for ch in raw.ch_names if ch.upper() not in montage_chs]
    raw.info["bads"].extend(not_in_montage_chs)
    print(
        f"Dropping these channels that are not in {montage_name} montage: ",
        not_in_montage_chs,
    )

    # get additional reference channels
    # that needs to be hardcoded for 1020 montage
    ref_chs = []
    if montage_name == "standard_1020":
        if "A1" in raw.ch_names:
            ref_chs.append("A1")
        if "A2" in raw.ch_names:
            ref_chs.append("A2")
        # if 'AF3' in raw.ch_names:
        #     ref_chs.append('AF3')
        # if 'AF4' in raw.ch_names:
        #     ref_chs.append('AF4')
        for ch_name in raw.ch_names:
            if "Z" in ch_name.upper():
                ref_chs.append(ch_name)
        raw.load_data()
        raw.info["bads"].extend(ref_chs)

    # load data into RAM
    raw.load_data()

    if rereference:
        print(f"Re-referencing data to {ref_chs}")
        # set reference now and also pick the types afterwards
        raw.set_eeg_reference(ref_channels=ref_chs)

    # preprocess data
    # 1. filter data - bandpass [0.5, Nyquist]
    nyq_freq = raw.info["sfreq"] // 2 - 1
    line_freq = raw.info["line_freq"]
    l_freq = 1.0
    h_freq = min(30, nyq_freq)
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=verbose)

    # 2. filter data - notch filter [PowerLineFrequency]
    # usually 60 Hz in USA and 50 Hz in Europe
    # if h_freq > line_freq:
    #     freqs = np.arange(line_freq, nyq_freq, line_freq)
    #     raw.notch_filter(freqs=freqs, method="fir", verbose=verbose)

    # 3. set reference
    if reference == "average":
        raw.set_eeg_reference("average")

    return raw
