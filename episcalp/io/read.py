import numpy as np
from mne_bids import read_raw_bids

from episcalp.utils import (
    get_best_matching_montage,
)


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
        raw.resample(sfreq=resample_sfreq)
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
    h_freq = min(55, nyq_freq)
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=verbose)

    # 2. filter data - notch filter [PowerLineFrequency]
    # usually 60 Hz in USA and 50 Hz in Europe
    if h_freq > line_freq:
        freqs = np.arange(line_freq, nyq_freq, line_freq)
        raw.notch_filter(freqs=freqs, method="fir", verbose=verbose)

    # 3. set reference
    if reference == "average":
        raw.set_eeg_reference("average")

    return raw
