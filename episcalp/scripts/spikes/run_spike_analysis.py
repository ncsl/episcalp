import logging
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from mne.utils import warn
from mne_bids import BIDSPath, get_entity_vals, read_raw_bids
import json


def run_spike_analysis(
    source_path,
    subject,
    deriv_path=None,
    figures_path=None,
    reference="monopolar",
    fill=True,
    prune=True,
    save_json=True,
    **model_params,
):
    # get the root derivative path
    deriv_chain = Path("spikes") / reference / f"sub-{subject}"
    output_basename = source_path.name
    output_fname = Path(f"{output_basename}_spikes").with_suffix(".json")
    output_fpath = deriv_path / deriv_chain / output_fname

    raw = mne.io.read_raw_persyst(source_path)
    annotations = raw.annotations
    mins = raw.n_times / (60 * raw.info["sfreq"])
    n_chs = raw.info["nchan"]

    spike_channels = [
        a["description"].split(" ")[1]
        for a in annotations
        if "spike" in a["description"]
    ]
    spike_channels = [s.split("-")[0] for s in spike_channels]
    spike_counts = _count_spikes(spike_channels)
    if fill:
        for ch in raw.ch_names:
            if not spike_counts.get(ch.lower(), None):
                spike_counts[ch.lower()] = 0
    if prune:
        ch_names_upper = [ch.upper() for ch in raw.ch_names]
        bad_keys = []
        bad_keys_ = _get_non_montage_chs(raw.ch_names)
        for key in spike_counts.keys():
            if key.upper() not in ch_names_upper:
                bad_keys.append(key)
            if key.upper() in bad_keys_:
                bad_keys.append(key)
        for bk in bad_keys:
            del spike_counts[bk]

    spike_rates = {}
    for key, val in spike_counts.items():
        spike_rates[key] = val / mins
    total_spike_rate = sum(spike_counts.values()) / (
        n_chs * mins
    )  # normalizing by total time and number of channels
    spike_rates["total"] = total_spike_rate

    if save_json:
        with open(output_fpath, "w+") as fid:
            json.dump(spike_rates, fid, indent=4)
    return spike_rates


def _count_spikes(spike_list):
    spike_dict = {}
    for ch in list(set(spike_list)):
        spike_dict.update({ch: spike_list.count(ch)})
    return spike_dict


def _get_non_montage_chs(ch_list):
    standard_1020_chs = [
        "Fp1",
        "Fp2",
        "F3",
        "F4",
        "F7",
        "F8",
        "T3",
        "T4",
        "C3",
        "C4",
        "T5",
        "T6",
        "P3",
        "P4",
        "O1",
        "O2",
        "T7",
        "T8",
        "P7",
        "P8",
    ]
    montage_upper = [c.upper() for c in standard_1020_chs]
    bad_chs = []
    for ch in ch_list:
        if ch.upper() not in montage_upper:
            bad_chs.append(ch)
    return bad_chs


if __name__ == "__main__":
    root = Path(
        "D:/OneDriveParent/OneDrive - Johns Hopkins/Shared Documents/40Hz-30/derivatives/spikes"
    )
    lay_fpaths = [f for f in root.rglob("*.lay")]
    subjects = [f.name.split("_")[0].split("-")[1] for f in lay_fpaths]
    deriv_path = Path(
        "D:/OneDriveParent/OneDrive - Johns Hopkins/Shared Documents/40Hz-30/derivatives"
    )
    for subject, lay_fpath in zip(subjects, lay_fpaths):
        run_spike_analysis(lay_fpath, subject, deriv_path)
