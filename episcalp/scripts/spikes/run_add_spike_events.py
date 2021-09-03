from pathlib import Path
import numpy as np

import mne
from mne_bids import BIDSPath, get_entity_vals, read_raw_bids
from mne_bids.write import write_raw_bids

from episcalp.utils.standard_1020_montage import get_standard_1020_montage


def _fix_channel_name(ch_name):
    base_ch, ref_ch = ch_name.split("-")
    standard_chs = get_standard_1020_montage()
    for ch in standard_chs:
        if base_ch.upper() == ch.upper():
            base_ch = ch
    return f"{base_ch}-{ref_ch}"


def _extract_spike_annotations(raw_persyst):
    # Duration for spikes is for some reason encoded in the description and not in the duration section
    # Iterate and move the duration to the correct place
    annotations = raw_persyst.annotations
    num_comments = len(annotations)
    onsets = np.zeros(num_comments, float)
    durations = np.zeros(num_comments, float)
    descriptions = [""] * num_comments
    for ind, annot in enumerate(annotations):
        onsets[ind] = annot.get("onset", 0)
        description_ = annot.get("description", "")
        if description_.startswith("spike"):
            spike, ch, duration = description_.split(" ")

            # found a spike burst which is labeled differently in Persyst...
            if duration.endswith("s"):
                duration = duration[:-1]
                durations[ind] = float(duration)
                description = f"{spike} {ch}"
                descriptions[ind] = description
            else:
                durations[ind] = float(duration)
                description = f"{spike} {_fix_channel_name(ch)}"
                descriptions[ind] = description
        else:
            descriptions[ind] = description_
    # Add correct annotations to the base raw object
    annotations_ = mne.Annotations(
        onsets, durations, descriptions, orig_time=raw_persyst.info["meas_date"]
    )
    return annotations_


def add_spikes_jhh():
    root = Path("/Users/adam2392/Johns Hopkins/Scalp EEG JHH - Documents/bids")
    spike_root = root / "derivatives" / "spikes" / "monopolar"
    site_id = "jhh"

    datatype = "eeg"
    suffix = "eeg"

    # get all subjects
    subjects = get_entity_vals(spike_root, "subject")
    subjects = [f"{site_id}{subject}" for subject in subjects]

    for subject in subjects:
        bids_path_ = BIDSPath(
            subject=subject, root=root, datatype=datatype, suffix=suffix
        )

        fpaths = bids_path_.match(check=True)
        if fpaths == []:
            raise RuntimeError(f"The parameters {bids_path_} do not match anything")

        sub = subject.split(site_id)[1]
        spike_sub = f"sub-{sub}"
        # read the original raw file
        for bids_path in fpaths:
            raw_src = read_raw_bids(bids_path)
            run = bids_path.run

            # Read in persyst data that contains spike information
            spike_dir = Path(spike_root) / spike_sub
            task = bids_path.task

            spike_fpath = list(spike_dir.glob(f"{spike_sub}_*run-{run}_spikes.lay"))
            if len(spike_fpath) > 1 or len(spike_fpath) == 0:
                if task == "asleep":
                    task = "sleep"
                search_str = f"{spike_sub}_*{task}*_spikes.lay"
                spike_fpath = list(spike_dir.glob(search_str))
                if len(spike_fpath) > 1:
                    assert False

            spike_fpath = spike_fpath[0]
            raw_persyst = mne.io.read_raw_persyst(spike_fpath)

            # extract spike annotations
            spike_annotations = _extract_spike_annotations(raw_persyst)

            # add them to the raw file
            annotations = raw_src.annotations

            if any([annots in annotations for annots in spike_annotations]):
                print(f"Already there for {bids_path}")
                continue

            annotations = annotations + spike_annotations
            raw_src.set_annotations(annotations)

            # write to BIDS and overwrite existing dataset
            write_raw_bids(raw_src, bids_path, overwrite=True, format="EDF")


if __name__ == "__main__":
    add_spikes_jhh()
