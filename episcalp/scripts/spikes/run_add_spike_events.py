from pathlib import Path
from mne_bids.path import get_entities_from_fname
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

        # contains the @ and (Peryst) symbol - only if Persyst ran by
        # clinicians in their own hospital
        # contains_spike = ((description_.startswith("@Spike "))
        #                   or (description_.startswith("@SpikeGen")))

        if (
            (description_.startswith("Spike "))
            or (description_.startswith("SpikeGen"))
            and onsets[ind] != 0
        ):
            if description_.count(" ") == 2:
                spike, ch, duration = description_.split(" ")
            else:
                continue
                raise RuntimeError(
                    f"Description: {description_} has more then 2 spaces..."
                )

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


def find_nearest(array, value):
    array = np.asarray(array)
    idx = np.argsort(abs(array - value))[0]
    return array[idx]


def add_spikes_for_sites():
    root = Path("/Users/adam2392/Johns Hopkins/Scalp EEG JHH - Documents/bids")
    root = Path("/Users/adam2392/Johns Hopkins/Jefferson_Scalp - Documents/root/")
    spike_root = root / "derivatives" / "spikes"

    extension = ".edf"
    datatype = "eeg"
    suffix = "eeg"

    # get all subjects
    subjects = get_entity_vals(spike_root, "subject")
    # subjects = [f"{site_id}{subject}" for subject in subjects]
    # subjects = ['jhh001']

    for subject in subjects:
        bids_path_ = BIDSPath(
            subject=subject,
            root=root,
            datatype=datatype,
            suffix=suffix,
            extension=extension,
        )

        fpaths = bids_path_.match(check=True)
        if fpaths == []:
            raise RuntimeError(f"The parameters {bids_path_} do not match anything")

        spike_sub = f"sub-{subject}"
        # read the original raw file
        for bids_path in fpaths:
            print(f"\n\nAdding spikes for {bids_path}")

            raw_src = read_raw_bids(bids_path, verbose=False)
            annotations = raw_src.annotations
            run = bids_path.run

            # Read in persyst data that contains spike information
            spike_dir = Path(spike_root) / spike_sub
            task = bids_path.task

            spike_fpaths = list(spike_dir.glob("*.lay"))
            for fpath in spike_fpaths:
                entities = get_entities_from_fname(fpath.name)
                if all(
                    ent == bids_path.entities[key]
                    for key, ent in entities.items()
                    if key != "suffix"
                ):
                    spike_fpath = [fpath.as_posix()]
                    break

            # spike_fpath = list(spike_dir.glob(
            #     f"{spike_sub}_*run-{run}_spikes.lay"))

            print(f"Initially found paths: {spike_fpath}.")
            if len(spike_fpath) > 1 or len(spike_fpath) == 0:
                if task == "asleep":
                    task = "sleep"
                print(f"Re-searching with task. {spike_fpath}")
                print(f"Search string was: {spike_sub}_*run-{run}_eeg_spikes.lay")
                search_str = f"{spike_sub}_*{task}*_spikes.lay"
                spike_fpath = list(spike_dir.glob(search_str))
                if len(spike_fpath) > 1:
                    assert False
            if spike_fpath == []:
                raise RuntimeError(
                    f"No spike files found for {bids_path} in {spike_dir}."
                )

            spike_fpath = spike_fpath[0]
            raw_persyst = mne.io.read_raw_persyst(spike_fpath)

            # extract spike annotations
            spike_annotations = _extract_spike_annotations(raw_persyst)

            # add them to the raw file
            # Note: Does not support writing the channel name to annotation
            # as of MNE v0.23+
            wrote_annotations = False
            for idx, annot in enumerate(spike_annotations):
                # only looking at spikes
                if "spike" not in annot["description"].lower():
                    continue

                # print(annot['description'], annot['onset'])
                # print(annotations.description)
                # print(annotations.onset)
                nearest_onset = find_nearest(annotations.onset, annot["onset"])
                if annot["description"] in annotations.description and (
                    (annot["onset"] - nearest_onset) < 0.01
                ):
                    print("\nskipping...")
                    continue
                    # if annot['description'].startswith('@Warning'):
                    #     continue
                    # print(f'Appending {annot}')
                    # print(annotations.description)
                    # assert False
                annotations.append(
                    annot["onset"], annot["duration"], annot["description"]
                )
                wrote_annotations = True

            if (
                not wrote_annotations
            ):  # any([annots in annotations for annots in spike_annotations]):
                print(f"Already there for {bids_path}")
                continue

            # annotations = annotations + spike_annotations
            raw_src.set_annotations(annotations)

            # write to BIDS and overwrite existing dataset
            print("Writing new spike events to {bids_path}")
            write_raw_bids(
                raw_src, bids_path, overwrite=True, format="EDF", verbose=False
            )


if __name__ == "__main__":
    add_spikes_for_sites()
