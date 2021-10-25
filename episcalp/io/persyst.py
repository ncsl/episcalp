from logging import warn
from mne_bids.path import get_bids_path_from_fname
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mne.utils import warn
from mne_bids import get_entities_from_fname, BIDSPath
from mne.annotations import Annotations
from mne_bids.read import read_raw_bids
from mne_bids.write import write_raw_bids

from ..montage import get_standard_1020_channels


def _process_annot(annot):
    # process annotations into a spike/spikegen event with
    # channel name annotated
    desc = annot.description[0]

    # Pattern is: <spike> <ch_name>-Av12
    ch_str = desc.split(" ")[1]
    ch_name = ch_str.split("-")[0]
    onset = annot.onset
    duration = annot.duration
    return desc, ch_name


def _get_spike_annots(raw, verbose=True):
    """Create Annotations object from Persyst spikes.

    Searches the raw object Annotations for hardcoded Persyst
    characters, such as: 'Spike ', or 'SpikeGen '.
    """
    onsets = []
    durations = []
    description = []
    ch_names = []

    standard_chs = get_standard_1020_channels()

    # loop through annotations
    for annot in raw.annotations:
        annot = Annotations(**annot)
        curr_desc = annot.description[0]
        if curr_desc.startswith("Spike ") or curr_desc.startswith("SpikeGen"):
            # or curr_desc.startswith('@Spike ') or curr_desc.startswith('@SpikeGen'):
            desc, ch_name = _process_annot(annot)

            # only standard_1020 montage is supported rn
            if ch_name not in standard_chs:
                continue
            # we want to remove midline contacts rn because
            # Bayview doesn't use them
            if ch_name in ["Cz", "Pz", "Fz"]:
                continue

            onsets.extend(annot.onset)
            durations.extend(annot.duration)
            description.append(desc)
            ch_names.append([ch_name])

    # create Annotations object
    spike_annots = Annotations(
        onset=onsets,
        duration=durations,
        description=description,
        ch_names=ch_names,
        orig_time=raw.info["meas_date"],
    )
    return spike_annots


def read_report(fname, root, overwrite=True):
    """Read Persyst generated spike report.

    Add persyst spike events as mne Annotations
    inside the raw path.

    Assumes:

    1. that the raw file exists at the corresponding BIDS path.
    2. the Persyst spike report filename is BIDS compliant.
    """
    df = pd.read_csv(fname, delimiter=",", index_col=None)

    # get the original BIDS dataset
    entities = get_entities_from_fname(fname)
    bids_path = BIDSPath(
        root=root,
        datatype="eeg",
        suffix="eeg",
        extension=".edf",
        **entities,
    )
    raw = read_raw_bids(bids_path)

    # remove the old spike annotations
    annotations = raw.annotations
    if overwrite:
        remove_idx = []
        for idx, annot in enumerate(annotations):
            descrip = annot["description"]
            if descrip.startswith("Spike ") or descrip.startswith("SpikeGen "):
                remove_idx.append(idx)
        annotations.delete(remove_idx)
        raw.set_annotations(annotations)

    # create onset, duration and description of spike events
    onsets = []
    durations = []
    descriptions = []

    # loop over each spike found
    for idx, row in df.iterrows():
        # convert Persyst storage of time to onset in seconds
        spike_time = row.get("Time")
        day, dt_time = spike_time.split(" ")
        hr, minute, sec = dt_time.split(":")
        onset_dt = raw.info["meas_date"]

        # first add the days
        if day == "d2":
            day = day + 1
            td = timedelta(days=1)
        else:
            td = timedelta(days=0)
        spike_onset_day_dt = onset_dt + td
        # next create datetime with the hours minutes seconds
        spike_onset_dt = datetime(
            year=spike_onset_day_dt.year,
            month=spike_onset_day_dt.month,
            day=spike_onset_day_dt.day,
            hour=int(hr),
            minute=int(minute),
            second=int(sec),
            tzinfo=timezone.utc,
        )

        onset = (spike_onset_dt - onset_dt).total_seconds()
        # print(row)
        print(onset)
        # assert False
        # get channel name
        ch_name = row.get("Channel")  # .split('-')[0]

        # get probability value
        y_pred_proba = row.get("Perception")

        # duration in ms -> seconds
        duration = row.get("Duration") / 1000.0

        # height of spike in uV
        height = row.get("Height")

        # format description of the annotation
        description = f"Spike {ch_name} perception:{y_pred_proba} height:{height}"

        onsets.append(onset)
        durations.append(duration)
        descriptions.append(description)

    # append the new spike annotations
    annots = Annotations(
        onset=onsets,
        duration=durations,
        description=descriptions,
        orig_time=raw.info["meas_date"],
    )
    annotations += annots

    # update the raw file
    raw.set_annotations(annotations)
    return raw


def compute_spike_df(raw):
    """Compute channel spike counts from Raw object."""
    raw.pick_types(eeg=True)
    raw.drop_channels(raw.info["bads"])
    ch_spike_count = dict()

    # extract spike annotations
    spike_annots = _get_spike_annots(raw)
    if len(spike_annots) == 0:
        for ch_name in raw.ch_names:
            ch_spike_count[ch_name] = 0
        return pd.DataFrame()

    # convert to dataframe and then count number of events
    spike_df = spike_annots.to_data_frame()

    meas_date = raw.info["meas_date"].replace(tzinfo=None)
    spike_df["onset"] = (spike_df["onset"] - meas_date).apply(
        lambda x: x.total_seconds()
    )
    spike_df["sample"] = (
        (spike_df["onset"]).apply(lambda x: x * raw.info["sfreq"]).astype(int)
    )

    # convert ch name column to single str
    spike_df["ch_name"] = spike_df["ch_names"].apply(lambda x: x[0])
    spike_df.drop("ch_names", axis=1, inplace=True)

    # extract duration and perception from the spike df description
    descriptions = spike_df["description"]

    if any("perception:" not in descrip for descrip in descriptions):
        warn(
            f"Not all descriptions have the perception: string. Check "
            f"the persyst annootations for {raw}."
        )

    perceptions = []
    heights = []
    for descrip in descriptions:
        if "perception:" not in descrip:
            raise RuntimeError(f"Perception not in {raw} spike annotations.")

        # extract perception
        perception = descrip.split("perception:")[1].split(" ")[0]

        # extract height in uV
        height = descrip.split("height:")[1].split(" ")[0]

        perceptions.append(perception)
        heights.append(height)
    spike_df["perception"] = perceptions
    spike_df["height"] = heights
    return spike_df


if __name__ == "__main__":
    root = Path("/Users/adam2392/Johns Hopkins/Scalp EEG JHH - Documents/bids/")
    fname = (
        root
        / "derivatives"
        / "spike_reports"
        / "sub-jhh201"
        / "sub-jhh201_run-01_spikereview.csv"
    )
    raw = read_report(fname, root=root)

    print(raw.annotations.to_data_frame())
    bids_path = get_bids_path_from_fname(raw.filenames[0])
    write_raw_bids(raw, bids_path, format="EDF", overwrite=True)
