import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path

from mne_bids import get_entities_from_fname, BIDSPath
from mne.annotations import Annotations
from mne_bids.read import read_raw_bids
from mne_bids.write import write_raw_bids


def read_report(fname, root, overwrite=True):
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
    print(raw.annotations.to_data_frame())
    write_raw_bids(raw, bids_path)


if __name__ == "__main__":
    root = Path("/Users/adam2392/Johns Hopkins/Scalp EEG JHH - Documents/bids/")
    fname = (
        root
        / "derivatives"
        / "spike_reports"
        / "sub-jhh001"
        / "sub-jhh001_run-01_spikereview.csv"
    )
    read_report(fname, root=root)
