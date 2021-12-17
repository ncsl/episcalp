import logging
from pathlib import Path
from mne_bids.path import get_bids_path_from_fname
from mne_bids.write import write_raw_bids
import numpy as np
import pandas as pd
import mne
from mne import make_fixed_length_epochs
from mne.utils import warn
from mne.preprocessing import annotate_muscle_zscore

from mne_bids import BIDSPath, get_entity_vals, read_raw_bids, get_bids_path_from_fname
from autoreject import AutoReject, set_matplotlib_defaults
import matplotlib.pyplot as plt

from eztrack.utils import logger

from episcalp.montage import get_standard_1020_channels

import sys
import os

logger.setLevel(logging.DEBUG)


def annotate_muscle_artifacts(raw, l_freq, h_freq, bids_path):
    """Annotate automatically muscle artifacts using frequency power.

    NOTE WE WILL DROP ALL EXISTING ANNOTATIONS CURRENTLY.

    Parameters
    ----------
    raw : [type]
        [description]
    l_freq : [type]
        [description]
    h_freq : [type]
        [description]
    bids_path : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    # drop all bad channels
    raw = raw.drop_channels(raw.info['bads'])
    orig_raw = raw.copy()
    raw.load_data()

    # get standard montage channels
    montage_chs = get_standard_1020_channels()

    drop_chs = [ch for ch in raw.ch_names if ch not in montage_chs]
    raw = raw.drop_channels(drop_chs)

    # now annotate high frequency muscle activity
    raw.set_annotations(None)
    muscle_annots, scores = annotate_muscle_zscore(raw, ch_type='eeg',
            filter_freq=[l_freq, h_freq], threshold=5)
    
    # add annotations to raw
    annots = raw.annotations
    annots += muscle_annots
    orig_raw.set_annotations(annots)

    # write to disc
    bids_path = write_raw_bids(orig_raw, bids_path, format='EDF', overwrite=True)
    return bids_path


def main():
    root = Path("/Users/adam2392/Johns Hopkins/Scalp EEG JHH - Documents/bids/")
    # root = Path("/Users/adam2392/Johns Hopkins/Jefferson_Scalp - Documents/root/")
    deriv_root = root / "derivatives"
    bids_root = deriv_root / "ICA" / "1-90Hz-30" / "win-20"

    # define BIDS entities
    IGNORE_SUBJECTS = []
    extension = ".edf"

    # analysis parameters
    n_jobs = -1
    consensus = 0.95
    reference = "monopolar"
    overwrite = False

    l_freq = 30
    h_freq = 90

    # get the runs for this subject
    all_subjects = get_entity_vals(bids_root, "subject")

    # the autoreject algorithm without interpolation and instantiating a 
    # threshold for number of bad electrodes required for consensus as a bad epoch
    ar = AutoReject(n_interpolate=[0], consensus=[consensus], n_jobs=n_jobs)

    for subject in all_subjects:
        if subject in IGNORE_SUBJECTS:
            continue
        subj_dir = bids_root / f"sub-{subject}"

        fpaths = subj_dir.rglob(f"*{extension}")

        for fpath in fpaths:
            bids_path = get_bids_path_from_fname(fpath)
            print(f"Analyzing {bids_path}")

            ar_bids_path = bids_path.copy().update(
                check=False,
                processing="autoreject",
                extension=".h5",
                suffix=f"desc-thresh{consensus}_eeg",
            )
            fig_bids_path = ar_bids_path.copy().update(extension=".pdf")
            if ar_bids_path.fpath.exists() and not overwrite:
                continue

            # read in that file
            raw = read_raw_bids(bids_path)
            raw.set_montage("standard_1020")

            bids_path = annotate_muscle_artifacts(raw.copy(), 
                l_freq, h_freq, bids_path)

            # make these into epochs
            epochs = make_fixed_length_epochs(raw, duration=1)
            epochs.load_data()

            # run autoreject and compute a threshold and bad channels per epoch
            ar.fit(epochs)
            print(ar_bids_path)
            ar.save(fname=ar_bids_path, overwrite=overwrite)
            _, reject_log = ar.transform(epochs, return_log=True)
            set_matplotlib_defaults(plt)
            fig = reject_log.plot(orientation="horizontal", show=False)
            fig.savefig(fig_bids_path)


if __name__ == "__main__":
    main()
