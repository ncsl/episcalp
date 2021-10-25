import logging
from pathlib import Path
from mne_bids.path import get_bids_path_from_fname
import numpy as np
import pandas as pd
import mne
from mne import make_fixed_length_epochs
from mne.utils import warn
from mne_bids import BIDSPath, get_entity_vals, read_raw_bids, get_bids_path_from_fname
from autoreject import AutoReject, set_matplotlib_defaults
import matplotlib.pyplot as plt

from eztrack.utils import logger

import sys
import os

logger.setLevel(logging.DEBUG)


def main():
    root = Path("/Users/adam2392/Johns Hopkins/Scalp EEG JHH - Documents/bids/")
    # root = Path("/Users/adam2392/Johns Hopkins/Jefferson_Scalp - Documents/root/")
    deriv_root = root / "derivatives"
    bids_root = deriv_root / "ICA" / "1-30Hz-30" / "win-20"

    # define BIDS entities
    IGNORE_SUBJECTS = []
    extension = ".edf"

    # analysis parameters
    n_jobs = -1
    consensus = 0.8
    reference = "monopolar"
    overwrite = False

    # get the runs for this subject
    all_subjects = get_entity_vals(bids_root, "subject")

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
                suffix="eeg",
            )
            fig_bids_path = ar_bids_path.copy().update(extension=".pdf")
            if ar_bids_path.fpath.exists() and not overwrite:
                continue

            # read in that file
            raw = read_raw_bids(bids_path)
            raw.set_montage("standard_1020")

            # make these into epochs
            epochs = make_fixed_length_epochs(raw, duration=1)
            epochs.load_data()

            ar.fit(epochs)
            print(ar_bids_path)
            ar.save(fname=ar_bids_path, overwrite=overwrite)
            _, reject_log = ar.transform(epochs, return_log=True)
            set_matplotlib_defaults(plt)
            fig = reject_log.plot(orientation="horizontal", show=False)
            fig.savefig(fig_bids_path)


if __name__ == "__main__":
    main()
