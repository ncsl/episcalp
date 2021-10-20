import logging
from pathlib import Path
import numpy as np
import pandas as pd

from mne_bids.path import get_entities_from_fname
from mne_bids import BIDSPath, get_entity_vals
from mne import make_fixed_length_epochs
from mne_connectivity import (
    spectral_connectivity,
    envelope_correlation,
    phase_slope_index,
)

import sys
import os

sys.path.append("../../")
sys.path.append("./episcalp")
print(os.getcwd())
from episcalp.io import read_scalp_eeg


def main_spectral():
    # root = Path("/Users/adam2392/Johns Hopkins/Scalp EEG JHH - Documents/bids/")
    root = Path("/Users/adam2392/Johns Hopkins/Jefferson_Scalp - Documents/root/")
    deriv_root = root / "derivatives"
    bids_root = deriv_root / "ICA" / "1-30Hz-30" / "win-20"

    figures_path = deriv_root

    # define BIDS entities
    IGNORE_SUBJECTS = []

    datatype = "eeg"
    extension = ".edf"
    n_jobs = -1
    # analysis parameters
    reference = "monopolar"
    sfreq = None
    overwrite = False

    # get the runs for this subject
    all_subjects = get_entity_vals(bids_root, "subject")

    for subject in all_subjects:
        if subject in IGNORE_SUBJECTS:
            continue

        ignore_subs = [sub for sub in all_subjects if sub != subject]
        subj_dir = bids_root / f"sub-{subject}"

        fpaths = list(subj_dir.rglob("*.edf"))

        print(f"Found filepaths for {subject}: {fpaths}.")

        for idx, fpath in enumerate(fpaths):
            entities = get_entities_from_fname(fpath.name)

            # create path for the dataset
            bids_path = BIDSPath(
                **entities,
                datatype=datatype,
                root=bids_root,
                extension=extension,
            )
            print(f"Analyzing {bids_path}")

            raw = read_scalp_eeg(bids_path, reference=reference)

            # run TFR analysis
            epochs = make_fixed_length_epochs(raw, duration=2, overlap=1)

            # compute connectivity
            sfreq = raw.info["sfreq"]

            # compute envelope conn
            env_conn = envelope_correlation(epochs, names=epochs.ch_names)
            method = "envcorr"
            deriv_chain = Path("conn") / method
            deriv_fpath = (
                deriv_root
                / deriv_chain
                / f"sub-{subject}"
                / bids_path.copy()
                .update(check=False, suffix=f"desc-{method}_eeg", extension=".nc")
                .basename
            )
            deriv_fpath.parent.mkdir(exist_ok=True, parents=True)
            if deriv_fpath.exists() and not overwrite:
                continue
            env_conn.save(deriv_fpath)

            # compute spectral conn
            # method = ['coh', 'cohy', 'imcoh', 'plv', 'ciplv', 'ppc',
            #             'pli', 'wpli', 'wpli2_debiased']
            # spectral_conn = spectral_connectivity(epochs, names=epochs.ch_names,
            #     method=method, sfreq=sfreq)

            # for method, conn in zip(method, spectral_conn):
            #     deriv_chain = Path("conn") / method
            #     deriv_fpath = (
            #         deriv_root
            #         / deriv_chain
            #         / f"sub-{subject}"
            #         / bids_path.copy()
            #         .update(
            #             check=False, suffix=f"desc-{method}_eeg", extension=".nc"
            #         )
            #         .basename
            #     )
            #     deriv_fpath.parent.mkdir(exist_ok=True, parents=True)
            #     if deriv_fpath.exists() and not overwrite:
            #         continue

            #     # save the output object to disc
            #     print(f"Saving to {deriv_fpath}")
            #     conn.save(deriv_fpath)


if __name__ == "__main__":
    main_spectral()
