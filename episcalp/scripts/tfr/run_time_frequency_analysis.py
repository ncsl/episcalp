import logging
from pathlib import Path
from mne.time_frequency.tfr import tfr_multitaper
from mne_bids.path import get_entities_from_fname
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, get_entity_vals
from mne.time_frequency import stft
from mne import make_fixed_length_epochs

from eztrack.utils import logger

import sys
import os

sys.path.append("../../")
sys.path.append("./episcalp")
print(os.getcwd())
from episcalp.io import read_scalp_eeg

logger.setLevel(logging.DEBUG)


def main():
    root = Path("/Users/adam2392/Johns Hopkins/Scalp EEG JHH - Documents/bids/")
    deriv_root = root / "derivatives"
    bids_root = deriv_root / 'ICA' / '1-30Hz-30' / 'win-20'

    figures_path = deriv_root

    # define BIDS entities
    IGNORE_SUBJECTS = []
    FREQ_BANDS = {
        'delta': np.arange(1, 4, 0.5),
        'theta': np.arange(4, 8, 0.5),
        'alpha': np.arange(8, 12, 0.5),
        'beta': np.arange(12, 30, 1)
    }
    freq_band = 'delta'

    deriv_chain = Path('tfr') / freq_band

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

        fpaths = list(subj_dir.rglob('*.edf'))

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
            deriv_fpath = deriv_root / deriv_chain / f'sub-{subject}' / bids_path.copy().update(
                check=False, suffix=f'desc-{freq_band}_eeg-tfr', extension='.h5').basename
            deriv_fpath.parent.mkdir(exist_ok=True, parents=True)
            if deriv_fpath.exists() and not overwrite:
                continue

            raw = read_scalp_eeg(bids_path, reference=reference)

            # run TFR analysis
            epochs = make_fixed_length_epochs(raw, duration=2, overlap=1)
            freqs = FREQ_BANDS[freq_band]

            # compute multitaper FFT
            epochs_tfr  = tfr_multitaper(epochs, freqs, n_jobs=n_jobs, n_cycles = freqs/2., return_itc=False, average=False)
            
            # save the output object to disc        
            print(f'Saving to {deriv_fpath}')
            epochs_tfr.save(deriv_fpath)


if __name__ == "__main__":
    main()
