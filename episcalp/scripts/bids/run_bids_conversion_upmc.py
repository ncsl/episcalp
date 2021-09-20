"""Run this file for converting Jefferson scalp EEG data."""
from pathlib import Path
import pandas as pd
import re

from mne_bids.path import BIDSPath

from episcalp.bids.bids_conversion import (
    write_epitrack_bids,
    append_original_fname_to_scans,
)
from episcalp.bids.utils import update_participants_info


def convert_upmc_to_bids():
    root = Path("D:/OneDriveParent/Johns Hopkins/UPMC_Scalp - Documents/scalp_study/root")
    source_root = root / "sourcedata"

    # epilepsy source data from UPMC is organized into three sub-folders
    # 0xx is non epilepsy
    # 1xx is epilepsy normal
    # 2xx is epilepsy abnormal
    non_epi_src = source_root / "non-epilepsy-normal-EEG"
    epi_norm_src = source_root / "epilepsy-wout-abnormalities"
    epi_abnorm_src = source_root / "epilepsy-with-abnormalities"

    # site ID prefix to subject IDs
    subj_site = "upmc"

    # define BIDS identifiers
    datatype = "eeg"

    montage = "standard_1020"
    format = "EDF"
    line_freq = 60
    overwrite = False
    verbose = False

    for source_dir in [non_epi_src, epi_norm_src, epi_abnorm_src]:
        _convert_folder(
            source_dir,
            root,
            subj_site,
            datatype,
            montage,
            format,
            line_freq,
            overwrite,
            verbose,
        )


def _map_subject_to_exp(subject, source_root):
    excel_fpath = source_root / "UPMC_scalp_clinical_datasheet.xlsx"
    metadata = pd.read_excel(Path(excel_fpath))

    sub_row = metadata.loc[metadata["hospital_id"] == int(subject)].iloc[0, :]

    new_id = str(sub_row["patient_id"]).zfill(3)
    return new_id


def _convert_folder(
    source_dir,
    root,
    subj_site,
    datatype,
    montage,
    format,
    line_freq,
    overwrite,
    verbose,
):
    subjects = []
    for idx, fpath in enumerate(source_dir.glob("*.edf")):
        if " " in fpath.name:
            # source subject ID
            subject_id = fpath.name.split(" ")[0]
        else:
            subject_id = fpath.name.split(".")[0]
        subjects.append(subject_id)

    # run BIDS conversion for each set of files
    for subject in subjects:
        fpaths = [fpath for fpath in source_dir.glob("*.edf")]
        fpaths = [f for f in fpaths if f.name.split(".")[0] == subject]

        og_subject = subject
        subject = _map_subject_to_exp(subject, root / "sourcedata")

        # get experimental condition
        if subject.startswith("0"):
            exp_condition = "non-epilepsy-normal-eeg"
        elif subject.startswith("1"):
            exp_condition = "epilepsy-wout-abnormalities"
        elif subject.startswith("2"):
            exp_condition = "epilepsy-with-abnormalities"
        else:
            raise RuntimeError(f"There is an issue with this subject  {subject}")

        # prefix it with site id
        subject = f"{subj_site}{subject}"

        for idx, fpath in enumerate(fpaths):

            site_id = 'upmc'

            run_id = idx + 1
            bids_kwargs = {
                "subject": subject,
                "run": run_id,
                "datatype": datatype,
                "suffix": datatype,
            }
            bids_path = BIDSPath(**bids_kwargs, root=root)
            if bids_path.fpath.exists() and not overwrite:
                print("Skipping...", bids_path)
                continue
            bids_path = write_epitrack_bids(
                source_path=fpath,
                bids_path=bids_path,
                montage=montage,
                format=format,
                line_freq=line_freq,
                overwrite=overwrite,
                verbose=verbose,
            )

            # update the participants tsv file
            update_participants_info(
                root,
                subject,
                "site",
                site_id,
                description="UPMC",
                levels=None,
                units=None,
            )

            update_participants_info(
                root,
                subject,
                "exp_condition",
                exp_condition,
                description="Non-epilepsy with normal EEG, epilepsy with normal EEG, or epilepsy with abnormal EEG",
                levels=None,
                units=None,
            )

            update_participants_info(
                root,
                subject,
                "orig_sub_id",
                og_subject,
                description="Original subject ID provided by the clinician",
                levels=None,
                units=None,
            )

            # add original scan name to scans.tsv
            append_original_fname_to_scans(fpath.name, root, bids_path.basename)


if __name__ == "__main__":
    convert_upmc_to_bids()
