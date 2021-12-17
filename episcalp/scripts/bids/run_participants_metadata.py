from pathlib import Path
import pandas as pd
import json


from episcalp.bids.utils import update_participants_info

from mne_bids import get_entity_vals


def pipeline_jhh():
    bids_root = Path("D:/OneDriveParent/OneDrive - Johns Hopkins/Shared Documents/bids")
    bids_root = Path("/Users/adam2392/Johns Hopkins/Scalp EEG JHH - Documents/bids/")
    source_dir = bids_root / "sourcedata"

    site_pref = "jhh"
    excel_metadata_fpath = source_dir / "JHU_scalp_clinical_datasheet_raw_local.xlsx"

    # read in metadata as dataframe
    meta_df = pd.read_excel(excel_metadata_fpath)

    columns = {
        "outcome": "0 is non epilepsy, 1 is epilepsy normal with normal EEG and 2 is epilepsy with abnormal EEG",
        # "epileptiform EEG": "Whether or not there was epileptoform activity in EEG",
        "num_AEDs": "Number of Anti-epileptic drugs during first EEG",
        "final_diagnosis": "What was the final clinical diagnosis",
        "epilepsy_type": "Focal or generalized epilepsy",
        "epilepsy_hemisphere": "Right, left, or bihemispheric",
        "epilepsy_lobe": "For focal epilepsy, which lobes it occurs in.",
        "age": "",
        "sex": "",
        "hand": "",
        "exp_condition": "Non-epilepsy with normal EEG, epilepsy with normal EEG, or epilepsy with abnormal EEG",
    }

    subjects = get_entity_vals(bids_root, "subject")
    print("All the subjects are: ", subjects)

    for idx, row in meta_df.iterrows():
        record_id = row["hospital_id"]
        site = row["CLINICAL_CENTER"]
        if site == "jhh":
            site = "JHH"
        else:
            site = "BV"
        subject = str(row["patient_id"]).zfill(3)
        if subject.startswith("0"):
            outcome = 0
            exp_condition = "non-epilepsy-normal-eeg"
        elif subject.startswith("1"):
            outcome = 1
            exp_condition = "epilepsy-normal-eeg"
        elif subject.startswith("2"):
            outcome = 2
            exp_condition = "epilepsy-abnormal-eeg"

        if site_pref not in str(subject):
            subject = f"{site_pref}{subject}"

        if subject not in subjects:
            print(f"Skipping {subject}")
            continue

 
        _update_site(bids_root, subject, site)

        # for key, description in columns.items():
        #     value = row[key]
        #     if pd.isnull(value):
        #         value = "n/a"
        #     update_participants_info(
        #         root=bids_root,
        #         subject=subject,
        #         key=key,
        #         value=value,
        #         description=description,
        #     )
        for key, description in columns.items():
            if key == "outcome":
                update_participants_info(
                    root=bids_root,
                    subject=subject,
                    key=key,
                    value=outcome,
                    description=description,
                )
            elif key == 'exp_condition':
                update_participants_info(
                    root=bids_root,
                    subject=subject,
                    key=key,
                    value=exp_condition,
                    description=description,
                )
            else:
                value = row[key]
                if pd.isnull(value):
                    value = "n/a"
                if isinstance(value, str):
                    value = value.lower()
                update_participants_info(
                    root=bids_root,
                    subject=subject,
                    key=key,
                    value=value,
                    description=description,
                )


def pipeline_jefferson():
    bids_root = Path("D:/OneDriveParent/OneDrive - Johns Hopkins/Shared Documents/bids")
    bids_root = Path("/Users/adam2392/Johns Hopkins/Jefferson_Scalp - Documents/root/")
    source_dir = bids_root / "sourcedata"

    site = "jeff"
    excel_metadata_fpath = source_dir / "Jefferson_scalp_clinical_datasheet.xlsx"

    # read in metadata as dataframe
    meta_df = pd.read_excel(excel_metadata_fpath)

    columns = {
        "outcome": "0 is non epilepsy, 1 is epilepsy normal with normal EEG and 2 is epilepsy with abnormal EEG",
        # "epileptiform EEG": "Whether or not there was epileptoform activity in EEG",
        "num_AEDs": "Number of Anti-epileptic drugs during first EEG",
        # "final_diagnosis": "What was the final clinical diagnosis",
        "epilepsy_type": "Focal or generalized epilepsy",
        "epilepsy_hemisphere": "Right, left, or bihemispheric",
        "epilepsy_lobe": "For focal epilepsy, which lobes it occurs in.",
        # "age": "",
        # "sex": "",
        # "hand": "",
        "exp_condition": "Non-epilepsy with normal EEG, epilepsy with normal EEG, or epilepsy with abnormal EEG",
    }

    subjects = get_entity_vals(bids_root, "subject")
    print("All the subjects are: ", subjects)

    for idx, row in meta_df.iterrows():
        record_id = row["hospital_id"]

        subject = str(row["patient_id"]).zfill(3)
        if subject.startswith("0"):
            outcome = 0
            exp_condition = "non-epilepsy-normal-eeg"
        elif subject.startswith("1"):
            outcome = 1
            exp_condition = "epilepsy-normal-eeg"
        elif subject.startswith("2"):
            outcome = 2
            exp_condition = "epilepsy-abnormal-eeg"

        if site not in str(subject):
            subject = f"{site}{subject}"

        if subject not in subjects:
            print(f"Skipping {subject}")
            continue

        _update_site(bids_root, subject, site)

        for key, description in columns.items():
            if key == "outcome":
                update_participants_info(
                    root=bids_root,
                    subject=subject,
                    key=key,
                    value=outcome,
                    description=description,
                )
            elif key == 'exp_condition':
                update_participants_info(
                    root=bids_root,
                    subject=subject,
                    key=key,
                    value=exp_condition,
                    description=description,
                )
            else:
                value = row[key]
                if pd.isnull(value):
                    value = "n/a"
                if isinstance(value, str):
                    value = value.lower()
                update_participants_info(
                    root=bids_root,
                    subject=subject,
                    key=key,
                    value=value,
                    description=description,
                )


def _update_site(bids_root, subject, site):
    update_participants_info(
        root=bids_root,
        subject=subject,
        key="site",
        value=site,
        description="Clinical center at which subject was from",
    )


if __name__ == "__main__":
    pipeline_jhh()
    pipeline_jefferson()
