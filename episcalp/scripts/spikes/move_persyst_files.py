from pathlib import Path
import shutil
import pandas as pd
from mne_bids import BIDSPath


def move_persyst_files(sourcedir, deriv_dir, excel_fpath, reference, bids_identifiers):
    source_fpaths = Path(sourcedir).glob("*.lay")
    hospital_ids = [s.name.split("-")[0] for s in source_fpaths]
    source_fpaths = Path(sourcedir).glob("*.lay")

    meta_df = pd.read_excel(excel_fpath)
    meta_df = meta_df[meta_df["CLINICAL_CENTER"].isin(["jhh", "jhh-bayview"])]
    for hospital_id, source_fpath in zip(hospital_ids, source_fpaths):
        subject_row = meta_df[meta_df["hospital_id"] == int(hospital_id)]
        print(subject_row)
        subject_id = str(list(subject_row.get("PATIENT_ID").values)[0]).zfill(3)
        print(subject_id)

        bids_path = BIDSPath(
            subject=subject_id,
            session=bids_identifiers["session"],
            task=bids_identifiers["task"],
            run=bids_identifiers["run"],
            datatype="eeg",
            root=deriv_dir,
        )

        out_fname = Path(f"{bids_path.basename}_spikes.lay")
        out_dir = deriv_dir / "spikes" / reference / f"sub-{subject_id}"
        out_fpath = out_dir / out_fname

        out_dir.mkdir(exist_ok=True, parents=True)

        shutil.copy(source_fpath, out_fpath)

        dat_fpath = source_fpath.with_suffix(".dat")
        out_fname = Path(out_fname.name.replace(".lay", ".dat"))
        out_fpath = out_dir / out_fname
        shutil.copy(dat_fpath, out_fpath)


if __name__ == "__main__":
    bids_identifiers = {"session": "initialvisit", "task": "monitor", "run": "01"}
    sourcedir = Path("D:/ScalpData/detect_test/sourcedata")
    deriv_dir = Path(
        "D:/OneDriveParent/OneDrive - Johns Hopkins/Shared Documents/40Hz-30/derivatives"
    )
    excel_fpath = Path("D:/ScalpData/JHU_scalp_clinical_datasheet_raw_local.xlsx")
    reference = "monopolar"
    move_persyst_files(sourcedir, deriv_dir, excel_fpath, reference, bids_identifiers)
