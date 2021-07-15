from pathlib import Path

from eztrack import write_eztrack_bids
from eztrack.preprocess import append_original_fname_to_scans
from mne_bids.path import BIDSPath


def convert_hep_to_bids():
    root = Path("D:/OneDriveParent/OneDrive - Johns Hopkins/Shared Documents - HEP Data")
    source_dir = root / "sourcedata" / "zip1"

    # define BIDS identifiers
    acquisition = "eeg"
    datatype = "eeg"
    task = "monitor"
    session = "original"

    montage = "standard_1020"
    set_fpaths = [
        fpath
        for fpath in source_dir.glob("*.set")
        if "original" in fpath.name
    ]
    print(set_fpaths)

    for fpath in set_fpaths:
        subject, run_id, _ = fpath.name.split("_")
        bids_kwargs = {
            "subject": subject,
            "session": session,
            "task": task,
            "acquisition": acquisition,
            "run": run_id,
            "datatype": datatype,
            "suffix": datatype
        }
        bids_path = BIDSPath(**bids_kwargs, root=root)
        bids_path = write_eztrack_bids(
            source_fpath=fpath, bids_path=bids_path, line_freq=60, montage=montage,
            strong_anonymize=False
        )

        append_original_fname_to_scans(fpath.name, root, bids_path.basename)


if __name__ == "__main__":
    convert_hep_to_bids()
