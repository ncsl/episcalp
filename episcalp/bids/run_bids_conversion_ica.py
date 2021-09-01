from pathlib import Path
from mne_bids import BIDSPath

from episcalp.bids import write_epitrack_bids, append_original_fname_to_scans


def convert_icaeeg_to_bids(root, ext=".set"):
    source_dir = root / "sourcedata"
    session = 'initialvisit'
    task = 'monitor'
    run = "01"
    datatype = "eeg"
    montage = "standard_1020"

    fpaths = source_dir.glob(f"*{ext}")
    fpaths_list = [f for f in fpaths]
    fpaths = source_dir.glob(f"*{ext}")
    subjects = [f.name.replace(ext, "").split("_")[0].split("-")[0] for f in fpaths]
    fpaths = source_dir.glob(f"*{ext}")
    sessions = []
    for fpath in fpaths:
        fname = fpath.name
        fname_split = fname.split("-")
        if len(fname_split) > 1:
            sessions.append(fname_split[1].split(".")[0])
        else:
            sessions.append("initialvisit")
    for idx, subject in enumerate(subjects):
        source_path = fpaths_list[idx]
        if "-" in subject:
            subject = subject.replace("-", "")
        session = sessions[idx]

        bids_path = BIDSPath(
            root=root,
            subject=subject,
            session=session,
            task=task,
            run=run,
            datatype=datatype,
        )
        bids_path = write_epitrack_bids(
            source_fpath=source_path,
            bids_path=bids_path,
            overwrite=True,
            montage=montage,
            verbose=False,
            strong_anonymize=False
        )

        # append scans original filenames
        append_original_fname_to_scans(Path(source_path).name, root, bids_path.basename)


if __name__ == "__main__":
    root = Path("D:/OneDriveParent/OneDrive - Johns Hopkins/Shared Documents/40Hz-30")
    ext = ".set"
    convert_icaeeg_to_bids(root, ext)