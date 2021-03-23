from pathlib import Path
from mne_bids import BIDSPath

from episcalp.bids.bids_conversion import write_epitrack_bids, append_original_fname_to_scans


def convert_ica_to_bids(root, bids_identifiers, montage="standard_1020", ext=".set", to_ext=".edf"):
    source_dir = root / "sourcedata"
    # Find all the potential files in the sourcedata directory
    source_fpaths = [
        fpath
        for fpath in source_dir.rglob(f"*{ext}")
        if fpath.is_file()
    ]
    [print(f) for f in source_fpaths]
    for fpath in source_fpaths:
        # parse bids identifiers from filename
        # Assumes sourcedata filename structure as <subject>_s<session>_t<run><ext>
        subject, session, run = fpath.name.split("_")
        session = session.replace("s", "")
        run = run.replace(ext, "").replace("t", "")
        task = fpath.parent.name
        bids_identifiers.update(subject=subject, session=session, task=task, run=run, root=root, extension=to_ext)
        bids_path = BIDSPath(**bids_identifiers)

        bids_path = write_epitrack_bids(
            source_path=fpath,
            bids_path=bids_path,
            overwrite=True,
            montage=montage,
            verbose=False
        )

        # It is often useful to be able to backtrack the bids file to the sourcefile.
        # This will add an additional column in the scans.tsv file to allow this
        # back-tracking
        append_original_fname_to_scans(Path(fpath).name, root, bids_path.basename)


if __name__ == "__main__":
    bids_root = Path("D:/ScalpData/test_convert/derivatives/ICA")
    bids_identifiers = {
        "acquisition": "eeg"
    }
    convert_ica_to_bids(bids_root, bids_identifiers)
