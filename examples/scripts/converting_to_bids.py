"""Converting a sample dataset to BIDS format."""
from pathlib import Path

from mne_bids.path import BIDSPath

from episcalp.bids.bids_conversion import write_epitrack_bids, append_original_fname_to_scans


def convert_dataset_to_bids(root, bids_identifiers, montage="standard_1020", ext=".edf"):
    """
    Convert a sample dataset to bids format.

    You will need to modify this function depending on your sourcedata structure.
    This current formula assumes the source data files are named as follows:
        <sub>_<run>.edf

    Parameters
    ----------
    root: Path
        Absolute path to the bids root directory
    bids_identifiers: dict
        Dictionary of bids identifiers you want to apply to your dataset globally
    montage: str
        The name of the montage to apply
    ext: str
        The extension of the files in the sourcedata directory

    Returns
    -------

    """
    source_dir = root / "sourcedata"
    # Find all the potential files in the sourcedata directory
    source_fpaths = [
        fpath
        for fpath in source_dir.glob(f"*{ext}")
        if fpath.is_file()
    ]

    # Grab the global bids identifiers
    session = bids_identifiers.get("session")
    task = bids_identifiers.get("task")
    acquisition = bids_identifiers.get("acquisition")
    datatype = bids_identifiers.get("datatype")

    for ind, fpath in enumerate(source_fpaths):
        # Assumes sourcedata filename structure as <subject>_<run><ext>
        subject, run = fpath.name.split("_")
        run = run.replace(ext, "")

        # Set up the bids path for this individual source file
        bids_path = BIDSPath(
            root=root,
            subject=subject,
            session=session,
            acquisition=acquisition,
            task=task,
            run=run,
            datatype=datatype,
        )

        # Convert the file to bids
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
    bids_root = Path("D:/ScalpData/test_convert")
    bids_identifiers = {
        "session": "initialvisit",
        "task": "monitor",
        "acquisition": "eeg"
    }
    convert_dataset_to_bids(bids_root, bids_identifiers)
