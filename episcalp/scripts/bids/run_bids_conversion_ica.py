from pathlib import Path
from mne_bids import BIDSPath
from mne_bids.path import get_entities_from_fname, get_entity_vals

from episcalp.bids import write_epitrack_bids, append_original_fname_to_scans


def convert_eeglab_ica_to_bids():
    root = Path("D:/OneDriveParent/OneDrive - Johns Hopkins/Shared Documents/40Hz-30")
    root = Path("/Users/adam2392/Johns Hopkins/Scalp EEG JHH - Documents/bids/")

    # root and filtering scheme
    ica_root = root / "derivatives" / "30Hz-30"
    ica_src = ica_root / "sourcedata"
    ext = ".set"

    # BIDS entities
    montage = "standard_1020"
    datatype = "eeg"

    # get all subjects
    subjects = get_entity_vals(root, "subject")

    for subject in subjects:
        # get all subject file paths
        subject_fpaths = ica_src.glob(f"sub-{subject}*{ext}")

        for fpath in subject_fpaths:
            entities = get_entities_from_fname(fpath.name)
            bids_path = BIDSPath(root=ica_root, datatype=datatype, **entities)

            # write to BIDS again
            bids_path = write_epitrack_bids(
                source_fpath=fpath,
                bids_path=bids_path,
                overwrite=True,
                format="EDF",
                montage=montage,
                verbose=False,
                strong_anonymize=False,
            )

            # append scans original filenames
            append_original_fname_to_scans(Path(fpath).name, root, bids_path.basename)


if __name__ == "__main__":

    convert_icaeeg_to_bids(root, ext)
