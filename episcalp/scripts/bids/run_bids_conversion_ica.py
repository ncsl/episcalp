from pathlib import Path
import shutil as sh
import os
from tempfile import TemporaryDirectory, tempdir
from mne_bids import (
    BIDSPath,
    make_dataset_description,
    update_sidecar_json,
    read_raw_bids,
)
from mne_bids.path import get_entities_from_fname, get_entity_vals, print_dir_tree
from natsort import natsorted
from episcalp import bids

from episcalp.bids import write_epitrack_bids, append_original_fname_to_scans


def convert_eeglab_ica_to_bids():
    root = Path("D:/OneDriveParent/OneDrive - Johns Hopkins/Shared Documents/40Hz-30")
    root = Path("/Users/adam2392/Johns Hopkins/Scalp EEG JHH - Documents/bids/")
    # root = Path("/Users/adam2392/Johns Hopkins/Jefferson_Scalp - Documents/root/")

    filter_scheme = "1-30Hz-30"
    window_scheme = "win-20"
    authors = "Patrick Myers and Adam Li"
    dataset_name = "Jefferson-ICLabel-Cleaned"
    dataset_name = 'JHH-ICLabel-Cleaned'

    # root and filtering scheme
    ica_root = root / "derivatives" / "ICA" / filter_scheme / window_scheme
    ica_src = ica_root / "sourcedata"
    ica_ext = ".set"

    # BIDS entities
    montage = "standard_1020"
    datatype = "eeg"
    suffix = "eeg"

    # desired extension
    extension = ".edf"

    # get all subjects
    subjects = get_entity_vals(ica_src, "subject")

    print("Got these subjects...", subjects)
    for subject in subjects:
        subj_src = ica_src / f"sub-{subject}"
        # get all subject file paths
        subject_fpaths = natsorted(list(subj_src.glob(f"*{subject}*{ica_ext}")))

        for idx, fpath in enumerate(subject_fpaths):
            entities = get_entities_from_fname(fpath.name)
            print("Entities: ", entities)
            print(fpath)
            bids_path = BIDSPath(
                root=ica_root,
                datatype=datatype,
                extension=extension,
                **entities
                #  subject=subject, suffix=suffix
            )
            run = bids_path.run

            # write to BIDS again
            bids_path = write_epitrack_bids(
                source_path=fpath,
                bids_path=bids_path,
                overwrite=True,
                format="EDF",
                montage=montage,
                verbose=False,
            )

            # helper function to generate a URI link to sourcedata
            make_dataset_description(
                ica_root, dataset_name, authors=authors, dataset_type="derivative"
            )
            derivative_json = {
                "Description": "ICLabel in EEGLab cleaned data with 1-30Hz band pass filtering, "
                "threshold of 30 to detect brain signals and a window size of 20 seconds.",
            }
            dataset_path = BIDSPath(
                root=ica_root,
                suffix="dataset_description",
                extension=".json",
                check=False,
            )
            update_sidecar_json(dataset_path, derivative_json)

            # copy metadata from original source files
            part_src = root / "participants.json"
            part_dest = ica_root / "participants.json"
            sh.copyfile(part_src, part_dest)

            part_src = root / "participants.tsv"
            part_dest = ica_root / "participants.tsv"
            sh.copyfile(part_src, part_dest)

            # append scans original filenames
            append_original_fname_to_scans(
                Path(fpath).name, ica_root, bids_path.basename
            )


def _merge_src_metadata():
    root = Path("D:/OneDriveParent/OneDrive - Johns Hopkins/Shared Documents/40Hz-30")
    root = Path("/Users/adam2392/Johns Hopkins/Scalp EEG JHH - Documents/bids/")
    root = Path("/Users/adam2392/Johns Hopkins/Jefferson_Scalp - Documents/root/")

    filter_scheme = "1-30Hz-30"
    window_scheme = "win-20"

    # root and filtering scheme
    ica_root = root / "derivatives" / "ICA" / filter_scheme / window_scheme

    # BIDS entities
    datatype = "eeg"
    suffix = "eeg"

    # get all subjects
    subjects = get_entity_vals(ica_root, "subject")
    src_subjects = get_entity_vals(root, "subject")
    for subject in subjects:
        # find the corresponding raw source subjects
        if subject not in src_subjects:
            raise RuntimeError(f"{subject} not in the original raw source {root}.")

        # get all metadata files
        suffixes = ["events"]
        for suffix in suffixes:
            bids_path = BIDSPath(
                subject=subject, datatype=datatype, suffix=suffix, root=root
            )

            matched_files = bids_path.match()
            for meta_fpath in matched_files:
                dest_fpath = BIDSPath(
                    root=ica_root,
                    datatype=meta_fpath.datatype,
                    extension=meta_fpath.extension,
                    suffix=meta_fpath.suffix,
                    **meta_fpath.entities,
                )

                # copy over the file
                print(f"Copied {meta_fpath} to {dest_fpath}")
                sh.copyfile(meta_fpath.fpath, dest_fpath.fpath)


if __name__ == "__main__":
    # convert_eeglab_ica_to_bids()
    _merge_src_metadata()
