from pathlib import Path
import shutil as sh
import os
from tempfile import TemporaryDirectory, tempdir
from mne_bids import BIDSPath, make_dataset_description, update_sidecar_json
from mne_bids.path import get_entities_from_fname, get_entity_vals, print_dir_tree
from natsort import natsorted

from episcalp.bids import write_epitrack_bids, append_original_fname_to_scans


def convert_eeglab_ica_to_bids():
    root = Path("D:/OneDriveParent/OneDrive - Johns Hopkins/Shared Documents/40Hz-30")
    root = Path("/Users/adam2392/Johns Hopkins/Scalp EEG JHH - Documents/bids/")

    filter_scheme = "1-30Hz-30"
    window_scheme = "win-20"
    authors = "Patrick Myers and Adam Li"
    dataset_name = "JHH-ICLabel-Cleaned"

    # root and filtering scheme
    ica_root = root / "derivatives" / "ICA" / filter_scheme / window_scheme
    ica_src = ica_root / "sourcedata"
    ext = ".set"

    # BIDS entities
    montage = "standard_1020"
    datatype = "eeg"
    suffix = "eeg"

    # desired extension
    extension = ".edf"

    # get all subjects
    # subjects = natsorted(list(ica_src.glob('*')))
    subjects = get_entity_vals(ica_src, "subject")
    print("Got these subjects...", subjects)
    for subject in subjects:
        subj_src = ica_src / f"sub-{subject}"
        # get all subject file paths
        subject_fpaths = list(subj_src.glob(f"*{subject}*{ext}"))

        for fpath in subject_fpaths:
            entities = get_entities_from_fname(fpath.name)
            print(entities)
            print(fpath)
            bids_path = BIDSPath(
                root=ica_root,
                datatype=datatype,
                **entities
                #  subject=subject, suffix=suffix
            )

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

            # copy files for each subject
            # subject_files =  list(BIDSPath(subject=subject, root=root).fpath.rglob('*'))
            subj_root = BIDSPath(subject=subject, root=root).directory
            with TemporaryDirectory() as fdir:
                temp_root = Path(fdir) / f"sub-{subject}"
                sh.copytree(subj_root, temp_root)
                # print_dir_tree(fdir)
                subject_files = BIDSPath(
                    subject=subject, root=fdir, extension=extension
                ).match()
                for fpath in subject_files:
                    os.remove(fpath)
                # print_dir_tree(fdir)

            # append scans original filenames
            append_original_fname_to_scans(
                Path(fpath).name, ica_root, bids_path.basename
            )


if __name__ == "__main__":
    convert_eeglab_ica_to_bids()
