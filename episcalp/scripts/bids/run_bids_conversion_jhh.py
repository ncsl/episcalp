import enum
from pathlib import Path

from eztrack import write_eztrack_bids
from mne_bids.path import BIDSPath

from episcalp.bids.bids_conversion import write_epitrack_bids, append_original_fname_to_scans
from episcalp.bids.utils import update_participants_info

def convert_jhh_to_bids():
    # root = Path(
        # "D:/OneDriveParent/OneDrive - Johns Hopkins/Shared Documents - HEP Data")
    root = Path('/Users/adam2392/Johns Hopkins/Scalp EEG JHH - Documents')
    source_root = Path('/Users/adam2392/Johns Hopkins/Khalil Husari - EPITrack-RS/')

    # epilepsy source data from JHH is organized into three sub-folders
    # 0xx is non epilepsy
    # 1xx is epilepsy normal
    # 2xx is epilepsy abnormal
    non_epi_src = source_root / 'non-epilepsy normal EEG'
    epi_norm_src = source_root / 'epilepsy with normal EEG'
    epi_abnorm_src = source_root / 'epilepsy with abnormal EEG'

    # site ID prefix to subject IDs
    subj_site = 'jhh'

    # define BIDS identifiers
    datatype = "eeg"

    montage = 'standard_1020'
    format = 'EDF'
    line_freq = 60
    overwrite = False
    verbose = False

    for source_dir in [non_epi_src, epi_norm_src, epi_abnorm_src]:
        _convert_folder(source_dir, root, datatype, montage, format, line_freq,
    overwrite, verbose)

def _extract_meta_from_fname(fpath):
    task = None
    if '--' in fpath.name:
        return None, None
    elif fpath.name.count('-') == 2:
        _, site_id, task = fpath.name.split('-')
    elif fpath.name.count('-') == 1:
        _, site_id = fpath.name.split('-')
    return site_id, task

def _map_subject_to_exp(subject):
    pass

def _convert_folder(source_dir, root, datatype, montage, format, line_freq,
overwrite, verbose):
    subjects = []
    for idx, fpath in enumerate(source_dir.glob("*.edf")):
        if '-' in fpath.name:
            # source subject ID
            subject_id = fpath.name.split('-')[0]
        else:
            subject_id = fpath.name.split('.')[0]
        subjects.append(subject_id)

    # run BIDS conversion for each set of files
    for subject in subjects:
        fpaths = [
            fpath
            for fpath in source_dir.glob(f'{subject}*.edf')
        ]
        print(fpaths)
        subject = _map_subject_to_exp(subject)

        for idx, fpath in enumerate(fpaths):
            # extract rule for task and site
            site_id, task = _extract_meta_from_fname(fpath)

            run_id = idx + 1
            bids_kwargs = {
                "subject": subject,
                "run": run_id,
                'task': task,
                "datatype": datatype,
                "suffix": datatype
            }
            bids_path = BIDSPath(**bids_kwargs, root=root)
            bids_path = write_epitrack_bids(source_path=fpath,
                        bids_path=bids_path,
                        montage=montage,
                        format=format,
                        line_freq=line_freq,
                        overwrite=overwrite,
                        verbose=verbose
            )

            # update the participants tsv file
            update_participants_info(
                root, subject, 'site', site_id, description='Bayview or JHH', 
                levels=None, units=None
                )
            # add original scan name to scans.tsv
            append_original_fname_to_scans(fpath.name, root, bids_path.basename)


if __name__ == "__main__":
    convert_jhh_to_bids()
