from pathlib import Path
from mne_bids import BIDSPath, update_sidecar_json, read_raw_bids

from episcalp.bids.bids_conversion import write_epitrack_bids, append_original_fname_to_scans


def convert_ica_to_bids(root, source_dir, bids_identifiers,
                        montage="standard_1020", ext=".set", to_ext=".vhdr"):
    datatype = 'eeg'
    suffix = 'eeg'

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
        bids_identifiers.update(subject=subject, session=session, task=task,
                                run=run, root=root, extension=to_ext,
                                datatype=datatype, suffix=datatype)
        bids_path = BIDSPath(**bids_identifiers)

        bids_path.directory.mkdir(exist_ok=True, parents=True)
        print(f'Converting {bids_path} to BIDS... of {fpath}')
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

        # update sidecar json
        sidecar_path = bids_path.copy().update(extension='.json')
        raw = read_raw_bids(bids_path)
        sfreq = raw.info['sfreq']
        line_freq = raw.info['line_freq']
        freqs = ''  # function of line frequency and sampling freq

        # ICA pipeline filtered apriori
        # update_dict = {
        #     'SoftwareFilters': {
        #         'notch filter': {
        #             'order': '',
        #             'freqs': freqs,
        #         },
        #         'band-pass filter': {
        #             'l_freq': '',
        #             'h_freq': '',
        #         }
        #     }
        # }
        # update_sidecar_json(sidecar_path, update_dict)

        # update description.json
        descrip_path = BIDSPath(root=bids_path.root, suffix='dataset_description',
                                extension='.json', check=False)
        update_dict = {
            'DatasetType': 'derivative',
            'Name': 'EEGLab binICA',
            'Version': '',  # include version used
            'Description': 'See README for more info. Automated ICA procedure was ran with binICA from '
                           'EEGLab and then saved as .set files. Preprocessing, such as basic filtering '
                           'was done. Detected ICA components were removed from the data as part of '
                           'the automated procedure.'
        }
        update_sidecar_json(descrip_path, update_dict)
        break


if __name__ == "__main__":
    root = Path("D:/ScalpData/test_convert/derivatives/ICA")
    root = Path("/home/adam2392/hdd3/tuh_epileptic_abnormal_vs_normal_EEG/derivatives/ICA")
    root = Path("/home/adam2392/hdd3/tuh_epilepsy_vs_normal/derivatives/ICA")

    source_dir = root / "sourcedata"
    bids_identifiers = {
        # "acquisition": "eeg"
    }
    convert_ica_to_bids(root, source_dir, bids_identifiers)
