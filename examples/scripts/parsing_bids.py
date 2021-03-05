from mne_bids import get_entity_vals, BIDSPath


def get_bids_paths(root, task=None, acquisition=None, ext=".vhdr"):
    """
    Get a list of all the bids_paths in a certain dataset.

    Once you have these paths, you can easily use the eeg data with read_raw_bids.

    Parameters
    ----------
    root: str
        The root path where your bids directory is stored.

    Returns
    -------
    bids_fpaths: list
        List of all bids_paths in the root directory.

    """
    bids_fpaths = []
    subjects = get_entity_vals(root, 'subject')
    for subject in subjects:
        ignore_subjects = [sub for sub in subjects if sub != subject]
        sessions = get_entity_vals(root, "session", ignore_subjects=ignore_subjects)
        for session in sessions:
            ignore_sessions = [ses for ses in sessions if ses != session]
            runs = get_entity_vals(root, "run", ignore_subjects=ignore_subjects, ignore_sessions=ignore_sessions)
            for run in runs:
                # The other entities here are likely all standard across your dataset
                bids_path = BIDSPath(subject=subject,
                                     session=session,
                                     task=task,
                                     acquisition=acquisition,
                                     datatype=acquisition,
                                     run=run,
                                     suffix=acquisition,
                                     extension=ext, root=root)
                # Finds all the runs that match the above params
                add_paths = bids_path.match()
                bids_fpaths.extend(add_paths)
    return bids_fpaths


if __name__ == "__main__":
    bids_root = "D:/ScalpData/test_convert"
    bids_paths = get_bids_paths(bids_root, task="monitor", acquisition="eeg", ext=".edf")
    print(bids_paths)