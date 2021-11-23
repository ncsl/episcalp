from pathlib import Path
import mne
from mne_bids import BIDSPath, get_entity_vals
from episcalp import read_scalp_eeg


def run_analysis(
        bids_path,
        reference="monopolar",
        resample_sfreq=None,
        deriv_path=None,
        figures_path=None,
        verbose=True,
        overwrite=False,
        extra_channels=None,
):
    subject = bids_path.subject
    root = bids_path.root
    rereference = False
    raw = read_scalp_eeg(
        bids_path,
        reference=reference,
        rereference=rereference,
        resample_sfreq=resample_sfreq,
        verbose=verbose,
    )


def main():
    root = Path(
        "D:/OneDriveParent/OneDrive - Johns Hopkins/Shared Documents/bids/derivatives/ICA/1-30Hz-30/win-20"
    )
    deriv_root = Path("D:/OneDriveParent/OneDrive - Johns Hopkins/Shared Documents/bids/derivatives")
    figures_root = deriv_root
    figure_ext = ".png"

    subjects = get_entity_vals(root, "subject")
    for subject in subjects:
        ignore_subjects = [sub for sub in subjects if sub is not subject]
        sessions = get_entity_vals(root, "session", ignore_subjects=ignore_subjects)
        if len(sessions) == 0:
            sessions = [None]
        for session in sessions:
            ignore_sessions = [ses for ses in sessions if ses is not session]
            tasks = get_entity_vals(
                root,
                "task",
                ignore_subjects=ignore_subjects,
                ignore_sessions=ignore_sessions,
            )
            if len(tasks) == 0:
                tasks = [None]
            for task in tasks:
                ignore_tasks = [tsk for tsk in tasks if tsk is not task]
                runs = get_entity_vals(
                    root,
                    "run",
                    ignore_subjects=ignore_subjects,
                    ignore_sessions=ignore_sessions,
                    ignore_tasks=ignore_tasks,
                )
                for run in runs:
                    bids_params = {
                        "subject": subject,
                        "session": session,
                        "task": task,
                        "run": run,
                        "datatype": "eeg",
                        "extension": ".edf",
                    }
                    bids_path = BIDSPath(root=root, **bids_params)
                    run_analysis(
                        bids_path,
                        deriv_path=deriv_root,
                        figures_path=figures_root,
                        resample_sfreq=256,
                        overwrite=True,
                        figure_ext=figure_ext,
                    )


if __name__ == "__main__":
    main()