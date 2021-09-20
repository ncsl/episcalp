import logging
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from mne.utils import warn
from mne_bids import BIDSPath, get_entity_vals

from eztrack.fragility.sourcesink import lds_raw_sourcesink, state_sourcesink_derivative
from eztrack.io import read_derivative_npy
from eztrack.utils import logger

import sys
import os

from sample_code.study import generate_patient_features

sys.path.append("../../")
sys.path.append("./episcalp")
print(os.getcwd())
from episcalp.io import read_scalp_eeg

logger.setLevel(logging.DEBUG)


def run_sourcesink_analysis(
    bids_path,
    reference="monopolar",
    resample_sfreq=None,
    deriv_path=None,
    figures_path=None,
    figure_ext='.pdf',
    verbose=True,
    overwrite=False,
    plot_heatmap=True,
    plot_raw=True,
    extra_channels=None,
    **model_params,
):
    subject = bids_path.subject
    root = bids_path.root
    rereference = False

    # get the root derivative path
    deriv_chain = Path("sourcesink") / reference / f"sub-{subject}"
    figures_path = figures_path / "figures" / deriv_chain
    raw_figures_path = root / "derivatives" / "figures" / "raw" / f"sub-{subject}"
    deriv_path = deriv_path / deriv_chain

    # check if we have original dataset
    source_basename = bids_path.copy().update(extension=None, suffix=None).basename
    deriv_fpaths = deriv_path.glob(f"{source_basename}*.npy")
    if not overwrite and len(list(deriv_fpaths)) > 0:
        warn(
            f"Not overwrite and the derivative file path for {source_basename} already exists. "
            f"Skipping..."
        )
        return
    print(bids_path)
    # load in raw data
    raw = read_scalp_eeg(
        bids_path,
        reference=reference,
        rereference=rereference,
        resample_sfreq=resample_sfreq,
        verbose=verbose,
    )
    if extra_channels:
        drop_chs = [ec for ec in extra_channels if ec in raw.ch_names]
        raw.info["bads"].extend(drop_chs)

    # use the same basename to save the data
    raw.drop_channels(raw.info["bads"])

    model_params = {
        "winsize": 500,
        "stepsize": 250,
    }
    # run heatmap
    ss_deriv, sink_deriv, state_arr_deriv = lds_raw_sourcesink(
        raw,
        reference=reference,
        return_all=True,
        **model_params,
        # progress_file='./test.txt'
    )

    ss_deriv_fpath = deriv_path / ss_deriv.info._expected_basename
    state_deriv_fpath = deriv_path / state_arr_deriv.info._expected_basename

    ss_deriv.save(ss_deriv_fpath, overwrite=overwrite)
    state_arr_deriv.save(state_deriv_fpath, overwrite=overwrite)

    # normalize and plot heatmap
    if plot_heatmap:
        figures_path.mkdir(exist_ok=True, parents=True)
        #ss_deriv.normalize()
        fig_basename = ss_deriv_fpath.with_suffix(figure_ext).name

        bids_path.update(suffix="channels", extension=".tsv")

        # read in sidecar channels.tsv
        channels_pd = pd.read_csv(bids_path.fpath, sep="\t")
        description_chs = pd.Series(
            channels_pd.description.values, index=channels_pd.name
        ).to_dict()
        print(description_chs)
        resected_chs = [
            ch
            for ch, description in description_chs.items()
            if description == "resected"
        ]
        print(f"Resected channels are {resected_chs}")

        print(f"saving figure to {figures_path} {fig_basename}")
        ss_deriv.plot_heatmap(
            soz_chs=resected_chs,
            cbarlabel="Fragility",
            cmap="turbo",
            # vertical_markers=vertical_markers,
            # soz_chs=soz_chs,
            # figsize=(10, 8),
            # fontsize=12,
            # vmax=0.8,
            title=fig_basename,
            figure_fpath=(figures_path / fig_basename),
        )

        fig_basename = fig_basename.replace(figure_ext, f"_topomap{figure_ext}")
        ss_deriv.plot_topomap(
            # soz_chs=resected_chs,
            cbarlabel="Fragility",
            cmap="turbo",
            # soz_chs=soz_chs,
            # figsize=(10, 8),
            # fontsize=12,
            # vmax=0.8,
            title=fig_basename,
            figure_fpath=(figures_path / fig_basename),
        )


def run_post_analysis(deriv_path=None, subject=None, features=None):
    if subject is not None:
        subjects = [subject]
    else:
        subjects = None
    generate_patient_features(
        deriv_path, "sourcesink", features, subjects=subjects, verbose=True
    )


if __name__ == "__main__":
    root = Path("D:/OneDriveParent/Johns Hopkins/Jefferson_Scalp - Documents/root/derivatives/ICA/1-30Hz-30/win-20")
    deriv_root = Path("D:/OneDriveParent/Johns Hopkins/Jefferson_Scalp - Documents/root/derivatives")
    figures_root = deriv_root
    figure_ext = ".png"

    subjects = get_entity_vals(root, 'subject')
    for subject in subjects:
        ignore_subjects = [sub for sub in subjects if sub is not subject]
        sessions = get_entity_vals(root, 'session', ignore_subjects=ignore_subjects)
        if len(sessions) == 0:
            sessions = [None]
        for session in sessions:
            ignore_sessions = [ses for ses in sessions if ses is not session]
            tasks = get_entity_vals(root, 'task', ignore_subjects=ignore_subjects, ignore_sessions=ignore_sessions)
            if len(tasks) == 0:
                tasks = [None]
            for task in tasks:
                ignore_tasks = [tsk for tsk in tasks if tsk is not task]
                runs = get_entity_vals(root, 'run', ignore_subjects=ignore_subjects, ignore_sessions=ignore_sessions, ignore_tasks=ignore_tasks)
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
                    run_sourcesink_analysis(bids_path, deriv_path=deriv_root, figures_path=figures_root,
                                            resample_sfreq=256, overwrite=True, figure_ext=figure_ext)