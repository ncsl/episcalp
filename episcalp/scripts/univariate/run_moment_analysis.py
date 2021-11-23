from pathlib import Path
import mne
from mne_bids import BIDSPath, get_entity_vals, get_entities_from_fname
from episcalp import read_scalp_eeg
from mne_features.univariate import (
    compute_mean,
    compute_std,
    compute_variance,
    compute_skewness,
    compute_kurtosis
)
import json


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

    deriv_chain = Path("moments") / reference / f"sub-{subject}"

    rereference = False
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

    data = raw.get_data()
    ch_names = raw.ch_names

    mean = compute_mean(data)
    std = compute_std(data)
    variance = compute_variance(data)
    skewness = compute_skewness(data)
    kurtosis = compute_kurtosis(data)

    feat_map = {
        "mean": mean,
        "variance": variance,
        "std": std,
        "skewness": skewness,
        "kurtosis": kurtosis
    }

    for featname, featarr in feat_map.items():
        source = bids_path.basename
        source_entities = get_entities_from_fname(source)
        # do not store suffix
        if source_entities.get("suffix") is not None:
            suffix = source_entities.pop("suffix")
        else:
            suffix = source.split("_")[-1].split(".")[0]
        # set it as datatype if we find it
        if suffix in ["ieeg", "eeg", "meg"]:
            datatype = suffix
        entities = source_entities

        # create an expected BIDS basename using
        # rawsources and description
        deriv_basename = BIDSPath(**source_entities).basename
        deriv_basename = deriv_basename + f"_desc-{featname}_{suffix}"

        fname = deriv_basename + ".json"
        fdir = deriv_path / deriv_chain
        fdir.mkdir(parents=True, exist_ok=True)
        fpath = fdir / fname

        feat_dict = {}
        for ch_name, featval in zip(ch_names, featarr):
            feat_dict[ch_name] = featval

        with open(fpath, 'w+') as fid:
            json.dump(feat_dict, fid)


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
                    )


if __name__ == "__main__":
    main()