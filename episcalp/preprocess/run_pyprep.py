#!python
import mne

from episcalp.io.read import read_scalp_eeg
from episcalp.utils import (
    _parse_sz_events,
)


def run_pyprep(bids_path, figures_path, reference, rereference=False, plot_raw=True, verbose=True):
    raw = read_scalp_eeg(bids_path, reference, rereference=rereference, verbose=verbose)

    # save output data
    bids_path.update(
        # processing='pyprep',
        root=None
    )

    # get the seizure event from @Seizure (persyst)
    # find all events
    events, events_id = mne.events_from_annotations(raw)
    sfreq = raw.info["sfreq"]
    orig_time = raw.info["meas_date"]
    annotations = _parse_sz_events(
        events,
        events_id,
        descriptions=["onset", "@seizure", "offset", "sz event"],
        sfreq=sfreq,
        orig_time=orig_time,
    )
    raw.set_annotations(annotations)
    events, _ = mne.events_from_annotations(raw)

    if plot_raw:
        # find out where to start the raw plot at
        if annotations is not None:
            start = max(0, annotations[[0]].onset - 10)
        else:
            start = 0

        # save output figure
        figure_fpath = (
            figures_path / "raw" / bids_path.update(extension=".png", check=False).basename
        )
        figure_fpath.parent.mkdir(parents=True, exist_ok=True)
        scale = 25e-6
        for title, proj in zip(["Original", "Referenced"], [False, True]):
            fig = raw.plot(
                start=start,
                events=events,
                # proj=proj,
                decim=5,
                duration=20,
                scalings={
                    "eeg": scale,
                },
                n_channels=len(raw.ch_names),
                clipping=None,
            )
            # make room for title
            fig.subplots_adjust(top=0.9)
            fig.suptitle(f"{title} reference", size="xx-large", weight="bold")
        fig.savefig(figure_fpath)
    # raw.apply_proj()
    return raw
