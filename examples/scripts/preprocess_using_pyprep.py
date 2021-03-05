#!python
from episcalp import run_pyprep
from examples.scripts.parsing_bids import get_bids_paths


def using_pyprep(root, reference="monopolar", figures_path=None, plot_raw=False):
    """
    Use pyprep to simplify the eeg loading and preprocessing steps.

    The pyprep pipeline will read in the scalp eeg data, perform some
    basic preprocessing steps like setting the montage, notch and band-
    pass filtering, basic setting of 'bad' channels, and plotting the
    raw data with events if that flag is passed.


    Parameters
    ----------
    root: str
        The root path where your bids directory is stored.
    reference: str
        Either "monopolar" or "average".
    figures_path: str
        Where you want figures to be saved. Recommended location is bids_root/derivatives/figures
    plot_raw: bool
        Whether you want pyprep to plot the raw data

    Returns
    -------
    raw: mne.io.Raw
        instance of Raw Object that contains "prepped" data.

    """
    bids_fpath = get_bids_paths(root, task="monitor", acquisition="eeg", ext=".edf")[0]  # For simplicity just grabbing the first file

    raw = run_pyprep(
        bids_fpath,
        figures_path=figures_path,
        reference=reference,
        plot_raw=plot_raw
    )

    return raw


def parsing_annotations_with_pyprep(root, reference="monopolar", annotation_list=None, figures_path=None,
                                    plot_raw=False):
    """
    Simplify the events by passing description key words into pyprep.

    Sometimes in your plots, you will want to know when events happen, such as when a medication
    was pushed. Pyprep can subset the annotations object associated with mne.io.Raw
    to match key words. The below example looks for any mentions of Keppra.


    Parameters
    ----------
    root: str
        The root path where your bids directory is stored.
    reference: str
        Either "monopolar" or "average".
    annotation_list: list
        List of key words to match for annotations
    figures_path: str
        Where you want figures to be saved. Recommended location is bids_root/derivatives/figures
    plot_raw: bool
        Whether you want pyprep to plot the raw data

    Returns
    -------
    raw: mne.io.Raw
        instance of Raw Object that contains "prepped" data.

    """
    if annotation_list is None:
        annotation_list = ["keppra"]
    bids_fpath = get_bids_paths(root, task="monitor", acquisition="eeg", ext=".edf")[0]  # For simplicity just grabbing the first file

    raw = run_pyprep(
        bids_fpath,
        figures_path=figures_path,
        reference=reference,
        plot_raw=plot_raw,
        descriptions=annotation_list
    )

    return raw


if __name__ == "__main__":
    bids_root = "D:/ScalpData/test_convert"
    raw = parsing_annotations_with_pyprep(bids_root)
    print(raw.info)
