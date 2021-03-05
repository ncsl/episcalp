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
    bids_fpath = get_bids_paths(root)[0]  # For simplicity just grabbing the first file

    raw = run_pyprep(
        bids_fpath,
        figures_path=figures_path,
        reference=reference,
        plot_raw=plot_raw
    )
    return raw


if __name__ == "__main__":
    bids_root = "D:/ScalpData/R01_ministudy"
    raw = using_pyprep(bids_root)
    print(raw.info)