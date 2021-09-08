import numpy as np

from episcalp.utils.utils import compute_sample_points


def identify_artifacts(data, lower_perc=None, higher_perc=None):
    """
    Identify artifact points in data.

    Parameters
    ----------
    data: np.ndarray
        nCh x nT array of EEG data
    lower_perc: float
        Value of cutoff where any data below this value is considered artifact. Default (None) will calculate
        this calculate this value as the 5th percentile from the data input.
    higher_perc: float
        Value of cutoff where any data above this value is considered artifact. Default (None) will calculate
        this calculate this value as the 95th percentile from the data input.

    Returns
    -------
    np.ndarray
        nCh x nT array where -1 denotes a low artifact and 1 denotes a high artifact.

    Raises
    ------
    RuntimeError
        [description]

    """

    if data.ndim > 2:
        raise RuntimeError(f"Data should be 2D. Yours was {data.ndim}D.")

    n_signals, n_times = data.shape

    # for each channel, find the threshold
    if lower_perc is None:
        lower_perc = np.percentile(data, 5, axis=1)
    if higher_perc is None:
        higher_perc = np.percentile(data, 95, axis=1)

    # create mask for artifacts
    artifacts = np.zeros(data.shape)

    artifacts[data <= lower_perc] = -1
    artifacts[data >= higher_perc] = 1

    return artifacts


def identify_artifact_windows(data, winsize, perc_chans, perc_time, lower_perc=None, higher_perc=None):
    """
    Get windows that surpass the channel and timepoint threshold.

    Parameters
    ----------
    data: np.ndarray
        nCh x nT array of EEG data
    winsize: int
        nSamples that define a window
    perc_chans: float
        Percentage of channels that must be simultaneously artifactual to be considered an artifact.
    perc_time: float
        Percentage of time within a window that the channels must be artifactual to be considered an artifact.
    lower_perc: float
        Value of cutoff where any data below this value is considered artifact. Default (None) will calculate
        this calculate this value as the 5th percentile from the data input.
    higher_perc: float
        Value of cutoff where any data above this value is considered artifact. Default (None) will calculate
        this calculate this value as the 95th percentile from the data input.

    Returns
    -------

    """
    # Get artifact points per channel
    artifacts = identify_artifacts(data, lower_perc, higher_perc)

    n_signals, n_times = data.shape

    # Get required number of channels and timepoints to be considered a valid window.
    r_signals = np.ceil(n_signals * perc_chans)
    r_times = np.ceil(winsize, perc_time)

    # Initialize an array for the windows
    n_wins = np.ceil(n_times / winsize)
    artifact_wins = np.zeros((n_wins, 1))

    sample_points = compute_sample_points(n_times, winsize, winsize)
    for ind, (start_win, stop_win) in enumerate(sample_points):
        artifact_win = artifacts[:, start_win:stop_win]
        # Get number of channels per time point that were artifactual
        artifact_ch_count = np.sum(artifact_wins, axis=0)
        # Get timepoints where number of channel threshold was met
        artifact_ch_pos = [a >= r_signals for a in artifact_ch_count]
        # Get the number of timepoints that passed the above
        artifact_timepoints = np.sum(artifact_ch_pos)
        if artifact_timepoints >= r_times:
            artifact_wins[ind] = 1

    return artifact_wins


