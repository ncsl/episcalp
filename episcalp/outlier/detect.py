import numpy as np


def identify_artifacts(data, win_size, ch_threshold, time_threshold):
    """[summary]

    Parameters
    ----------
    data : np.ndarray (n_signals, n_times)
        The EEG data.
    win_size : int
        The window size in sample points.

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    RuntimeError
        [description]
    """

    if data.ndim > 2:
        raise RuntimeError(f"Data should be 2D. Yours was {data.ndim}D.")

    n_signals, n_times = data.shape
    n_windows = np.floor(n_times / win_size)

    # get thresholds based on number of channnels and windows
    ch_threshold = np.round(ch_threshold * n_signals)
    time_threshold = np.round(time_threshold * win_size)

    # for each channel, find the threshold
    lower_perc = np.percentile(data, 5, axis=1)
    higher_perc = np.percentile(data, 95, axis=1)

    # create mask for artifacts
    artifacts = np.zeros(data.shape)

    artifacts[artifacts <= lower_perc] = -1
    artifacts[artifacts >= higher_perc] = 1

    # split data into smaller windows
    data_windows = np.nan((n_signals, win_size, n_windows))
    data_window_artifacts = np.nan((n_signals, win_size, n_windows))
    for idx in range(n_windows):
        win_start = idx * win_size
        win_end = (idx + 1) * win_size

        data_windows[..., idx] = data[:, win_start:win_end]
        data_window_artifacts[..., idx] = artifacts[:, win_start:win_end]

    # find the number of artifact time points in each window
    n_artifacts_ch = np.sum(np.abs(data_window_artifacts), axis=1)
    n_artifacts_time = n_artifacts_ch >= ch_threshold

    # artifact_wins =
    return artifacts
