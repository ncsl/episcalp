import numpy as np


def identify_artifacts(data):

    if data.ndim > 2:
        raise RuntimeError(f"Data should be 2D. Yours was {data.ndim}D.")

    n_signals, n_times = data.shape

    # for each channel, find the threshold
    lower_perc = np.percentile(data, 5, axis=1)
    higher_perc = np.percentile(data, 95, axis=1)

    # create mask for artifacts
    artifacts = np.zeros(data.shape)

    artifacts[artifacts <= lower_perc] = -1
    artifacts[artifacts >= higher_perc] = 1
    return artifacts
