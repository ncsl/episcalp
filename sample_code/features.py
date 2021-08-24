from pathlib import Path
import os
import numpy as np
from eztrack.io import DerivativeNumpy, DerivativeArray
from scipy.stats import entropy, skew, kurtosis
from sample_code.utils import _standard_lobes


def calculate_distribution(arr, normalize=True):
    row_sum = np.sum(arr, axis=1)
    if not normalize:
        return row_sum
    row_sum_pdf = row_sum / np.sum(row_sum)
    return row_sum_pdf


def _create_normal_distribution(n_chs):
    dist = np.ones((n_chs, 1))/n_chs
    return dist


def calculate_entropy(dist):
    return entropy(dist)


def calculate_variance(dist):
    return np.var(dist)


def calculate_skew(dist):
    return skew(dist)


def calculate_kurtosis(dist):
    return kurtosis(dist)


def calculate_kl_div(dist):
    comparison_dist = _create_normal_distribution(len(dist))
    p = np.array(dist).reshape(-1, 1)
    q = np.array([c[0] for c in comparison_dist]).reshape(-1, 1)
    print(f"p: {p.shape}, q: {q.shape}")
    return entropy(p, q)


def get_spike_rate(spike_dict):
    return spike_dict['total']


def get_max_spike_rate(spike_dict):
    total = spike_dict.pop('total')
    return max(spike_dict.values())


def get_lobe_spike_rate(spike_dict, lobe, separate=False):
    lobe_dict = _standard_lobes(separate)
    lobe_chs = [l.upper() for l in lobe_dict[lobe]]
    lobe_rates = [rate for ch, rate in spike_dict.items() if ch.upper() in lobe_chs]
    return np.mean(lobe_rates)


def get_slowing_count(slowing_dict):
    total_wins = []
    [total_wins.extend(vals) for vals in slowing_dict.values()]
    return len(total_wins)
