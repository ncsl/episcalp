import numpy as np
from numpy.testing import assert_array_equal
from scipy.stats import entropy, skew, kurtosis

from .montage import _standard_lobes

#   heights = []
#     for descrip in descriptions:
#         if "perception:" not in descrip:
#             raise RuntimeError(f"Perception not in {raw} spike annotations.")

#         # extract perception
#         perception = descrip.split("perception:")[1].split(" ")[0]

#         # extract height in uV
#         height = descrip.split("height:")[1].split(" ")[0]

#         perceptions.append(perception)
#         heights.append(height)
#     spike_df["perception"] = perceptions
#     spike_df["height"] = heights


def spike_feature_vector(ch_spike_df, ch_names, type="rate"):
    ch_spike_count = dict()
    ch_spike_rates = dict()
    feature_vec = np.zeros((6,))

    if ch_spike_df.empty:
        return feature_vec

    # compute the count of spike events per channel
    for ch_name in ch_names:
        if any(ch_name in tup for tup in ch_spike_df["ch_name"]):
            ch_spike_count[ch_name] = len(
                ch_spike_df[ch_spike_df["ch_name"] == ch_name]
            )
        else:
            ch_spike_count[ch_name] = 0

    n_secs = ch_spike_df["n_secs"].tolist()[0]
    # normalize spike counts
    for ch_name, val in ch_spike_count.items():
        ch_spike_rates[ch_name] = val / n_secs

    # total spike rate = total # of spikes regardless of channel / times / n_chs
    total_spike_rate = np.sum(list(ch_spike_rates.values())) / len(ch_names) / n_secs

    # max spike per lobe
    ch_spikes = np.array(list(ch_spike_rates.values()))
    lobe_dict = _standard_lobes(separate_hemispheres=False)

    # value of the max spike rate
    feature_vec = np.hstack(
        (
            ch_spikes.mean(),
            ch_spikes.std(),
            np.quantile(ch_spikes, q=0.1),
            np.quantile(ch_spikes, q=0.5),
            np.quantile(ch_spikes, q=0.9),
            total_spike_rate,
        )
    )

    assert len(ch_names) == len(ch_spike_rates.keys())
    assert_array_equal(ch_names, list(ch_spike_rates.keys()))
    return feature_vec


def heatmap_features(feature_map, ch_names=None, types=None):
    if types is None:
        types = ["quantile"]
    if ch_names is None and "lobes" in types:
        raise ValueError('ch_names must be passed to calculate lobe features')
    # average over time
    this_data = np.nanmean(feature_map, axis=1)
    # features = np.empty((0,))

    feature_vec = []
    if "quantile" in types:
        # distributional features of the EEG electrodes
        quantile_vec = np.hstack(
            [np.quantile(this_data, q=q) for q in [0.1, 0.5, 0.9]]
            + [this_data.mean()]
            + [this_data.std()]
        )
        feature_vec.extend(list(quantile_vec))
    if "spatial" in types:
        spatial_vec = this_data
        feature_vec.extend(spatial_vec)

    if "lobes" in types:
        lobe_dict = _standard_lobes(separate_hemispheres=False)
        lobe_vals = []
        for lobe, lobe_chs in lobe_dict.items():
            idx = [idx for idx in range(len(ch_names)) if ch_names[idx] in lobe_chs]
            if idx == []:
                #lobe_vals.append(-1)
                continue
            lobe_vals.append(np.nanmean(this_data[idx]))
            lobe_vals.append(np.std(this_data[idx]))
        feature_vec.extend(lobe_vals)

    if "distribution" in types:
        n_chs = len(ch_names)
        distribution_vals = []
        uni_dist = np.ones((n_chs, 1)) / n_chs
        uni_dist = uni_dist.reshape((len(uni_dist),))

        distribution_vals.append(entropy(this_data))  # entropy
        distribution_vals.append(np.var(this_data))  # variance
        distribution_vals.append(skew(this_data))  # skew
        distribution_vals.append(kurtosis(this_data))  # kurtosis
        # distribution_vals.append(np.mean(this_data))

        distribution_vals.append(entropy(this_data, uni_dist))  # kl divergence from uniform

        feature_vec.extend(distribution_vals)

    #feature_vec = np.array(feature_vec)
    return feature_vec
