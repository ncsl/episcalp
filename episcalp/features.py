import numpy as np
from numpy.testing import assert_array_equal
from scipy.stats import entropy, skew, kurtosis
from sklearn.decomposition import PCA

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


def heatmap_features(feature_map, types, ch_names=None, summary_method="average", verbose=False, **kwargs):
    if verbose:
        print(f"Looking at types: {types}; with summary_method: {summary_method}; which has kwargs: {kwargs}")
    if types is None:
        return None
    elif len(types) == 0:
        return None
    type_names = types
    if ch_names is None and "lobes" in types:
        raise ValueError('ch_names must be passed to calculate lobe features')
    n_chan = feature_map.shape[0]

    possible_summary_methods = ["average", "variance", "pca", "singular_values"]
    if summary_method not in possible_summary_methods:
        raise ValueError(f"summary_method must be one of {possible_summary_methods}")

    # Reduce the dimensions of the data either by averaging or performing PCA on time domain
    if summary_method == "average":
        this_data = np.nanmean(feature_map, axis=1).reshape(-1, 1)
    elif summary_method == "variance":
        this_data = np.nanvar(feature_map, axis=1).reshape(-1, 1)
    elif summary_method == "pca":
        n_components = kwargs.get("n_components", 1)
        pca = PCA(n_components=n_components)
        pca.fit(feature_map)

        this_data = []
        for idx in range(n_chan):
            pcai = pca.transform(feature_map[idx, :].reshape(1, -1))
            this_data.append(pcai[0])
        this_data = np.array(this_data)
    elif summary_method == "singular_values":
        if "lobes" in type_names:
            raise ValueError(f"Cannot compute lobe features from singular values.")
        n_components = kwargs.get("n_components", n_chan)
        pca = PCA(n_components=n_components)
        pca.fit(feature_map)
        this_data = pca.singular_values_.reshape(-1, 1)

    feature_vec = []
    if "quantile" in type_names:
        for col in range(this_data.shape[1]):
            this_data_ = np.array([td[col] for td in this_data])
            # distributional features of the EEG electrodes
            quantile_vec = np.hstack(
                [np.quantile(this_data_, q=q) for q in [0.1, 0.5, 0.9]]
                + [this_data_.mean()]
                + [this_data_.std()]
            )
            feature_vec.extend(list(quantile_vec))
    if "spatial" in type_names:
        spatial_vec = this_data
        feature_vec.extend(spatial_vec)

    if "lobes" in type_names:
        separate = kwargs.get("separate_hemispheres", False)
        lobe_dict = _standard_lobes(separate_hemispheres=separate)
        for col in range(this_data.shape[1]):
            this_data_ = np.array([td[col] for td in this_data])
            lobe_vals = []
            for lobe, lobe_chs in lobe_dict.items():
                idx = [idx for idx in range(len(ch_names)) if ch_names[idx] in lobe_chs]
                if idx == []:
                    lobe_vals.append(-1)
                    lobe_vals.append(-1)
                    continue
                lobe_vals.append(np.nanmean(this_data_[idx]))
                lobe_vals.append(np.std(this_data_[idx]))
            feature_vec.extend(lobe_vals)
    if "first_n" in type_names:
        n_keep = kwargs.get("n_keep", 2)
        first_n = [td[0] for td in this_data[0:n_keep]]
        feature_vec.extend(first_n)

    if "distribution" in type_names:
        n_chs = len(ch_names)
        distribution_vals = []
        uni_dist = np.ones((n_chs, 1)) / n_chs
        uni_dist = uni_dist.reshape((len(uni_dist),))

        for col in range(this_data.shape[1]):
            this_data_ = np.array([td[col] for td in this_data])
            distribution_vals.append(entropy(this_data_))  # entropy
            distribution_vals.append(np.var(this_data_))  # variance
            distribution_vals.append(skew(this_data_))  # skew
            distribution_vals.append(kurtosis(this_data_))  # kurtosis

            distribution_vals.append(entropy(this_data_, uni_dist))  # kl divergence from uniform

            feature_vec.extend(distribution_vals)

    return feature_vec


def single_feature(feature_map, ch_names, metric_name, feat_type, summary_method, feat_kw, return_name=False, **kwargs):
    """
    Enable retrieval of a single feature.

    This is a bit complicated, but that is the nature of the features we are currently using.
    """
    if ch_names is None and feat_type == "lobes":
        raise ValueError('ch_names must be passed to calculate lobe features')
    n_chan = feature_map.shape[0]

    possible_summary_methods = ["average", "variance", "pca", "singular_values"]
    if summary_method not in possible_summary_methods:
        raise ValueError(f"summary_method must be one of {possible_summary_methods}")

    this_data = feature_map
    # Reduce the dimensions of the data either by averaging or performing PCA on time domain
    add_text = ""
    if summary_method == "average":
        this_data = np.nanmean(feature_map, axis=1).reshape(-1, 1)
    elif summary_method == "variance":
        this_data = np.nanvar(feature_map, axis=1).reshape(-1,1)
    elif summary_method == "pca":
        add_text = f"_component{feat_kw['component']}"
        comp_idx = feat_kw["component"]
        n_components = kwargs.get("n_components", comp_idx)
        pca = PCA(n_components=n_components)
        pca.fit(feature_map)

        this_data_ = []
        for idx in range(n_chan):
            pcai = pca.transform(feature_map[idx, :].reshape(1, -1))
            this_data_.append(pcai[0])

        this_data = np.array([td[comp_idx-1] for td in this_data_])
    elif summary_method == "singular_values":
        if feat_type == "lobes":
            raise ValueError(f"Cannot compute lobe features from singular values.")
        n_components = kwargs.get("n_components", n_chan)
        pca = PCA(n_components=n_components)
        pca.fit(feature_map)
        this_data = pca.singular_values_.reshape(-1, 1)

    if feat_type == "quantile":
        # distributional features of the EEG electrodes
        quantile_vec = np.hstack(
            [np.quantile(this_data, q=q) for q in [0.1, 0.5, 0.9]]
            + [this_data.mean()]
            + [this_data.std()]
        )
        quantile_names = ["q0.1", "q0.5", "q0.9", "mean", "std"]
        feat = quantile_vec[feat_kw["feat_ind"]]
        if return_name:
            feat_name = f"{metric_name}_{summary_method}{add_text}_{quantile_names[feat_kw['feat_ind']]}"
            return feat, feat_name
        return feat
    if feat_type == "spatial":
        spatial_vec = this_data
        ch_idx = ch_names.index(feat_kw["channel"])
        feat = spatial_vec[ch_idx]
        if return_name:
            feat_name = f"{metric_name}_{summary_method}{add_text}_{feat_kw['channel']}_mean"
            return feat, feat_name
        return feat
    if feat_type == "lobes":
        separate = kwargs.get("separate_hemispheres", False)
        lobe_dict = _standard_lobes(separate_hemispheres=separate)
        lobe = feat_kw["lobe"]
        lobe_chs = lobe_dict[lobe]
        idx = [idx for idx in range(len(ch_names)) if ch_names[idx] in lobe_chs]
        if feat_kw["stat"] == "mean":
            feat = np.nanmean(this_data[idx])
        else:
            feat = np.std(this_data[idx])
        if return_name:
            feat_name = f"{metric_name}_{summary_method}{add_text}_{lobe}_{feat_kw['stat']}"
            return feat, feat_name
        return feat
    if feat_type == "first_n":
        nidx = feat_kw["svn"]
        feat = this_data[nidx]
        if return_name:
            feat_name = f"{metric_name}_{summary_method}{add_text}_{feat_kw['svn']}"
            return feat, feat_name
        return feat
    else:
        uni_dist = np.ones((n_chan, 1)) / n_chan
        uni_dist = uni_dist.reshape((len(uni_dist),))
        if feat_type == "entropy":
            feat = entropy(this_data)
        elif feat_type == "variance":
            feat = np.var(this_data)
        elif feat_type == "skew":
            feat = skew(this_data)
        elif feat_type == "kurtosis":
            feat = kurtosis(this_data)
        elif feat_type == "kldiv":
            feat = entropy(this_data, uni_dist)
        else:
            raise ValueError(f"You passed an unknown feat_type: {feat_type}")
        if return_name:
            feat_name = f"{metric_name}_{summary_method}{add_text}_{feat_type}"
            return feat, feat_name
        return feat
