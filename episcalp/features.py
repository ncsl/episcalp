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


def heatmap_features(feature_map, types, ch_names=None, summary_method="average", **kwargs):
    if types is None:
        return None
    elif len(types) == 0:
        return None
    type_names = types
    if ch_names is None and "lobes" in types:
        raise ValueError('ch_names must be passed to calculate lobe features')
    n_chan = feature_map.shape[0]

    # Reduce the dimensions of the data either by averaging or performing PCA on time domain
    if summary_method == "average":
        this_data = np.nanmean(feature_map, axis=1).reshape(-1, 1)
    elif summary_method == "pca":
        n_components = kwargs.get("n_components", 1)
        pca = PCA(n_components=n_components)
        pca.fit(feature_map)

        this_data = []
        for idx in range(n_chan):
            pcai = pca.transform(feature_map[idx, :].reshape(1, -1))
            this_data.append(pcai[0])
        this_data = np.array(this_data)

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

def heatmap_features2(feature_map, types, ch_names=None, summary_method="average", **kwargs):
    if types is None:
        return None
    elif len(types) == 0:
        return None
    type_names = types
    if ch_names is None and "lobes" in types:
        raise ValueError('ch_names must be passed to calculate lobe features')
        
    n_chan = feature_map.shape[0]
    n_timepoints = feature_map.shape[1]
    
    # Compute features before reducing dimensions 
    if "quantile" in type_names:
        big_feature_vec = np.zeros((5,n_timepoints))
        # Compute features before reducing dimensions 
        for col in range(n_timepoints):
            feature_map_ = np.array([td[col] for td in feature_map])

            # quantile features of the EEG electrodes
            quantile_vec = np.hstack(
                [np.quantile(feature_map_, q=q) for q in [0.1, 0.5, 0.9]]
                + [feature_map_.mean()]
                + [feature_map_.std()]
            )
            big_feature_vec[:,col] = quantile_vec

    if "lobes" in type_names:
        lobe_dict = _standard_lobes(separate_hemispheres=separate)
        n_lobes = len(lobe_dict)

        big_feature_vec = np.zeros((n_lobes*2,n_timepoints))
        for col in range(n_timepoints):
            feature_map_ = np.array([td[col] for td in feature_map])
            i_lobe = -2
            for lobe, lobe_chs in lobe_dict.items():
                i_lobe+=2
                idx = [idx for idx in range(len(ch_names)) if ch_names[idx] in lobe_chs]
                if idx == []:
                    big_feature_vec[i_lobe,:] = -1*np.ones((1,n_timepoints))
                    big_feature_vec[i_lobe+1,:] = -1*np.ones((1,n_timepoints))
                    continue
                big_feature_vec[i_lobe,:]  = np.nanmean(feature_map[idx,:], axis=0)
                big_feature_vec[i_lobe+1,:]  =np.std(feature_map[idx,:], axis=0)

    # if "distribution" in type_names:
        # n_chs = len(ch_names)
        # distribution_vals = []
        # uni_dist = np.ones((n_chs, 1)) / n_chs
        # uni_dist = uni_dist.reshape((len(uni_dist),))

        # for col in range(this_data.shape[1]):
        #     this_data_ = np.array([td[col] for td in this_data])
        #     distribution_vals.append(entropy(this_data_))  # entropy
        #     distribution_vals.append(np.var(this_data_))  # variance
        #     distribution_vals.append(skew(this_data_))  # skew
        #     distribution_vals.append(kurtosis(this_data_))  # kurtosis

        #     distribution_vals.append(entropy(this_data_, uni_dist))  # kl divergence from uniform

        #     big_feature_vec.extend(distribution_vals)

    # Reduce the dimensions of the data either by averaging or performing PCA on time domain
    if summary_method == "average":
        feature_vec_ = np.nanmean(big_feature_vec, axis=1).reshape(-1, 1)
        feature_vec = [feature for avg_features in feature_vec_ for feature in avg_features]
    elif summary_method == "pca":
        n_components = kwargs.get("n_components", 1)
        pca = PCA(n_components=n_components)
        pca.fit(big_feature_vec)

       feature_vec = pca.transform(big_feature_vec)        # Need to verify this is ok 
       print(feature_vec)

    return feature_vec

