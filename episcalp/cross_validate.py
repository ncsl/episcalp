from pathlib import Path
import numpy as np

from .preprocess.montage import _standard_lobes
from .scripts.spikes.summary import _get_spike_annots


def _compute_spike_rates(raw):
    raw.pick_types(eeg=True)
    raw.drop_channels(raw.info["bads"])
    ch_spike_rates = dict()

    # extract spike annotations
    spike_annots = _get_spike_annots(raw)
    if len(spike_annots) == 0:
        for ch_name in raw.ch_names:
            ch_spike_rates[ch_name] = 0
        return ch_spike_rates

    spike_df = spike_annots.to_data_frame()

    for ch_name in raw.ch_names:
        if any(ch_name in tup for tup in spike_df["ch_names"]):
            ch_spike_rates[ch_name] = len(
                spike_df[
                    spike_df["ch_names"].apply(
                        lambda x: True if ch_name in x else False
                    )
                ]
            )
        else:
            ch_spike_rates[ch_name] = 0

    return ch_spike_rates


def load_derived_datasets(roots, deriv_chains, load_func, **kwargs):
    """Load dataset as dictionary of lists.

    Parameters
    ----------
    roots : [type]
        [description]
    deriv_chains : str
        [description]
    load_func : function
        A lambda function that loads in datasets given a
        derivative root path. Should return a dictionary of
        lists.

    Returns
    -------
    derived_dataset : dictonary of list
        [description]

    Raises
    ------
    load_func should return a dictionary of lists:

        {
            'subject': [<subjects>],
            'data': [<list of numpy arrays for each subject dataset>],
            'ch_names': [<list of channel names for each subject dataset>],
        }
    """
    # ensure roots is a list of BIDS root
    if not isinstance(roots, list):
        roots = [roots]
    if not isinstance(deriv_chains, list):
        deriv_chains = [deriv_chains]

    if len(roots) != len(deriv_chains):
        raise RuntimeError(
            f"Passed in {len(roots)} BIDS root datasets and "
            f"{len(deriv_chains)} derivative chains. Need to pass in one root for every chain."
        )

    # loop over all datasets and load in the derived_dataset
    deriv_dataset = []
    for root, deriv_chain in zip(roots, deriv_chains):
        root = Path(root)
        deriv_root = root / "derivatives" / deriv_chain

        # load in the dataset
        dataset = load_func(deriv_root, **kwargs)

        # for read_tfrs, they come as a list
        if isinstance(dataset, list):
            dataset = dataset[0]
        deriv_dataset.append(dataset)

    dataset = deriv_dataset[0]
    for deriv in deriv_dataset:
        for key in deriv.keys():
            if key not in dataset.keys():
                raise RuntimeError(
                    f"All keys in {dataset.keys()} must match every other derived dataset. "
                    f"{key}, {deriv.keys()}."
                )

    # convert to a dictionary of lists
    derived_dataset = {key: [] for key in dataset.keys()}
    for deriv in deriv_dataset:
        for key in derived_dataset.keys():
            derived_dataset[key].extend(deriv[key])

    return derived_dataset


def get_X_features(derived_dataset, feature_name="data"):
    """Compute the feature X matrix.

    Will turn each spatiotemporal feature map into a
    ``n_features`` length vector for ``n_samples``.

    Parameters
    ----------
    derived_dataset : dictionary of lists
        A dataset comprising the data stored as a dictionary
        of lists. Each list component corresponds to another
        separate dataset with a total of ``n_samples`` datasets.
    feature_name : str, optional
        The key to access the spatiotemporal feature map inside
        derived_dataset, by default 'data'.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
        The feature matrix.
    """
    X = []

    ch_names = derived_dataset["ch_names"][0]

    # compute feature for every spatiotemporal heatmap
    for idx, feature_map in enumerate(derived_dataset[feature_name]):
        # across time for all electrodes
        # features = np.hstack(
        #     (
        #         feature_map.mean(axis=1),
        #         feature_map.std(axis=1),
        #     ),
        # )

        # average over time
        this_data = np.nanmean(feature_map, axis=1)
        features = np.empty((0,))

        # distributional features of the EEG electrodes
        # features = np.hstack(
        #     [
        #         np.quantile(this_data, q=q) for q in np.linspace(0.1, 1.0, 9)
        #     ]
        # )

        # values per lobe
        lobe_dict = _standard_lobes(separate_hemispheres=False)
        lobe_vals = []
        for lobe, lobe_chs in lobe_dict.items():
            idx = [idx for idx in range(len(ch_names)) if ch_names[idx] in lobe_chs]
            if idx == []:
                lobe_vals.append(-1)
                continue
            lobe_vals.append(np.nanmean(this_data[idx]))

        features = np.hstack([features, lobe_vals])

        X.append(features)

    X = np.array(X)
    return X
