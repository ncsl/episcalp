from pathlib import Path
import numpy as np
from scipy.stats import entropy, skew, kurtosis
import pandas as pd

from .preprocess.montage import _standard_lobes
from .scripts.spikes.summary import _get_spike_annots
from numpy.random import SeedSequence, default_rng

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


def get_X_features(derived_dataset, data_name="data", feature_names=None):
    """Compute the feature X matrix.

    Will turn each spatiotemporal feature map into a
    ``n_features`` length vector for ``n_samples``.

    Parameters
    ----------
    derived_dataset : dictionary of lists
        A dataset comprising the data stored as a dictionary
        of lists. Each list component corresponds to another
        separate dataset with a total of ``n_samples`` datasets.
    data_name : str, optional
        The key to access the spatiotemporal feature map inside
        derived_dataset, by default 'data'.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
        The feature matrix.
    """
    X = []

    if feature_names is None:
        feature_names = ['lobes']

    ch_names = derived_dataset["ch_names"][0]

    # compute feature for every spatiotemporal heatmap
    for idx, feature_map in enumerate(derived_dataset[data_name]):
        # across time for all electrodes
        # features = np.hstack(
        #     (
        #         feature_map.mean(axis=1),
        #         feature_map.std(axis=1),
        #     ),
        # )

        # average over time
        this_data = np.nanmean(feature_map, axis=1)
        n_chs = len(this_data)
        # features = np.empty((0,))

        # distributional features of the EEG electrodes
        if 'quantiles' in feature_names:
            features = np.hstack(
                [np.quantile(this_data, q=q) for q in [0.1, 0.5, 0.9]]
                + [this_data.mean()]
                + [this_data.std()]
            )
            X.append(features)


        # values per lobe
        if 'lobes' in feature_names:
            lobe_dict = _standard_lobes(separate_hemispheres=False)
            lobe_vals = []
            for lobe, lobe_chs in lobe_dict.items():
                idx = [idx for idx in range(len(ch_names)) if ch_names[idx] in lobe_chs]
                if idx == []:
                    lobe_vals.append(-1)
                    continue
                lobe_vals.append(np.nanmean(this_data[idx]))
                lobe_vals.append(np.std(this_data[idx]))
            # features = np.hstack([features, lobe_vals])
            features = lobe_vals

            X.append(features)


        # distribution features
        if 'distribution' in feature_names:
            distribution_vals = []
            uni_dist = np.ones((n_chs, 1)) / n_chs
            uni_dist = uni_dist.reshape((len(uni_dist),))


            distribution_vals.append(entropy(this_data))
            distribution_vals.append(np.var(this_data))
            distribution_vals.append(skew(this_data))
            distribution_vals.append(kurtosis(this_data))
            #distribution_vals.append(np.mean(this_data))

            distribution_vals.append(entropy(this_data, uni_dist))

            features = distribution_vals
            X.append(features)

    X = np.array(X)
    return X


def exclude_subjects(X, y, subjects, roots, categorical_exclusion_criteria, continuous_exclusion_criteria=None, return_inds=False):
    """
    Exclude subjects by providing column values from the participants.tsv to remove.

    Parameters
    ----------
    X
    y
    subjects
    roots
    categorical_exclusion_criteria: dict
        Keynames are categorical columns from the participants.tsv and the values are the values to exclude.
    continuous_exclusion_criteria: dict
        Keynames are continuous columns from the participants.tsv and the values are list of ranges to exclude.
        First character must be the modifier. i.e. ">65"

    Returns
    -------

    """
    if not continuous_exclusion_criteria:
        continuous_exclusion_criteria = dict()
        
    dfs = []
    for root in roots:
        participants_fpath = root / "participants.tsv"
        df = pd.read_csv(participants_fpath, sep="\t")
        dfs.append(df)
    participants_df = pd.concat(dfs, ignore_index=True)
    for colname, elist in categorical_exclusion_criteria.items():
        if elist is None:
            continue
        participants_df = participants_df[~participants_df[colname].isin(elist)]

    for colname, elist in continuous_exclusion_criteria.items():
        if elist is None:
            continue
        min_cutoff = None
        max_cutoff = None
        for erange in elist:
            modifier = erange[0]
            value = erange[1:]
            if modifier == ">":
                max_cutoff = value
            if modifier == "<":
                min_cutoff = value
        if min_cutoff is None and max_cutoff is None:
            continue
        if min_cutoff is None:
            participants_df = participants_df[participants_df[colname] < max_cutoff]
        elif max_cutoff is None:
            participants_df = participants_df[participants_df[colname] > min_cutoff]
        elif min_cutoff > max_cutoff:
            participants_df = participants_df[~participants_df[colname].between(max_cutoff, min_cutoff, inclusive=False)]
        else:
            participants_df = participants_df[participants_df[colname].between(min_cutoff, max_cutoff, inclusive=False)]

    keep_subjects = []
    for ind, row in participants_df.iterrows():
        keep_subjects.append(row['participant_id'].replace("sub-", ""))
    keep_idx_ = [idx for idx, s in enumerate(subjects) if s in keep_subjects]
    keep_idx = np.array(keep_idx_)
    print(f"X: {X}", X.shape)
    X = X[keep_idx, ...]
    y = y[keep_idx]
    keep_subjects = subjects[keep_idx]
    if return_inds:
        return X, y, keep_subjects, keep_idx
    return X, y, keep_subjects


def ensureBalancedLabels(n_splits,perc_train,y,random_state):

    """
    Splits dataset into training and test set while ensuring equal balance of classes in each set
    Parameters: 
        n_splits: Number of different train/test splits
        perc_train: Percentage of dataset used for training
        y: Class labels of patients in dataset [n_patients x 1]
        random_state: Seed for random number generator
    Returns:
        train_ind: Indices of patients in training set for each split (n_splits x n_train)
        test_ind: Indices of patients in test set of each split (n_splits x n_train)
    """

    np.random.seed(random_state)   

    y = np.array(y)
    ind_class1 = np.where(y==0)[0]
    ind_class2 = np.where(y==1)[0]

    n_class1 = len(ind_class1)
    n_class2 = len(ind_class2)
    n_patients = n_class1+n_class2
    # Number of patients from each class in training set
    n_train_class = round((n_patients*perc_train)/2)
    n_train = n_train_class*2
    n_test = n_patients-n_train

    if n_train_class > n_class1:
        raise ValueError("Not enough patients in class y=0 for a balanced split")
    elif n_train_class > n_class2:
        raise ValueError("Not enough patients in class y=1 for a balanced split")

    balanced_cv = []
    for split in range(n_splits):
        
        # Randomly select equal number of patients from each class for training set
        train_ind1 = np.random.choice(ind_class1,n_train_class,False)
        train_ind2 = np.random.choice(ind_class2,n_train_class,False)
        _train_ind = np.concatenate([train_ind1,train_ind2])
        _train_ind = np.sort(_train_ind)
        _test_ind = np.array(range(0,n_patients))
        _test_ind = np.delete(_test_ind,_train_ind)

        balanced_cv.append((_train_ind,_test_ind))

    return balanced_cv
