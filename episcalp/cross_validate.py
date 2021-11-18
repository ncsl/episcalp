from pathlib import Path
import numpy as np
from scipy.stats import entropy, skew, kurtosis
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from sklearn.metrics import (
    brier_score_loss,
    average_precision_score,
    roc_auc_score,
    f1_score,
    recall_score,
    jaccard_score,
    balanced_accuracy_score,
    precision_score,
    average_precision_score,
    cohen_kappa_score,
    make_scorer,
)

# if you installed sporf via README
from oblique_forests.sporf import ObliqueForestClassifier

from .preprocess.montage import _standard_lobes
from .features import heatmap_features
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
        feature_names = ["lobes"]

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
        if "quantiles" in feature_names:
            features = np.hstack(
                [np.quantile(this_data, q=q) for q in [0.1, 0.5, 0.9]]
                + [this_data.mean()]
                + [this_data.std()]
            )
            X.append(features)

        # values per lobe
        if "lobes" in feature_names:
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
        if "distribution" in feature_names:
            distribution_vals = []
            uni_dist = np.ones((n_chs, 1)) / n_chs
            uni_dist = uni_dist.reshape((len(uni_dist),))

            distribution_vals.append(entropy(this_data))
            distribution_vals.append(np.var(this_data))
            distribution_vals.append(skew(this_data))
            distribution_vals.append(kurtosis(this_data))
            # distribution_vals.append(np.mean(this_data))

            distribution_vals.append(entropy(this_data, uni_dist))

            features = distribution_vals
            X.append(features)

    X = np.array(X)
    return X


def exclude_subjects(X, y, subjects, roots, exclusion_criteria):
    dfs = []
    for root in roots:
        participants_fpath = root / "participants.tsv"
        df = pd.read_csv(participants_fpath, sep="\t")
        dfs.append(df)
    participants_df = pd.concat(dfs, ignore_index=True)
    for colname, elist in exclusion_criteria.items():
        if elist is None:
            continue
        participants_df = participants_df[~participants_df[colname].isin(elist)]
    keep_subjects = []
    for ind, row in participants_df.iterrows():
        keep_subjects.append(row["participant_id"].replace("sub-", ""))
    keep_idx_ = [idx for idx, s in enumerate(subjects) if s in keep_subjects]
    keep_idx = np.array(keep_idx_)
    X = X[keep_idx, ...]
    y = y[keep_idx]
    keep_subjects = subjects[keep_idx]
    return X, y, keep_subjects


def _get_exp_condition(subject, root):
    part_fname = os.path.join(root, "participants.tsv")
    df = pd.read_csv(part_fname, sep="\t")

    if not subject.startswith("sub-"):
        subject = f"sub-{subject}"

    return df[df["participant_id"] == subject]


def convert_experimental_cond_to_y(experimental_condition_list):
    """Encoder for y labels."""
    # Group name keys, assigned y-label values
    experimental_condition_map = {
        "non-epilepsy-normal-eeg": 0,
        "epilepsy-normal-eeg": 1,
        "epilepsy-abnormal-eeg": 2,
    }
    return [experimental_condition_map[cond] for cond in experimental_condition_list]


def run_exhaustive_heatmap_evaluation(
    heatmap_dict, names, feature_types, result_fname, clf_name="lr", random_state=None
):
    n_splits = 20
    train_size = 0.7
    stratified_cv = StratifiedShuffleSplit(
        n_splits=n_splits,
        train_size=train_size,
        random_state=random_state,
    )
    cvs = {
        "stratifiedshuffle": stratified_cv,
        #        "leaveonesubout": log_cv
    }

    scaler = StandardScaler()
    y_enc = LabelBinarizer()

    # get an example dataset to loop over subjects
    ex_dataset = heatmap_dict["fragility"]
    subjects = np.array(ex_dataset["subject"])
    roots = ex_dataset["roots"]

    # ensure feature types is a list
    feature_types = np.atleast_1d(feature_types).tolist()

    # create feature matrix
    features = []
    for idx in range(len(ex_dataset["subject"])):
        feature_vec = []
        for name in names:
            dataset = heatmap_dict[name].copy()

            # extract data and form feature vector
            data = dataset["data"][idx]
            ch_names = dataset["ch_names"][idx]
            _feature_vec = heatmap_features(data, ch_names, types=feature_types)
            feature_vec.extend(_feature_vec)
        features.append(feature_vec)

    features = np.array(features)
    assert features.ndim == 2
    assert features.shape[0] == len(subjects)

    # get the experimental conditions
    exp_conditions = []
    for subject, root in zip(subjects, roots):
        subj_df = _get_exp_condition(subject, root)
        exp_condition = subj_df["exp_condition"].values[0]
        exp_conditions.append(exp_condition)

    # map to y
    y = np.array(convert_experimental_cond_to_y(np.array(exp_conditions)))

    # Further subset the subjects if desired
    exclusion_criteria = {
        "exp_condition": ["epilepsy-abnormal-eeg"],
        "final_diagnosis": None,
        "epilepsy_type": ["generalized"],
        "epilepsy_hemisphere": None,
        "epilepsy_lobe": None,
    }
    X, y, keep_subjects = exclude_subjects(
        features, y, subjects, roots, exclusion_criteria
    )

    # Create classification model
    max_features = X.shape[1]
    rf_model_params = {
        "n_estimators": 1000,
        "max_features": max_features,
        "n_jobs": -1,
        "random_state": random_state,
    }
    lr_model_params = {
        "n_jobs": -1,
        "random_state": random_state,
        "penalty": "l1",
        "solver": "liblinear",
    }

    if clf_name == "rf":
        clf = RandomForestClassifier(**rf_model_params)
    elif clf_name == "sporf":
        # only used if you installed cysporf
        clf = ObliqueForestClassifier(**rf_model_params)
    elif clf_name == "lr":
        clf = LogisticRegression(**lr_model_params)

    steps = []
    if clf_name == "lr":
        steps.append(("scalar", StandardScaler()))
    steps.append(("clf", clf))

    clf = Pipeline(steps)

    # fit on entire dataset
    clf.fit(X, y)

    scoring_funcs = {
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "cohen_kappa_score": make_scorer(cohen_kappa_score),
        "roc_auc": roc_auc_score,  #  "roc_auc",  # roc_auc_score,
        "f1": f1_score,
        "recall": recall_score,
        "precision": precision_score,
        "jaccard": jaccard_score,
        "average_precision": average_precision_score,
        "neg_brier_score": brier_score_loss,
    }

    # evaluate the model performance
    train_scores = dict()
    for score_name, score_func in scoring_funcs.items():
        y_pred_proba = clf.predict_proba(X)
        # score = score_func(y, y_pred_proba)
        if score_name not in ['balanced_accuracy', 'cohen_kappa_score']:
            if score_name == "specificity":
                score_func = make_scorer(score_func, pos_label=0)
            else:
                score_func = make_scorer(score_func)
        score = score_func(clf, X, y)

        train_scores[score_name] = score

    for idx in np.unique(y):
        print(f"Class {idx} has ", len(np.argwhere(y == idx)))
    y_pred = clf.predict(X)

    scoring_funcs = {
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "cohen_kappa_score": make_scorer(cohen_kappa_score),
        "roc_auc": "roc_auc",  # roc_auc_score,
        "f1": "f1",  # f1_score,
        "recall": "recall",  # makerecall_score,
        "precision": "precision",  # precision_score,
        "jaccard": "jaccard",  # jaccard_score,
        "average_precision": "average_precision",  # average_precision_score,
        "neg_brier_score": "neg_brier_score",  # brier_score_loss,
    }
    cv_scores = {}
    for cv_name, cv in cvs.items():
        # run cross-validation
        scores = cross_validate(
            clf,
            X,
            y,
            groups=keep_subjects,
            cv=cv,
            scoring=scoring_funcs,
            return_estimator=True,
            return_train_score=False,
            n_jobs=-1,
            error_score="raise",
        )

        # get the estimators
        estimators = scores.pop("estimator")
        cv_scores[cv_name] = scores

    result_df = pd.DataFrame()

    idx = 0
    result_df["heatmaps"] = ""
    result_df.at[1, "heatmaps"] = str(names)
    result_df["exp"] = idx
    result_df["data_shape"] = str(X.shape)
    result_df["n_splits"] = n_splits
    result_df["n_classes"] = len(np.unique(y))
    result_df["clf"] = clf_name

    # keep track of training set scores
    for name, score in train_scores.items():
        result_df[f"train_{name}"] = score

    # keep track of cv scores
    for name, scores in cv_scores.items():
        for metric, score in scores.items():
            if not metric.startswith("test_"):
                continue
            result_df[f"{name}_{metric}"] = ''
            result_df.at[1, f"{name}_{metric}"] = score
            result_df[f"{name}_{metric}_avg"] = np.mean(score)
            result_df[f"{name}_{metric}_std"] = np.std(score)

    # append to the dataframe that exists
    if result_fname.exists():
        buff_df = pd.read_csv(result_fname, index_col=None)
        result_df = pd.concat((buff_df, result_df), axis=0)
    result_df.to_csv(result_fname, index=None)
    return result_df
