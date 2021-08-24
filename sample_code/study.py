import collections
import json
import os
from pathlib import Path
from typing import Dict


import numpy as np
from eztrack.io import DerivativeNumpy
from joblib import cpu_count
from sklearn.base import BaseEstimator
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    balanced_accuracy_score,
    accuracy_score,
)
from sklearn.model_selection import GroupKFold, cross_validate
from sample_code.features import (
    calculate_entropy,
    calculate_variance,
    calculate_skew,
    calculate_kurtosis,
    calculate_kl_div,
    calculate_distribution,
    get_spike_rate,
    get_max_spike_rate,
    get_lobe_spike_rate,
    get_slowing_count
)
from sample_code.utils import _get_feature_deriv_path

feature_types = {
    "entropy": calculate_entropy,
    "variance": calculate_variance,
    "skew": calculate_skew,
    "kurtosis": calculate_kurtosis,
    "kldiv": calculate_kl_div,
    "total_spike_rate": get_spike_rate,
    "max_spike_rate": get_max_spike_rate,
    "frontal_lobe_spike_rate": get_lobe_spike_rate,
    "temporal_lobe_spike_rate": get_lobe_spike_rate,
    "parietal_lobe_spike_rate": get_lobe_spike_rate,
    "occipital_lobe_spike_rate": get_lobe_spike_rate,
    'total_outlier_windows': get_slowing_count,
}

# make jobs use half the CPU count
num_cores = cpu_count() // 2


def _evaluate_model(
        clf_func,
        model_params,
        window,
        train_inds,
        X_formatted,
        y,
        groups,
        cv,
        dropped_inds=None,
):
    y = np.array(y).copy().squeeze()
    groups = np.array(groups).copy()
    train_inds = train_inds.copy()

    # if dropped_inds:
    #     for ind in dropped_inds:
    #         # if ind in train_inds:
    #         where_ind = np.where(train_inds >= ind)[0]
    #         train_inds[where_ind] -= 1
    #         train_inds = train_inds[:-1]
    #         # delete index in y, groups
    #         y = np.delete(y, ind)
    #         groups = np.delete(groups, ind)

    # instantiate model
    if clf_func == RandomForestClassifier:
        # instantiate the classifier
        clf = clf_func(**model_params)
    else:
        clf = clf_func

    # note that training data (Xtrain, ytrain) will get split again
    Xtrain, ytrain = X_formatted[train_inds, ...], y[train_inds]
    groups_train = groups[train_inds]

    # perform CV using Sklearn
    scoring_funcs = {
        "roc_auc": roc_auc_score,
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
        "average_precision": average_precision_score,
    }
    scores = cross_validate(
        clf,
        Xtrain,
        ytrain,
        groups=groups_train,
        cv=cv,
        scoring=list(scoring_funcs.keys()),
        return_estimator=True,
        return_train_score=True,
    )

    return scores


def tune_hyperparameters(
        clf_func,
        unformatted_X,
        y,
        groups,
        train_inds,
        test_inds,
        hyperparameters,
        dataset_params,
        **model_params,
):
    """Perform hyperparameter tuning.
    Pass in X and y dataset that are unformatted yet, and then follow
    a data pipeline that:
    - create the formatted dataset
    - applies hyperparameters
    - cross-validate
    Parameters
    ----------
    clf_func :
    unformatted_X :
    y :
    groups :
    train_inds :
    test_inds :
    hyperparameters :
    dataset_params :
    model_params :
    Returns
    -------
    master_scores: dict
    """

    # CV object to perform training of classifier
    # create Grouped Folds to estimate the mean +/- std performancee
    n_splits = 5
    cv = GroupKFold(n_splits=n_splits)

    # track all cross validation score dictionaries
    master_scores = []

    print(f"Using classifier: {clf_func}")
    for idx, hyperparam in enumerate(hyperparameters):
        # extract the hyperparameter explicitly
        window, threshold, weighting_func = hyperparam
        hyperparam_str = (
            f"window-{window}_threshold-{threshold}_weightfunc-{weighting_func}"
        )
        # apply the hyperparameters to the data
        #         print(unformatted_X.shape)
        X_formatted, dropped_inds = format_supervised_dataset(
            unformatted_X,
            **dataset_params,
            window=window,
            threshold=threshold,
            weighting_func=weighting_func,
        )

        scores = _evaluate_model(
            clf_func,
            model_params,
            window,
            train_inds,
            X_formatted,
            y,
            groups,
            cv,
            dropped_inds=dropped_inds,
        )
        # # get the best classifier based on pre-chosen metric
        # best_metric_ind = np.argmax(scores["test_roc_auc"])
        # best_estimator = scores["estimator"][best_metric_ind]
        #
        # # evaluate on the testing dataset
        # X_test, y_test = X_formatted[test_inds, ...], y[test_inds]
        # groups_test = groups[test_inds]
        #
        # y_pred_prob = best_estimator.predict_proba(X_test)[:, 1]
        # y_pred = best_estimator.predict(X_test)
        #
        # # store analysis done on the validation group
        # scores["validate_groups"] = groups_test
        scores["hyperparameters"] = hyperparam
        # scores["validate_ytrue"] = y_test
        # scores["validate_ypred_prob"] = y_pred_prob
        #
        # # pop estimator
        # scores.pop('estimator')
        # scores['estimator'] = best_estimator
        #
        # # resample the held-out test data via bootstrap
        # # test_sozinds_list = dataset_params['sozinds_list'][test_inds]
        # # test_onsetwin_list = dataset_params['onsetwin_list'][test_inds]
        # # X_boot, y_boot, sozinds, onsetwins = resample(X_test, y_test,
        # #                                               test_sozinds_list,
        # #                                               test_onsetwin_list,
        # #                                               n_samples=500)
        #
        # # store ROC curve metrics on the held-out test set
        # fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label=1)
        # fnr, tnr, neg_thresholds = roc_curve(y_test, y_pred_prob, pos_label=0)
        # scores["validate_fpr"] = fpr
        # scores["validate_tpr"] = tpr
        # scores["validate_fnr"] = fnr
        # scores["validate_tnr"] = tnr
        master_scores.append(scores)

    return master_scores


def generate_patient_features(deriv_path, model_name, features, subjects=None, verbose=True):
    deriv_path = Path(deriv_path)
    patient_result_dict = collections.defaultdict(list)

    # get path to this specific feature and its extension used
    feature_deriv_path, ext = _get_feature_deriv_path(deriv_path, model_name)
    if subjects is None:
        subjects = [
            fpath.name for fpath in feature_deriv_path.glob("*") if fpath.is_dir()
        ]
    for sub in subjects:
        if "sub" in sub:
            sub = sub.split("-")[1]
        feature_deriv_dir = Path(feature_deriv_path) / f"sub-{sub}"
        if model_name == "sourcesink":
            deriv_fname = f"sub-{sub}_ses-initialvisit_task-monitor_run-01_desc-ssindmatrix_eeg.npy"
        elif model_name == "fragility":
            deriv_fname = f"sub-{sub}_ses-initialvisit_task-monitor_run-01_desc-perturbmatrix_eeg.npy"
        else:
            return None
        deriv_fpath = Path(feature_deriv_dir) / deriv_fname
        if not deriv_fpath.exists():
            deriv_fname = deriv_fname.replace("initialvisit", "awake")
            deriv_fpath = Path(feature_deriv_dir) / deriv_fname
        deriv = DerivativeNumpy(deriv_fpath)
        deriv_arr = deriv.get_data()

        dist = calculate_distribution(deriv_arr, True)

        for feature in features:
            feature_func = feature_types[feature]
            feature_fname = f"{deriv_fname.split('_desc')[0]}_desc-{feature}_eeg.npy"
            feature_fpath = Path(feature_deriv_dir) / feature_fname
            val = feature_func(dist)
            np.save(feature_fpath, val)
    return None


def generate_spike_feature(spike_patient_dict, feature_name, include_subject_groups, extra_params=None):
    unformatted_X = []
    y = []
    pt_ids = []
    groups = []
    feature_func = feature_types[feature_name]
    for subject_id, spike_dict in spike_patient_dict.items():
        if extra_params is not None:
            x = feature_func(spike_dict, **extra_params)
        else:
            x = feature_func(spike_dict)
        if int(subject_id) < 100 and 'non-epilepsy' in include_subject_groups.keys():
            unformatted_X.append(x)
            y.append(include_subject_groups['non-epilepsy'])
            pt_ids.append(subject_id)
            groups.append(0)
        elif int(subject_id) > 200 and 'epilepsy-abnormal' in include_subject_groups.keys():
            unformatted_X.append(x)
            y.append(include_subject_groups['epilepsy-abnormal'])
            pt_ids.append(subject_id)
            groups.append(2)
        elif 100 < int(subject_id) < 200 and 'epilepsy-normal' in include_subject_groups.keys():
            unformatted_X.append(x)
            y.append(include_subject_groups['epilepsy-normal'])
            pt_ids.append(subject_id)
            groups.append(1)
    y_arr = np.array(y).flatten()
    groups_arr = np.array(groups).flatten()
    pt_ids_arr = np.array(pt_ids).flatten()
    return unformatted_X, y_arr, groups_arr, pt_ids_arr


def generate_slowing_features(slowing_patient_dict, feature_name, include_subject_groups):
    unformatted_X = []
    y = []
    pt_ids = []
    groups = []
    feature_func = feature_types[feature_name]
    for subject_id, slowing_dict in slowing_patient_dict.items():
        x = feature_func(slowing_dict)
        if int(subject_id) < 100 and 'non-epilepsy' in include_subject_groups.keys():
            unformatted_X.append(x)
            y.append(include_subject_groups['non-epilepsy'])
            pt_ids.append(subject_id)
            groups.append(0)
        elif int(subject_id) > 200 and 'epilepsy-abnormal' in include_subject_groups.keys():
            unformatted_X.append(x)
            y.append(include_subject_groups['epilepsy-abnormal'])
            pt_ids.append(subject_id)
            groups.append(2)
        elif 100 < int(subject_id) < 200 and 'epilepsy-normal' in include_subject_groups.keys():
            unformatted_X.append(x)
            y.append(include_subject_groups['epilepsy-normal'])
            pt_ids.append(subject_id)
            groups.append(1)
    y_arr = np.array(y).flatten()
    groups_arr = np.array(groups).flatten()
    pt_ids_arr = np.array(pt_ids).flatten()
    return unformatted_X, y_arr, groups_arr, pt_ids_arr


def load_patient_dict(deriv_path, feature_name, task=None, subjects=None, verbose=True):
    """Load comparative features patient dictionary of results."""
    deriv_path = Path(deriv_path)
    patient_result_dict = collections.defaultdict(list)

    freq_bands = ["delta", "theta", "alpha", "beta", "gamma", "highgamma"]

    # get path to this specific feature and its extension used
    feature_deriv_path, ext = _get_feature_deriv_path(deriv_path, feature_name)
    if subjects is None:
        subjects = [
            fpath.name for fpath in feature_deriv_path.glob("*") if fpath.is_dir()
        ]
    if any([band in feature_name for band in freq_bands]):
        band = feature_name.split("-")[0]
    else:
        band = None

    if verbose:
        print(f"Loading data from: {feature_deriv_path}")
        print(subjects)
        print(band, task, feature_name)

    for subject in subjects:
        if feature_name in freq_bands:
            patient_result_dict[subject] = load_patient_tfr(
                feature_deriv_path, subject, feature_name, task=task,
            )
        else:
            patient_result_dict[subject] = load_patient_graphstats(
                feature_deriv_path,
                subject,
                kind="ieeg",
                band=band,
                task=task,
                verbose=False,
            )
    # subj_results = Parallel(n_jobs=num_cores)(
    #     delayed(_load_features)(
    #         feature_name, subject, feature_deriv_path, freq_bands, band, task
    #     )
    #     for _, subject in enumerate(tqdm(subjects))
    # )
    # print(len(subj_results))
    # print(subj_results[0])
    # # transform list of dicts to dict
    # patient_result_dict = {
    #     subject: results for x in subj_results for subject, results in x.items()
    # }

    if verbose:
        print("Got ", len(patient_result_dict), " patients")
        print(patient_result_dict.keys())
    return patient_result_dict


def determine_feature_importances(clf, X, y, n_jobs):
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        estimator=clf,
        X=X,
        y=y,
        scoring="roc_auc",
        n_repeats=5,
        n_jobs=n_jobs,
        random_state=1,
    )

    std = result.importances_std
    indices = np.argsort(result.importances_mean)[::-1]

    return result


def compute_auc_optimism(clf, X, y, n_boot=500, alpha=0.05):
    from sklearn.utils import resample
    from sklearn.metrics import roc_auc_score

    # original auc
    orig_metric = roc_auc_score(y, clf.predict_proba(X))

    score_biases = []

    for boot_idx in range(n_boot):
        boot_X, boot_y = resample(X, y, replace=True, n_samples=len(y), stratify=y)
        clf.fit(boot_X)

        # bootstrap sample score
        y_predict_proba = clf.predict_proba(boot_X)
        C_boot_roc_auc = roc_auc_score(boot_y, y_predict_proba)

        # original sample score
        y_predict_proba = clf.predict_proba(X)
        C_orig_roc_auc = roc_auc_score(y, y_predict_proba)

        # store the bias
        score_biases.append(C_boot_roc_auc - C_orig_roc_auc)

    # compute CI
    lb = np.percentile(score_biases, q=alpha // 2)
    ub = np.percentile(score_biases, q=1 - alpha // 2)

    # compute optimism
    optimism = np.mean(score_biases)
    ci = [lb, ub]
    return orig_metric - optimism, ci


def _show_calibration_curve(estimators, X, y, name):
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    import matplotlib.pyplot as plt
    import seaborn as sns

    #
    sns.set_context("paper", font_scale=1.5)
    plt.figure(figsize=(7, 10))
    ax1 = plt.subplot2grid((2, 1), (0, 0), rowspan=1)
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    clf = estimators[0]
    y_predict_prob = clf.predict_proba(X)
    prob_pos = y_predict_prob[:, 1]
    # compute calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y, prob_pos, n_bins=100, strategy="quantile"
    )

    clf_score = np.round(brier_score_loss(y, prob_pos, pos_label=np.array(y).max()), 2)

    print(clf_score)
    print(fraction_of_positives, mean_predicted_value)

    # frac_pred_vals = []
    # mean_pred_values = np.linspace(0, 1.0, 200)
    # brier_scores = []
    # for i, clf in enumerate(estimators):
    #     y_predict_prob = clf.predict_proba(X)
    #     prob_pos = y_predict_prob[:, 1]
    #     # compute calibration curve
    #     fraction_of_positives, mean_predicted_value = calibration_curve(
    #         y, prob_pos, n_bins=10, strategy="quantile"
    #     )
    #
    #     clf_score = np.round(
    #         brier_score_loss(y, prob_pos, pos_label=np.array(y).max()), 2
    #     )
    #
    #     # create a linear interpolation of the calibration
    #     interp_frac_positives = np.interp(
    #         mean_pred_values, mean_predicted_value, fraction_of_positives
    #     )
    #     interp_frac_positives[0] = 0.0
    #
    #     # store curves + scores
    #     brier_scores.append(clf_score)
    #     frac_pred_vals.append(interp_frac_positives)
    #
    # mean_frac_pred_values = np.mean(frac_pred_vals, axis=0)
    # ax1.plot(
    #     mean_pred_values,
    #     mean_frac_pred_values,
    #     "s-",
    #     label=rf"{name.capitalize()} ({np.round(np.mean(brier_scores),2)} $\pm$ {np.round(np.std(brier_scores), 2)}",
    # )
    #
    # # get upper and lower bound for tpr
    # std_fpv = np.std(frac_pred_vals, axis=0)
    # tprs_upper = np.minimum(mean_frac_pred_values + std_fpv, 1)
    # tprs_lower = np.maximum(mean_frac_pred_values - std_fpv, 0)
    # ax1.fill_between(
    #     mean_pred_values,
    #     tprs_lower,
    #     tprs_upper,
    #     color="grey",
    #     alpha=0.2,
    #     # label=r"$\pm$ 1 std. dev.",
    # )

    # actually do the plot
    ax1.plot(
        mean_predicted_value,
        fraction_of_positives,
        "s-",
        label=f"{name.capitalize()} ({clf_score})",
    )

    # set
    ax1.plot()
    ax1.set(
        ylabel="Fraction of Success Outcomes (y label of 1)",
        xlabel="Mean predicted confidence statistic",
        ylim=[-0.05, 1.05],
        title="Calibration plots  (reliability curve)",
    )
    ax1.legend(loc="lower right")
    return ax1


def show_calibration_curves(clfs, X, y):
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    import matplotlib.pyplot as plt
    import seaborn as sns

    #
    sns.set_context("paper", font_scale=1.5)
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for name, clf in clfs.items():
        print(name, clf)
        y_predict_prob = clf.predict_proba(X)
        prob_pos = y_predict_prob[:, 1]
        # compute calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y, prob_pos, n_bins=10, strategy="quantile"
        )

        clf_score = brier_score_loss(y, prob_pos, pos_label=y.max())

        ax1.plot(
            mean_predicted_value,
            fraction_of_positives,
            "s-",
            label=f"{name} ({clf_score})",
        )

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name, histtype="step", lw=2)

    ax1.plot()
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title("Calibration plots  (reliability curve)")

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)
    plt.tight_layout()
    plt.show()


def format_supervised_dataset(
        X,
):
    X_formatted = preprocessing.normalize(X, norm='l2', axis=0)
    return X_formatted, None
