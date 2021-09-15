import json
from datetime import date, datetime

import numpy as np
from pathlib import Path


def _plot_roc_curve(
        mean_tpr,
        mean_fpr,
        std_tpr=0.0,
        mean_auc=0.0,
        std_auc=0.0,
        label=None,
        ax=None,
        color=None,
        plot_chance=True,
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    if ax is None:
        sns.set_context("paper", font_scale=1.5)
        fig, ax = plt.subplots(1, 1)

    if label is None:
        label = r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc)
    else:
        label = fr"{label} (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc)

    if color is None:
        color = "blue"

    # plot the actual curve
    ax.plot(
        mean_fpr,
        mean_tpr,
        color=color,
        label=label,
        lw=5,
        alpha=0.8,
    )

    # chance level
    if plot_chance:
        ax.plot(
            [0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8
        )

    # get upper and lower bound for tpr
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color=color,
        alpha=0.2,
        # label=r"$\pm$ 1 std. dev.",
    )

    # increase axis limits to see edges
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="Receiver operating characteristic curve",
    )
    ax.legend(ncol=2, loc=(1.04, 0))  # "lower right"
    return ax


def _apply_threshold(X, threshold, default_val=0.0):
    X = X.copy()
    X[X < threshold] = default_val
    return X


def _get_feature_deriv_path(deriv_path, model_name):
    deriv_dir = Path(deriv_path) / model_name / "radius1.25" / "monopolar"
    if model_name == "sourcesink":
        return  deriv_dir, ".npy"
    elif model_name == "fragility":
        return deriv_dir, ".npz"
    else:
        return deriv_dir, ".json"


def subset_patients(patient_result_dict, include_subject_groups):
    output_dict = patient_result_dict.copy()
    for ptID in patient_result_dict.keys():
        if int(ptID) > 200 and "epilepsy-abnormal" not in include_subject_groups:
            del output_dict[ptID]
        elif int(ptID) < 100 and "non-epilepsy" not in include_subject_groups:
            del output_dict[ptID]
        elif 100 < int(ptID) < 200 and "epilepsy-normal" not in include_subject_groups:
            del output_dict[ptID]
    return output_dict


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types.
    Pass to json.dump(), or json.load().
    """

    def default(self, obj):  # noqa
        if isinstance(
                obj,
                (
                        np.int_,
                        np.intc,
                        np.intp,
                        np.int8,
                        np.int16,
                        np.int32,
                        np.int64,
                        np.uint8,
                        np.uint16,
                        np.uint32,
                        np.uint64,
                ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  # This is the fix
            return obj.tolist()
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)


def _standard_channels():
    return ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "P3", "P4", "C3", "C4", "P7", "P8", "O1", "O2", "T3", "T4", "T5", "T6", "T7", "T8"]


def _standard_lobes(separate_hemispheres=False):
    if separate_hemispheres:
        lobe_dict = {
            "left_frontal": ["Fp1", "F3", "F7"],
            "right_frontal": ["Fp2", "F4", "F8"],
            "left_temporal": ["T3", "T5", "T7"],
            "right_temporal": ["T4", "T6", "T8"],
            "left_parietal": ["P7", "C3"],
            "right_parietal": ["P8", "C4"],
            "left_occipital": ["O1"],
            "right_occipital": ["O2"]
        }
    else:
        lobe_dict = {
            "frontal": ["Fp1", "F3", "F7", "Fp2", "F4", "F8"],
            "temporal": ["T3", "T5", "T7", "T4", "T6", "T8"],
            "parietal": ["P7", "C3", "P8", "C4"],
            "occipital": ["O1", "O2"]
        }
    return lobe_dict
