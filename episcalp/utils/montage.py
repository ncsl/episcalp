import numpy as np

from warnings import warn
import mne
from mne import channel_type, pick_channels
from mne.io.constants import FIFF


def get_best_matching_montage(ch_names, verbose=True) -> str:
    """Get the best matching template montage in mne-python.

    Parameters
    ----------
    ch_names : list
        List of channel names as strings.

    Returns
    -------
    montage : str
    """
    montages = mne.channels.get_builtin_montages()
    montages = ["standard_1020"] + montages
    best_montage_score = 0
    best_montage = None

    for montage_name in montages:
        # read in standardized montage
        montage = mne.channels.make_standard_montage(montage_name)

        # montage ch names should be in upper case to match us
        montage.ch_names = [ch.upper() for ch in montage.ch_names]

        # score this montage
        montage_score = len([ch for ch in ch_names if ch.upper() in montage.ch_names])
        if montage_score > best_montage_score:
            best_montage = montage_name
            best_montage_score = montage_score

    # get the template montage
    montage = mne.channels.make_standard_montage(best_montage)
    if verbose:
        print(f"Looked at {montages} to find good match...")
        print(f"Going to use {best_montage} as the montage.")

        # get not-matching channels
        to_drop_chs = [ch for ch in ch_names if ch not in montage.ch_names]
        print(f"Going to drop {to_drop_chs} channels...")

    return montage, best_montage


def compute_lobes_from_montage(info):
    """Hack in mne-python to map different electrodes to lobes."""
    # from mne.selection import _divide_to_regions
    # get channel groups - hashmap of channel indices
    ch_groups = _divide_to_regions(info, add_stim=False)
    ch_names = info["ch_names"]
    for group, inds in ch_groups.items():
        print(f"{group}: {np.array(ch_names)[inds]}")
    # print(ch_groups)
    return ch_groups


def set_bipolar_montage(raw, montage_scheme, drop_original_chs=True, verbose=True):
    """Rereference the raw data to the specified bipolar montage scheme."""
    ch_names = raw.ch_names
    bipolar_pairs = _get_bipolar_pairs(montage_scheme)
    for pair in bipolar_pairs:
        anode, cathode = pair
        if all(p in ch_names for p in pair):
            raw = mne.set_bipolar_reference(raw, anode, cathode, drop_refs=False)
        elif verbose:
            print(f"The pair of anode {anode} and cathode {cathode} not found in the data.")
    if drop_original_chs and bipolar_pairs is not None:
        raw.drop_channels(ch_names)
    return raw


def _get_bipolar_pairs(montage_scheme):
    """Get a list of (anode, cathode) pairs for the bipolar scheme."""
    if montage_scheme == "bipolar_longitudinal":
        pairs = [["Fp1", "F7"], ["F7", "T7"], ["T7", "P7"], ["F7", "T3"], ["T3", "T5"], ["P7", "O1"], ["T5", "O1"],
                 ["Fp1", "F3"], ["F3", "C3"], ["C3", "P3"], ["P3", "O1"], ["Fpz", "Fz"], ["Fz", "Cz"], ["Cz", "Pz"],
                 ["Pz", "Oz"], ["Fp2", "F4"], ["F4", "C4"], ["C4", "P4"], ["P4", "O2"], ["Fp2", "F8"], ["F8", "T8"],
                 ["T8", "P8"], ["P8", "O2"], ["F8", "T4"], ["T4", "T6"], ["T6", "O2"]]
        return pairs
    return None


def get_ch_renaming_map(current_ch_names, goal_ch_names):
    """Get a dict of renaming pairs where key is current name and val is goal name."""
    renaming_dict = {}
    for cur_chan in current_ch_names:
        for goal_chan in goal_ch_names:
            if cur_chan.upper() == goal_chan.upper():
                renaming_dict[cur_chan] = goal_chan
    return renaming_dict


def _divide_to_regions(info, add_stim=True):
    """Divide channels to regions by positions."""
    from scipy.stats import zscore

    picks = _pick_data_channels(info, exclude=[])
    chs_in_lobe = len(picks) // 4
    pos = np.array([ch["loc"][:3] for ch in info["chs"]])
    x, y, z = pos.T

    frontal = picks[np.argsort(y[picks])[-chs_in_lobe:]]
    picks = np.setdiff1d(picks, frontal)

    occipital = picks[np.argsort(y[picks])[:chs_in_lobe]]
    picks = np.setdiff1d(picks, occipital)

    temporal = picks[np.argsort(z[picks])[:chs_in_lobe]]
    picks = np.setdiff1d(picks, temporal)

    lt, rt = _divide_side(temporal, x)
    lf, rf = _divide_side(frontal, x)
    lo, ro = _divide_side(occipital, x)
    lp, rp = _divide_side(picks, x)  # Parietal lobe from the remaining picks.

    # Because of the way the sides are divided, there may be outliers in the
    # temporal lobes. Here we switch the sides for these outliers. For other
    # lobes it is not a big problem because of the vicinity of the lobes.
    with np.errstate(invalid="ignore"):  # invalid division, greater compare
        zs = np.abs(zscore(x[rt]))
        outliers = np.array(rt)[np.where(zs > 2.0)[0]]
    rt = list(np.setdiff1d(rt, outliers))

    with np.errstate(invalid="ignore"):  # invalid division, greater compare
        zs = np.abs(zscore(x[lt]))
        outliers = np.append(outliers, (np.array(lt)[np.where(zs > 2.0)[0]]))
    lt = list(np.setdiff1d(lt, outliers))

    l_mean = np.mean(x[lt])
    r_mean = np.mean(x[rt])
    for outlier in outliers:
        if abs(l_mean - x[outlier]) < abs(r_mean - x[outlier]):
            lt.append(outlier)
        else:
            rt.append(outlier)

    if add_stim:
        raise RuntimeError("Use mne-pythons function instead...")
        # stim_ch = _get_stim_channel(None, info, raise_error=False)
        # if len(stim_ch) > 0:
        #     for region in [lf, rf, lo, ro, lp, rp, lt, rt]:
        #         region.append(info['ch_names'].index(stim_ch[0]))
    return {
        "Left-frontal": lf,
        "Right-frontal": rf,
        "Left-parietal": lp,
        "Right-parietal": rp,
        "Left-occipital": lo,
        "Right-occipital": ro,
        "Left-temporal": lt,
        "Right-temporal": rt,
    }


def _divide_side(lobe, x):
    """Make a separation between left and right lobe evenly."""
    lobe = np.asarray(lobe)
    median = np.median(x[lobe])

    left = lobe[np.where(x[lobe] < median)[0]]
    right = lobe[np.where(x[lobe] > median)[0]]
    medians = np.where(x[lobe] == median)[0]

    left = np.sort(np.concatenate([left, lobe[medians[1::2]]]))
    right = np.sort(np.concatenate([right, lobe[medians[::2]]]))
    return list(left), list(right)


def _pick_data_channels(info, exclude="bads"):
    """Pick only data channels."""
    return pick_types(info, exclude=exclude, **_PICK_TYPES_DATA_DICT)


def pick_types(
    info,
    eeg=False,
    seeg=False,
    ecog=False,
    include=(),
    exclude="bads",
    selection=None,
):
    """Pick channels by type and names.

    Parameters
    ----------
    info : dict
        The measurement info.
    eeg : bool
        If True include EEG channels.
    seeg : bool
        Stereotactic EEG channels.
    ecog : bool
        Electrocorticography channels.
    include : list of str
        List of additional channels to include. If empty do not include any.
    exclude : list of str | str
        List of channels to exclude. If 'bads' (default), exclude channels
        in ``info['bads']``.
    selection : list of str
        Restrict sensor channels (MEG, EEG) to this list of channel names.

    Returns
    -------
    sel : array of int
        Indices of good channels.
    """
    # only issue deprecation warning if there are MEG channels in the data and
    # if the function was called with the default arg for meg
    deprecation_warn = False

    exclude = []
    nchan = info["nchan"]
    pick = np.zeros(nchan, dtype=bool)

    for param in (
        eeg,
        seeg,
        ecog,
    ):
        if not isinstance(param, bool):
            w = (
                "Parameters for all channel types (with the exception of "
                '"meg", "ref_meg" and "fnirs") must be of type bool, not {}.'
            )
            raise ValueError(w.format(type(param)))

    param_dict = dict(
        eeg=eeg,
        seeg=seeg,
        ecog=ecog,
    )
    # avoid triage if possible
    for k in range(nchan):
        ch_type = channel_type(info, k)
        pick[k] = param_dict[ch_type]

    # restrict channels to selection if provided
    if selection is not None:
        # the selection only restricts these types of channels
        sel_kind = [FIFF.FIFFV_EEG_CH]
        for k in np.where(pick)[0]:
            if (
                info["chs"][k]["kind"] in sel_kind
                and info["ch_names"][k] not in selection
            ):
                pick[k] = False

    myinclude = [info["ch_names"][k] for k in range(nchan) if pick[k]]
    myinclude += include

    if len(myinclude) == 0:
        sel = np.array([], int)
    else:
        sel = pick_channels(info["ch_names"], myinclude, exclude)

    if deprecation_warn:
        warn(
            "The default of meg=True will change to meg=False in version 0.22"
            ", set meg explicitly to avoid this warning.",
            DeprecationWarning,
        )
    return sel


_PICK_TYPES_DATA_DICT = dict(
    eeg=True,
    seeg=True,
    ecog=True
)