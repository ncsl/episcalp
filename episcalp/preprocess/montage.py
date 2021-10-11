def get_standard_1020_channels():
    return [
        "Fp1",
        "Fp2",
        "F3",
        "F4",
        "F7",
        "F8",
        "P3",
        "P4",
        # "C3",  # usually referenced against
        # "C4",  # usually referenced against
        "P7",
        "P8",
        "O1",
        "O2",
        "T3",
        "T4",
        "T5",
        "T6",
        "T7",
        "T8",
        "Cz",
        "Pz",
        "Fz",
    ]


def get_standard_1020_bipolar_pairs():
    """
    Get pairs (cathode, anode) for standard bipolar longitudinal montage.

    Returns
    -------
    bipolar_pairs: list(str, str)

    """
    return [
        ["Fp1", "F3"],
        ["F3", "C3"],
        ["C3", "P3"],
        ["P3", "O1"],
        ["Fp2", "F4"],
        ["F4", "C4"],
        ["C4", "P4"],
        ["P4", "O2"],
        ["Fp1", "F7"],
        ["F7", "T3"],
        ["T3", "T5"],
        ["T5", "O1"],
        ["Fp2", "F8"],
        ["F8", "T4"],
        ["T4", "T6"],
        ["T6", "O2"],
        ["Fz", "Cz"],
        ["Cz", "Pz"],
    ]


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
            "right_occipital": ["O2"],
        }
    else:
        lobe_dict = {
            "frontal": ["Fp1", "F3", "F7", "Fp2", "F4", "F8"],
            "temporal": ["T3", "T5", "T7", "T4", "T6", "T8"],
            "parietal": ["P7", "C3", "P8", "C4"],
            "occipital": ["O1", "O2"],
        }
    return lobe_dict
