from typing import Union, List
import logging as logger
import mne
import json
import os
from mne_bids.tsv_handler import _to_tsv, _from_tsv
from mne_bids.utils import _write_json


from episcalp.utils.standard_1020_montage import get_standard_1020_montage


def update_participants_info(
    root, subject, key, value, description=None, levels=None, units=None
):
    """Update ``participants`` tsv in BIDS dataset for a specific subject.

    Will also update the corresponding ``participants.json`` file if the key
    does not exist yet. Will try to add a ``description``, ``levels`` and ``units``
    of the column to the JSON file.

    Parameters
    ----------
    root : str | pathlib.Path
    subject : str
        The subject of the BIDS dataset. Corresponds to ``sub-`` entity.
    key : str
        The column name of the ``participants.tsv`` file. If it does not
        exist in the file, it will be created.
    value : str
        The value for the participant's column key to set.
    description : str
        The description of the column key. Added to the ``participants.json``
        if the column ``key`` does not exist yet.
    levels : dict
    units : str
    """
    participants_json_fname = os.path.join(root, "participants.json")
    participants_tsv_fname = os.path.join(root, "participants.tsv")

    # don't overwrite existing data
    with open(participants_json_fname, "r") as fin:
        participant_field = json.load(fin)

    # check is key inside here
    if key in participant_field:
        participant_field_key = participant_field[key]
    else:
        participant_field[key] = dict()
        participant_field_key = participant_field[key]

    if description is not None:
        participant_field_key.update(
            {
                "Description": description,
            }
        )

    if levels is not None:
        participant_field_key.update({"Levels": levels})
    if units is not None:
        participant_field_key.update({"Units": units})

    if not subject.startswith("sub-"):
        subject = f"sub-{subject}"

    participant_field[key] = participant_field_key
    _write_json(
        participants_json_fname, participant_field, overwrite=True, verbose=False
    )
    _update_sidecar_tsv_byname(
        participants_tsv_fname, subject, key, value, index_name="participant_id"
    )


def _update_sidecar_tsv_byname(
    sidecar_fname: str,
    name: Union[List, str],
    colkey: str,
    val: str,
    allow_fail=False,
    index_name="name",
    verbose=False,
):
    """Update a sidecar JSON file with a given key/value pair.

    Parameters
    ----------
    sidecar_fname : str
        Full name of the data file
    name : str
        The name of the row in column "name"
    colkey : str
        The lower-case column key in the sidecar TSV file. E.g. "type"
    val : str
        The corresponding value to change to in the sidecar JSON file.
    """
    # convert to lower case and replace keys that are
    colkey = colkey.lower()

    if isinstance(name, list):
        names = name
    else:
        names = [name]

    # load in sidecar tsv file
    sidecar_tsv = _from_tsv(sidecar_fname)

    for name in names:
        # replace certain apostrophe in Windows vs Mac machines
        name = name.replace("’", "'")

        if allow_fail:
            if name not in sidecar_tsv[index_name]:
                warnings.warn(
                    f"{name} not found in sidecar tsv, {sidecar_fname}. Here are the names: {sidecar_tsv['name']}"
                )
                continue

        # get the row index
        row_index = sidecar_tsv[index_name].index(name)

        # write value in if column key already exists,
        # else write "n/a" in and then adjust matching row
        if colkey in sidecar_tsv.keys():
            sidecar_tsv[colkey][row_index] = val
        else:
            sidecar_tsv[colkey] = ["n/a"] * len(sidecar_tsv[index_name])
            sidecar_tsv[colkey][row_index] = val

    _to_tsv(sidecar_tsv, sidecar_fname)


def _channel_text_scrub(raw: mne.io.BaseRaw):
    """
    Clean and formats the channel text inside a MNE-Raw data structure.

    Parameters
    ----------
    raw : MNE-raw data structure
    """

    def _reformatchanlabel(label):  # noqa
        """Process a single channel label.

        To make sure it is:

        - upper case
        - removed unnecessary strings (POL, eeg, -ref)
        - removed empty spaces
        """

        # hard coded replacement rules
        # label = str(label).replace("POL ", "").upper()
        label = str(label).replace("POL", "")
        label = label.replace("EEG", "").replace("-REF", "")  # .replace("!","1")
        label = label.replace("-Ref", "")  # .replace("!","1")

        # replace "Grid" with 'G' label
        label = label.replace("GRID", "G")
        # for BIDS format, you cannot have blank channel name
        if label == "":
            label = "N/A"
        return label

    # apply channel scrubbing
    # raw = raw.rename_channels(lambda x: x.upper())

    # encapsulated into a try statement in case there are blank channel names
    # after scrubbing these characters
    try:
        raw = raw.rename_channels(
            lambda x: x.strip(".")
        )  # remove dots from channel names
        raw = raw.rename_channels(
            lambda x: x.strip("-")
        )  # remove dashes from channel names
    except ValueError as e:
        logger.error(f"Ran into an issue when debugging: {raw.info}")
        logger.exception(e)

    raw = raw.rename_channels(lambda x: x.replace(" ", ""))
    raw = raw.rename_channels(
        lambda x: x.replace("â€™", "'")
    )  # remove dashes from channel names
    raw = raw.rename_channels(
        lambda x: x.replace("`", "'")
    )  # remove dashes from channel names
    raw = raw.rename_channels(lambda x: _reformatchanlabel(x))

    return raw


def bids_preprocess_raw(raw, bids_path, montage):
    """Preprocess raw channel names and types.

    Will look for regular expression for DC, EMG, ECG, EOG
    channels to set their channel types.

    Will also set channel types to 'eeg', 'ecog', or 'seeg',
    depending on the ``bids_path.acquisition`` parameter.

    Parameters
    ----------
    raw : mne.io.Raw
        The Raw object from MNE-Python
    bids_path : BIDSPath
        The BIDs path object.
    montage : str
        The scalp EEG montage

    Returns
    -------
    raw : mne.io.Raw
        The preprocessed Raw object.
    """
    # acquisition in EZTrack is encoded for eeg,ecog,seeg
    acquisition = bids_path.acquisition
    datatype = bids_path.datatype

    # these are considered Reference channels in scalp EEG
    ref_chs = ["A1", "A2", "M1", "M2", "E"]

    bio_chs = ["X1", "X2", "X3", "X4", "X5", "X6", "X7"]

    # set channel types
    if acquisition in ["seeg", "ecog", "eeg"]:
        logger.info(f"Setting channel types to: {acquisition}")
        raw.set_channel_types(
            {raw.ch_names[i]: acquisition for i in mne.pick_types(raw.info, eeg=True)}
        )

    # reformat channel text if necessary
    raw = _channel_text_scrub(raw)
    print(
        "Finished scrubbing channel text, and now the channel names are: ", raw.ch_names
    )
    # set DC channels -> MISC for now
    picks = mne.pick_channels_regexp(raw.ch_names, regexp="DC|[$]")
    raw.set_channel_types({raw.ch_names[pick]: "misc" for pick in picks})

    picks = mne.pick_channels_regexp(raw.ch_names, regexp="PULSE")
    raw.set_channel_types({raw.ch_names[pick]: "bio" for pick in picks})

    picks = mne.pick_channels_regexp(raw.ch_names, regexp="CO2")
    raw.set_channel_types({raw.ch_names[pick]: "bio" for pick in picks})

    picks = mne.pick_channels_regexp(raw.ch_names, regexp="ETCO2")
    raw.set_channel_types({raw.ch_names[pick]: "bio" for pick in picks})

    picks = mne.pick_channels_regexp(raw.ch_names, regexp="SPO2")
    raw.set_channel_types({raw.ch_names[pick]: "bio" for pick in picks})

    # set bio channels (e.g. EKG, EMG, EOG)
    picks = mne.pick_channels_regexp(raw.ch_names, regexp="EKG|ECG")
    raw.set_channel_types({raw.ch_names[pick]: "ecg" for pick in picks})
    picks = mne.pick_channels_regexp(raw.ch_names, regexp="EMG")
    raw.set_channel_types({raw.ch_names[pick]: "emg" for pick in picks})
    picks = mne.pick_channels_regexp(raw.ch_names, regexp="EOG")
    raw.set_channel_types({raw.ch_names[pick]: "eog" for pick in picks})

    if datatype == "eeg" and montage == "standard_1020":
        for ch in ref_chs:
            if ch in raw.ch_names:
                raw.set_channel_types({ch: "misc"})
        for ch in bio_chs:
            if ch in raw.ch_names:
                raw.set_channel_types({ch: "bio"})
        montage_chs = get_standard_1020_montage()
        ch_names = raw.ch_names
        ch_types = raw.get_channel_types()
        current_eeg_chs = [
            ch for idx, ch in enumerate(ch_names) if ch_types[idx] == "eeg"
        ]

        print("Current EEG channels...")
        for ch in current_eeg_chs:
            if ch not in montage_chs:
                raw.set_channel_types({ch: "misc"})
    elif montage != "standard_1020":
        raise RuntimeError(
            f"Montage {montage} isnt supported. Did you make a mistake, or did you mean standard_1020?"
        )
    return raw
