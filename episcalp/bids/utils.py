import logging as logger
import mne


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
        label = str(label).replace("POL", "").upper()
        label = label.replace("EEG", "").replace("-REF", "")  # .replace("!","1")

        # replace "Grid" with 'G' label
        label = label.replace("GRID", "G")
        # for BIDS format, you cannot have blank channel name
        if label == "":
            label = "N/A"
        return label

    # apply channel scrubbing
    raw = raw.rename_channels(lambda x: x.upper())

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
    """Preprocess raw channel names and types."""
    # acquisition in EZTrack is encoded for eeg,ecog,seeg
    acquisition = bids_path.acquisition
    datatype = bids_path.datatype

    # set channel types
    if acquisition in ["seeg", "ecog", "eeg"]:
        logger.info(f"Setting channel types to: {acquisition}")
        raw.set_channel_types(
            {raw.ch_names[i]: acquisition for i in mne.pick_types(raw.info, eeg=True)}
        )

    # reformat channel text if necessary
    raw = _channel_text_scrub(raw)

    # set DC channels -> MISC for now
    picks = mne.pick_channels_regexp(raw.ch_names, regexp="DC|[$]")
    raw.set_channel_types(
        {raw.ch_names[pick]: "misc" for pick in picks}
    )

    # set bio channels (e.g. EKG, EMG, EOG)
    picks = mne.pick_channels_regexp(raw.ch_names, regexp="EKG|ECG")
    raw.set_channel_types(
        {raw.ch_names[pick]: "ecg" for pick in picks}
    )
    picks = mne.pick_channels_regexp(raw.ch_names, regexp="EMG")
    raw.set_channel_types(
        {raw.ch_names[pick]: "emg" for pick in picks}
    )
    picks = mne.pick_channels_regexp(raw.ch_names, regexp="EOG")
    raw.set_channel_types(
        {raw.ch_names[pick]: "eog" for pick in picks}
    )

    if datatype == "eeg" and montage == "standard_1020":
        ref_chs = ["A1", "A2", "M1", "M2"]
        for ch in ref_chs:
            if ch in raw.ch_names:
                raw.set_channel_types({ch: "misc"})
    return raw
