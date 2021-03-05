from pathlib import Path
import logging as logger
from typing import Union

import numpy as np

import mne
from mne_bids import write_raw_bids, get_entities_from_fname, BIDSPath
from mne_bids.config import reader
from mne_bids.path import _parse_ext
from mne_bids.tsv_handler import _from_tsv, _to_tsv
from mne_bids.utils import _write_json

from episcalp.bids.utils import _channel_text_scrub, preprocess_raw


def _replace_ext(fname, ext, verbose=False):
    """Replace the extension of the fname with the passed extension."""
    if verbose:
        print(f"Trying to replace {fname} with extension {ext}")

    fname, _ext = _parse_ext(fname, verbose=verbose)
    if not ext.startswith("."):
        ext = "." + ext

    return fname + ext


def write_epitrack_bids(source_path,
                        bids_path,
                        montage,
                        line_freq=60,
                        overwrite=False,
                        verbose=False):
    """
    Write a scalp eeg file to bids format.

    Parameters
    ----------
    source_path: Path
        Path of the sourcefile to convert.
    bids_path: BIDSPath
        BIDSPath object corresponding to the converted file.
    montage: str
        Name of the montage for this recording
    line_freq: int
        Line frequency for the data. Either 60 or 50.
    overwrite: bool
        Whether to overwrite the file if it already exists
    verbose: bool
        Whether to have verbose outputs to console.

    Returns
    -------

    """
    # Grab identifiers from bids_path
    acquisition = bids_path.acquisition
    bids_root = bids_path.root
    datatype = bids_path.datatype

    if datatype is None:
        if acquisition in ["seeg", "ecog"]:
            datatype = "ieeg"
        elif acquisition == "eeg":
            datatype = "eeg"

    # Use mne-bids provided method to automatically determine which function to use to read in the
    # raw data
    extension = Path(source_path).suffix
    _reader = reader.get(extension)
    if _reader is None:
        # Checks to make sure the files are the correct type. Should never throw an error because of the way we structured
        # the call to this function, but good to have explicit check in case we change the loading
        if extension not in [".edf", ".vhdr"]:
            msg = (
                f"Attempted to upload a recording file of type {extension}. Only EDF (.edf) or "
                f"BrainVision (.vhdr) are allowed."
            )
            logger.exception(msg)
            raise ValueError(msg)
        raise RuntimeError(f"Reading {source_path} is not supported yet...")

    # Read in raw data into an mne.io.Raw object
    raw = _reader(source_path)

    # Get events from the raw data structure
    events, events_id = mne.events_from_annotations(raw, event_id=None)
    _events_id = events_id.copy()
    events_id = dict()
    for key, val in _events_id.items():
        events_id[key.strip()] = val
    if np.array(events).size == 0:
        events = None
        events_id = None

    # Ensure the line_freq is set
    if raw.info["line_freq"] is None:
        raw.info["line_freq"] = line_freq

    # Scrub channel names to remove extra characters like 'EEG' and '-Ref'
    # Set appropriate channel types
    raw = preprocess_raw(raw, bids_path, montage)

    # Set the montage
    if acquisition == "eeg":
        # make standard_1020 montage
        montage = mne.channels.make_standard_montage(montage)

        # find non-matching ch_names
        montage_chs = montage.ch_names
        eeg_picks = mne.pick_types(raw.info, eeg=True)
        eeg_chs = [raw.ch_names[pick] for pick in eeg_picks]
        non_eeg_chs = [ch for ch in eeg_chs if ch not in montage_chs]
        raw.set_channel_types({ch: "misc" for ch in non_eeg_chs})

        # set the montage now
        try:
            raw.set_montage(montage)
        except Exception as e:
            logger.exception(e)

    print(f"RAW CHANNELS: {raw.ch_names}")
    # Write the data to a file
    bids_fpath = write_raw_bids(
        raw,
        bids_path,
        overwrite=overwrite,
        verbose=verbose,
        # **kwargs,
    )
    return bids_fpath


def append_original_fname_to_scans(
    orig_fname: str,
    bids_root: Union[str, Path],
    bids_fname: str,
    overwrite: bool = True,
    verbose: bool = True,
):
    """Append the original filename to *scans.tsv in BIDS data structure.

    This will also create a sidecar *scans.json file alongside to document
    a description of the added column in the scans.tsv file.

    Parameters
    ----------
    orig_fname : str
        The original base filename that will be added into the
        'original_filename' columnn.
    bids_root : str | Path
        The root to the BIDS dataset.
    bids_fname : str | BIDSPath
        The BIDS filename of the BIDSified dataset. This should
        correspond to a specific 'filename' in the *scans.tsv file.
    overwrite : bool
        Whether or not to overwrite the row.
    verbose : bool
    """
    # create a BIDS path object noting that you only need
    # subject and session to define the *scans.tsv file
    entities = get_entities_from_fname(bids_fname)
    bids_path = BIDSPath(entities["subject"], entities["session"], root=bids_root)
    scans_fpath = bids_path.copy().update(suffix="scans", extension=".tsv")

    # make sure the path actually exists
    if not scans_fpath.fpath.exists():
        raise OSError(
            f"Scans.tsv file {scans_fpath} does not "
            f"exist. Please check the path to ensure it is "
            f"valid."
        )
    scans_tsv = _from_tsv(scans_fpath)

    # new filenames
    filenames = scans_tsv["filename"]
    ind = [i for i, fname in enumerate(filenames) if str(bids_fname) in fname]

    if len(ind) > 1:  # pragma: no cover
        msg = (
            "This should not happen. All scans should "
            "be uniquely identifiable from scans.tsv file. "
            "The current scans file has these filenames: "
            f"{filenames}."
        )
        logger.exception(msg)
        raise RuntimeError(msg)
    if len(ind) == 0:
        msg = (
            f"No filename, {bids_fname} found. "
            f"Scans.tsv has these files: {filenames}."
        )
        logger.exception(msg)
        raise RuntimeError(msg)

    # write scans.json
    scans_json_path = _replace_ext(scans_fpath, "json")
    scans_json = {
        "original_filename": "The original filename of the converted BIDs dataset. "
        "Provides possibly ictal/interictal, asleep/awake and "
        "clinical seizure grouping (i.e. SZ2PG, etc.)."
    }
    _write_json(scans_json_path, scans_json, overwrite=True, verbose=verbose)

    # write in original filename
    if "original_filename" not in scans_tsv.keys():
        scans_tsv["original_filename"] = ["n/a"] * len(filenames)
    if scans_tsv["original_filename"][ind[0]] == "n/a" or overwrite:
        scans_tsv["original_filename"][ind[0]] = orig_fname
    else:
        logger.warning(
            "Original filename has already been written here. "
            f"Skipping for {bids_fname}. It is written as "
            f"{scans_tsv['original_filename'][ind[0]]}."
        )
        return

    # write the scans out
    _to_tsv(scans_tsv, scans_fpath)
