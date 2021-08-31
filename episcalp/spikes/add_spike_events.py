from collections import OrderedDict
from pathlib import Path
import numpy as np

import mne
from mne import events_from_annotations
from mne_bids import BIDSPath, get_entity_vals, read_raw_bids
from mne_bids.write import _write_tsv

from episcalp.utils.standard_1020_montage import get_standard_1020_montage


def add_spikes_to_events_tsv(bids_path, source_fname):
    """Add spikes detected by Persyst CLI to events.tsv file in BIDS.

    Parameters
    ----------
    bids_path : str | pathlib.Path | BIDSPath
        [description]
    source_fname : str | pathlib.Path
        [description]
    """
    # get the corresponding events.tsv file
    events_fname = bids_path.copy().update(suffix='events', extension='.tsv')

    # read in Persyst data with annotations attached
    raw_persyst = mne.io.read_raw_persyst(source_fname)

    # Duration for spikes is for some reason encoded in the description and
    # not in the duration section
    # Iterate and move the duration to the correct place
    annotations = raw_persyst.annotations
    num_comments = len(annotations)
    
    # get events from Annotations
    events, event_id = events_from_annotations(raw_persyst)

    # store the onset, durations and descriptions
    onsets = np.zeros(num_comments, float)
    durations = np.zeros(num_comments, float)
    descriptions = [''] * num_comments

    # extract spike events from Annotations object
    for ind, annot in enumerate(annotations):
        onsets[ind] = annot.get('onset', 0)
        description_ = annot.get('description', '')
        if 'spike' in description_:
            spike, ch, duration = description_.split(" ")
            durations[ind] = float(duration)
            description = f"{spike} {_fix_channel_name(ch)}"
            descriptions[ind] = description
        else:
            descriptions[ind] = description_
        
    # Add correct annotations to the base raw object
    annotations_ = mne.Annotations(onsets, durations, descriptions)

    




def add_spikes_to_events_tsv(bids_root, spike_root, ignore_subjects=None, **bids_kwargs):
    """Add spike detections to events.tsv file in BIDS.

    Parameters
    ----------
    bids_root : [type]
        [description]
    spike_root : [type]
        [description]
    ignore_subjects : [type], optional
        [description], by default None

    Raises
    ------
    RuntimeError
        [description]
    """
    subjects = get_entity_vals(bids_root, 'subject', ignore_subjects=ignore_subjects)
    subjects = [subjects[0]]
    for sub in subjects:
        bids_path_ = BIDSPath(subject=sub, root=bids_root, **bids_kwargs)
        if bids_path_.match() is None:
            bids_path_.update(session="awake")
            if bids_path_.match() is None:
                raise RuntimeError(f'The parameters {bids_path_} do not match anything')
        # Read in raw data from bids_root
        bids_path = bids_path_.copy()
        raw_base = read_raw_bids(bids_path)
        sfreq = raw_base.info['sfreq']

        bids_name = bids_path.basename

        # Read in persyst data that contains spike information
        spike_fpath = Path(spike_root) / f"sub-{sub}" / f"{bids_name}_spikes.lay"
        raw_persyst = mne.io.read_raw_persyst(spike_fpath)

        # Duration for spikes is for some reason encoded in the description and not in the duration section
        # Iterate and move the duration to the correct place
        annotations = raw_persyst.annotations
        num_comments = len(annotations)
        onsets = np.zeros(num_comments, float)
        durations = np.zeros(num_comments, float)
        descriptions = [''] * num_comments
        for ind, annot in enumerate(annotations):
            onsets[ind] = annot.get('onset', 0)
            description_ = annot.get('description', '')
            if 'spike' in description_:
                spike, ch, duration = description_.split(" ")
                durations[ind] = float(duration)
                description = f"{spike} {_fix_channel_name(ch)}"
                descriptions[ind] = description
            else:
                descriptions[ind] = description_
        # Add correct annotations to the base raw object
        annotations_ = mne.Annotations(onsets, durations, descriptions)
        raw_base.set_annotations(annotations_)

        # Write these annotations out to an events tsv file
        events_fpath = bids_path.copy().update(suffix='events', extension='.tsv')
        events, events_id = mne.events_from_annotations(raw_base)

        values = np.zeros(num_comments, float)
        for ind, description in enumerate(descriptions):
            for event, event_id in zip(events, events_id):
                if list(events_id.keys())[0] == description:
                    values[ind] = event[2]
                else:
                    values[ind] = 0

        tsv_data = OrderedDict([('onset', onsets / sfreq),
                                ('duration', durations / sfreq),
                                ('trial_type', descriptions),
                                ('value', values),
                                ('sample', onsets)])

        _write_tsv(events_fpath, tsv_data, overwrite=False)


def _fix_channel_name(ch_name):
    base_ch, ref_ch = ch_name.split("-")
    standard_chs = get_standard_1020_montage()
    for ch in standard_chs:
        if base_ch.upper() == ch.upper():
            base_ch = ch
    return f"{base_ch}-{ref_ch}"


if __name__ == "__main__":
    bids_root = Path("D:/OneDriveParent/OneDrive - Johns Hopkins/Shared Documents/40Hz-30")
    spike_root = Path("D:/OneDriveParent/OneDrive - Johns Hopkins/Shared Documents/40Hz-30/derivatives/spikes/monopolar")
    bids_kwargs = {
        "session": "initialvisit",
        "task": "monitor",
        "run": "01",
        "datatype": "eeg",
        "extension": ".vhdr"
    }
    add_spike_events(bids_root, spike_root, **bids_kwargs)