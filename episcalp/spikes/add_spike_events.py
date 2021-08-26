from collections import OrderedDict
from pathlib import Path
import numpy as np

import mne
from mne_bids import BIDSPath, get_entity_vals, read_raw_bids
from mne_bids.write import _write_tsv

from episcalp.utils.standard_1020_montage import get_standard_1020_montage


def add_spike_events(bids_root, spike_root, ignore_subjects=None, **bids_kwargs):
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

        # Write these annotations out to an events tsv file
        events_fpath = bids_path.copy().update(suffix='events', extension='.tsv')

        annotations_, num_comments, onsets, descriptions, durations = _create_spike_events(raw_persyst)
        raw_persyst.set_annotations(annotations_)

        # Write these annotations out to an events tsv file
        events, events_id = mne.events_from_annotations(raw_persyst)

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


def _create_spike_events(raw_persyst):
    sfreq = raw_persyst.info['sfreq']
    # Duration for spikes is for some reason encoded in the description and not in the duration section
    # Iterate and move the duration to the correct place
    annotations = raw_persyst.annotations
    num_comments = len(annotations)
    onsets = np.zeros(num_comments, float)
    durations = np.zeros(num_comments, float)
    descriptions = [''] * num_comments
    for ind, annot in enumerate(annotations):
        description_ = annot.get('description', '')
        if 'spike ' in description_:
            onsets[ind] = annot.get('onset', 0)
            #description_ = annot.get('description', '')

            spike_packet = description_.split(" ")
            if len(spike_packet) == 3:
                spike, ch, duration = spike_packet
                numeric_filter = filter(str.isdigit, duration)
                durations[ind] = float("".join(numeric_filter))
                description = f"{spike} {_fix_channel_name(ch)}"
                descriptions[ind] = description
        #else:
        #    descriptions[ind] = description_
    onsets, durations, descriptions = _filter_empty(onsets, durations, descriptions)
    # Add correct annotations to the base raw object
    annotations_ = mne.Annotations(onsets, durations, descriptions)
    return annotations_, num_comments, onsets, descriptions, durations


def _fix_channel_name(ch_name):
    base_ch, ref_ch = ch_name.split("-")
    standard_chs = get_standard_1020_montage()
    for ch in standard_chs:
        if base_ch.upper() == ch.upper():
            base_ch = ch
    return f"{base_ch}-{ref_ch}"


def _filter_empty(onsets_, durations_, descriptions_):
    onsets = []
    durations = []
    descriptions = []
    for ind, (onset, duration, description) in enumerate(zip(onsets_, durations_, descriptions_)):
        if onset == 0 and duration == 0 and description == '':
            continue
        onsets.append(onset)
        durations.append(duration)
        descriptions.append(description)
    return np.array(onsets), np.array(durations), descriptions


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