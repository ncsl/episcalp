import collections

import mne
import numpy as np


def _parse_events_for_description(
    events, events_id, description: str, sfreq, orig_time
) -> mne.Annotations:
    """Parse events/eventsID from mne python for a specific description."""
    events = np.asarray(events)

    # look for the exact event fitting the description
    event_id = None
    selected_event = None
    for event, _id in events_id.items():
        if event == description:
            event_id = events_id[event]
            selected_event = event
            break
    new_events = events[np.where((events[:, -1] == event_id))]

    # if no events were found matching description return None
    if selected_event is None:
        return None

    # create new annotations data structure
    onsets = new_events[:, 0] / sfreq
    durations = np.zeros_like(onsets)  # assumes instantaneous events
    descriptions = [selected_event]
    annot_from_events = mne.Annotations(
        onset=onsets, duration=durations, description=descriptions, orig_time=orig_time
    )
    return annot_from_events


def _parse_sz_events(
    events, events_id, descriptions, sfreq, orig_time, match_case=False
):
    # This regex matches key-val pairs. Any characters are allowed in the key and
    # the value, except these special symbols: - _ . \ /
    # param_regex = re.compile(r'([^-_\.\\\/]+)-([^-_\.\\\/]+)')
    # if use_regex:
    # for match in re.finditer(param_regex, op.basename(fname)):
    #     key, value = match.groups()

    description_events = collections.defaultdict(list)
    # find all events containing the substring of the descriptions
    for description in descriptions:
        for event, _id in events_id.items():
            if not match_case:
                _event = event.lower()
                _description = description.lower()
            else:
                _event = event
                _description = description
            if _description in _event:
                description_events[description].append(event)

    # collect all inside an annotations data structure
    annotations = None
    for description, event_descriptions in description_events.items():
        for desc in event_descriptions:
            # get event matching description as an Annotation
            annot = _parse_events_for_description(
                events, events_id, desc, sfreq, orig_time
            )

            # build up annotations as a list
            if annotations is None:
                annotations = annot
            elif annot is not None:
                annotations += annot
    return annotations
