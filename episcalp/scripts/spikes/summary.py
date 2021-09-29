from mne import Annotations

from ...preprocess.montage import get_standard_1020_channels

def _annotlist_to_annots(annot_list):
    for idx in range(len(annot_list)):
        if idx == 0:
            annots = annot_list[idx]
        else:
            annots += annot_list[idx]
    return annots


def _process_annot(annot):
    # process annotations into a spike/spikegen event with
    # channel name annotated
    desc = annot.description[0]

    # Pattern is: <spike> <ch_name>-Av12
    spike_desc, ch_str = desc.split(' ')
    ch_name = ch_str.split('-')[0]
    onset = annot.onset
    duration = annot.duration
    return spike_desc, ch_name
    # return Annotations(onset=onset, duration=duration, description=spike_desc,
    #                    ch_names=[[ch_name]])


def _get_spike_annots(raw, verbose=True):
    onsets = []
    durations = []
    description = []
    ch_names = []

    standard_chs = get_standard_1020_channels()

    # loop through annotations
    for annot in raw.annotations:
        annot = Annotations(**annot)
        curr_desc = annot.description[0]
        if curr_desc.startswith('Spike ') or curr_desc.startswith('SpikeGen'):
            # or curr_desc.startswith('@Spike ') or curr_desc.startswith('@SpikeGen'):
            desc, ch_name = _process_annot(annot)

            # only standard_1020 montage is supported rn
            if ch_name not in standard_chs:
                continue
            # we want to remove midline contacts rn because
            # Bayview doesn't use them
            if ch_name in ['Cz', 'Pz', 'Fz']:
                continue

            onsets.extend(annot.onset)
            durations.extend(annot.duration)
            description.append(desc)
            ch_names.append([ch_name])

    # create Annotations object
    spike_annots = Annotations(onset=onsets, duration=durations, description=description,
                               ch_names=ch_names)
    return spike_annots


def process_spike_annots(raw, verbose=True):
    # get all the spike annotations
    spike_annots, spikegen_annots = _get_spike_annots(raw, verbose=verbose)
