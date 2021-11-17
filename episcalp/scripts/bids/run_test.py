from pathlib import Path

from mne_bids import BIDSPath, read_raw_bids


def write_edf(fname, raw):
    """Export raw to EDF/BDF file (requires pyEDFlib)."""
    import pyedflib

    suffixes = Path(fname).suffixes
    ext = "".join(suffixes[-1:])
    if ext == ".edf":
        filetype = pyedflib.FILETYPE_EDFPLUS
        dmin, dmax = -32768, 32767
    elif ext == ".bdf":
        filetype = pyedflib.FILETYPE_BDFPLUS
        dmin, dmax = -8388608, 8388607

    data = raw.get_data()
    fs = raw.info["sfreq"]
    nchan = raw.info["nchan"]
    ch_names = raw.info["ch_names"]
    if raw.info["meas_date"] is not None:
        meas_date = raw.info["meas_date"]
    else:
        meas_date = None
    hp, lp = raw.info["highpass"], raw.info["lowpass"]
    hp = "DC" if hp == 0 else f"{hp:.0f} Hz"
    lp = f"{lp:.0f} Hz"
    f = pyedflib.EdfWriter(fname, nchan, filetype)
    channel_info = []
    data_list = []
    for i, kind in enumerate(raw.get_channel_types()):
        if kind == "eeg":
            data[i] *= 1e6  # convert to microvolts
            dimension = "uV"
            prefilter = f"HP: {hp}; LP: {lp}"
            pmin, pmax = data[i].min(), data[i].max()
            transducer = "Electrode"
        elif kind == "stim":
            dimension = "Boolean"
            prefilter = "No filtering"
            pmin, pmax = dmin, dmax
            transducer = "Triggers and status"
        else:
            raise NotImplementedError(f"Channel type {kind} not supported (currently only "
                                      f"EEG and STIM channels work)")
        channel_info.append(dict(label=ch_names[i],
                                 dimension=dimension,
                                 sample_rate=fs,
                                 physical_min=pmin,
                                 physical_max=pmax,
                                 digital_min=dmin,
                                 digital_max=dmax,
                                 transducer=transducer,
                                 prefilter=prefilter))
        data_list.append(data[i])
    f.setTechnician("MNELAB")
    f.setSignalHeaders(channel_info)
    if raw.info["meas_date"] is not None:
        f.setStartdatetime(meas_date)
    # note that currently, only blocks of whole seconds can be written
    f.writeSamples(data_list)
    for annot in raw.annotations:
        f.writeAnnotation(annot["onset"], annot["duration"], annot["description"])


if __name__ == '__main__':
    fname = '/Users/adam2392/Downloads/test.edf'

    root = '/Users/adam2392/Johns Hopkins/Scalp EEG JHH - Documents/bids/'
    bids_path = BIDSPath(
        root=root,
        subject='jhh001',
        datatype='eeg',
        suffix='eeg',
        run='01',
        extension='.edf'
    )
    raw = read_raw_bids(bids_path)
    raw.pick_types(eeg=True)
    write_edf(fname, raw)