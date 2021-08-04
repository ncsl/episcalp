import collections
import os
from pathlib import Path

import json

import numpy as np
import pandas as pd
from mne_bids import get_entities_from_fname
from natsort import natsorted

from sample_code.read_datasheet import read_clinical_excel
from sample_code.utils import _standard_channels


def read_participants_tsv(root, participant_id=None, column_keys=None):
    """Read in ``participants.tsv`` file.

    Allows one to query the columns of the file as
    a DataFrame and say obtain the list of Engel scores.
    """
    participants_tsv_fname = Path(root) / 'participants.tsv'

    participants_df = pd.read_csv(participants_tsv_fname, delimiter='\t')
    if participant_id is not None:
        participants_df = participants_df[participants_df['participant_id'] == participant_id]

    if column_keys is not None:
        if not isinstance(column_keys, list):
            column_keys = [column_keys]
        participants_df = participants_df[column_keys]
    return participants_df


def load_source_sink_data(deriv_path, excel_fpath, feature_names, json_fpath,  reference='monopolar', verbose=False):

    modelname = "sourcesink"
    kind = "eeg"

    datadir = Path(deriv_path) / modelname

    patient_result_dict = _load_patient_dict(datadir, kind=kind, expname='sourcesink', pt_map_fpath=json_fpath, verbose=verbose)

    unformatted_X = []
    y = []
    subjects = []
    epilepsy_types = []
    centers = []
    groups = []
    ch_names = []
    for subject, datasets in patient_result_dict.items():
        pat_dict = read_clinical_excel(excel_fpath, subject=subject)
        center = pat_dict['CLINICAL_CENTER']
        group = pat_dict['GROUP']
        epilepsy_status = int(group) > 0
        for dataset in datasets:
            x = [dataset[feat].item() for feat in feature_names]
            #ch_names_list = dataset["ch_names"]
            ch_names_list = _standard_channels()

            # TODO: Look at morf-demo for how to handle mutliple recordings for one patient

            unformatted_X.append(x)
            y.append(epilepsy_status)
            groups.append(group)
            centers.append(center)
            ch_names.append(ch_names_list)
            subjects.append(subject)

    return unformatted_X, y, groups, subjects, ch_names, centers


def _load_patient_dict(datadir, kind, expname, pt_map_fpath, verbose=False):
    with open(pt_map_fpath) as f:
        pt_map = json.load(f)
    pt_names = []
    for cat, group in pt_map.items():
        pt_names.extend(group)

    patient_result_dict = collections.defaultdict(list)
    # get all files inside experiment
    trimmed_npz_fpaths = [x for x in datadir.rglob("*npy")]

    # get a hashmap of all subjects
    subjects_map = {}
    for fpath in trimmed_npz_fpaths:
        params = get_entities_from_fname(
            os.path.basename(fpath), on_error='ignore'
        )
        subjects_map[params["subject"]] = 1

    if verbose:
        print(f'Got {len(subjects_map)} subjects')

        # loop through each subject
    subject_list = natsorted(subjects_map.keys())
    for subject in subject_list:
        reference = "monopolar"
        subjdir = Path(datadir)
        fpaths = [x for x in subjdir.rglob(f"*sub-{subject}_*npy")]
        features = [f.name.split("desc-")[1].split("_")[0] for f in fpaths]
        data_dict = dict()
        # load in each subject's data
        for fpath, feat in zip(fpaths, features):
            # load in the data and append to the patient dictionary data struct
            feat_value = np.load(fpath)
            #with np.load(fpath) as feat_value:
            data_dict[feat] = feat_value
        patient_result_dict[subject].append(data_dict)

    if verbose:
        print("Got ", len(patient_result_dict), " patients")
        print(patient_result_dict.keys())

    return patient_result_dict

