import collections
import os
from pathlib import Path

import json

import numpy as np
import pandas as pd
from mne_bids import get_entities_from_fname
from natsort import natsorted

from sample_code.read_datasheet import read_clinical_excel
from sample_code.utils import _standard_channels, subset_patients


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


def load_feature_data(feature_name, ext, deriv_path, excel_fpath, feature_names, json_fpath,  reference='monopolar',
                      include_groups=None, verbose=False):
    if include_groups is None:
        include_groups = {
            "non-epilepsy": 0,
            "epilepsy-normal": 1,
            "epilepsy-abnormal": 1,
        }
    groups_dict = {
        "0": "non-epilepsy",
        "1": "epilepsy-normal",
        "2": "epilepsy-abnormal",
    }

    modelname = feature_name
    kind = "eeg"

    datadir = Path(deriv_path) / modelname

    if ext == "npy":
        patient_result_dict = _load_patient_dict(datadir,  kind=kind, expname=modelname, pt_map_fpath=json_fpath,
                                                 include_subject_groups=list(include_groups.keys()), verbose=verbose)
    elif ext == "json":
        patient_result_dict = _load_patient_json(datadir, kind, expname=modelname, pt_map_fpath=json_fpath,
                                                 include_subject_groups=list(include_groups.keys()), verbose=verbose)
    else:
        patient_result_dict = None

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
        epilepsy_status = include_groups[groups_dict[str(group)]]
        for dataset in datasets:
            try:
                x = [dataset[feat].item() for feat in feature_names]
            except AttributeError as e:
                x = [dataset[feat] for feat in feature_names]
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


def _load_patient_dict(datadir, kind, expname, pt_map_fpath, include_subject_groups, verbose=False):
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

    patient_result_dict = subset_patients(patient_result_dict, include_subject_groups)

    return patient_result_dict


def _load_patient_json(datadir, kind, expname, pt_map_fpath, include_subject_groups, verbose=False):
    with open(pt_map_fpath) as f:
        pt_map = json.load(f)
    pt_names = []
    for cat, group in pt_map.items():
        pt_names.extend(group)

    patient_result_dict = collections.defaultdict(list)
    # get all files inside experiment
    trimmed_npz_fpaths = [x for x in datadir.rglob("*json")]

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
        fpaths = [x for x in subjdir.rglob(f"*sub-{subject}_*json")]
        data_dict = dict()
        # load in each subject's data
        for fpath in fpaths:
            # load in the data and append to the patient dictionary data struct
            with open(fpath) as fid:
                pt_json = json.load(fid)
            values = list(pt_json.values())
            max_spikes = max(values)
            # with np.load(fpath) as feat_value:
            data_dict["spike_rate"] = pt_json['total']
            data_dict['max_spikes'] = max_spikes
        patient_result_dict[subject].append(data_dict)

    if verbose:
        print("Got ", len(patient_result_dict), " patients")
        print(patient_result_dict.keys())

    patient_result_dict = subset_patients(patient_result_dict, include_subject_groups)

    return patient_result_dict

