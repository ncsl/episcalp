# -*- coding: utf-8 -*-

from enum import Enum
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from mne.utils import logger


class ClinicalColumnns(Enum):
    """Clinical excel sheet columns to be used."""

    CURRENT_AGE = "SURGERY_AGE"
    ONSET_AGE = "ONSET_AGE"
    GENDER = "GENDER"
    HANDEDNESS = "HAND"
    SUBJECT_ID = "PATIENT_ID"
    HOSPITAL_ID = "HOSPITAL_ID"
    ETHNICITY = "ETHNICITY"
    CENTER = "CLINICAL_CENTER"
    GROUP = "GROUP"
    EPILEPSY_TYPE = "EPILEPSY_TYPE"


def _filter_column_name(name):
    """Hardcoded filtering of column names."""
    # strip parentheses
    name = name.split("(")[0]
    name = name.split(")")[0]

    # strip whitespace
    name = name.strip()

    return name


def _format_col_headers(df):
    """Hardcoded format of column headers."""
    df = df.apply(lambda x: x.astype(str).str.upper())
    # map all empty to nans
    df = df.fillna(np.nan)
    df = df.replace("NAN", "", regex=True)
    df = df.replace("", "n/a", regex=True)
    return df


def read_clinical_excel(
        excel_fpath: Union[str, Path],
        subject: str = None,
        keep_as_df: bool = False,
        verbose: bool = False,
):
    """Read clinical datasheet Excel file.
    Turns the entire datasheet into upper-case, and removes any spaces, and parentheses
    in the column names.
    Subjects:
        Assumes that there are rows stratified by subject ID, and columns can be arbitrarily
        added.
    Channels:
        Assumes that there are some columns that are specifically named in the Excel file,
        which we can use regex to expand into a full list of channel names as strings.
    Parameters
    ----------
    excel_fpath : str | pathlib.Path
        The file path for the Excel datasheet.
    subject : str | optional
        The subject that can be used to filter the Excel DataFrame into just a single row
        of that specific subject.
    keep_as_df : bool
        Whether or not to keep the loaded Excel sheet as a DataFrame, or a structured
        dictionary.
    verbose : bool
    Returns
    -------
    df : Dict | pd.DataFrame
    """
    # load in excel file
    df = pd.read_excel(excel_fpath, engine="openpyxl")


    # expand contact named columns
    # lower-case column names
    df.rename(str.upper, axis="columns", inplace=True)
    # filter column names
    column_names = df.columns
    column_mapper = {name: _filter_column_name(name) for name in column_names}
    # remove any markers (extra on clinical excel sheet)
    df.rename(columns=column_mapper, errors="raise", inplace=True)

    # format column headers
    df = _format_col_headers(df)

    # remove dataframes that are still Unnamed
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    if verbose:
        print("Dataframe that was created looks like: ")
        try:
            print(df.to_markdown())
        except Exception as e:
            print(df.head(2))
            # print(e)

    subject_dfs = df[ClinicalColumnns.SUBJECT_ID.value].tolist()
    subs = [str(s).zfill(3) for s in subject_dfs]
    df[ClinicalColumnns.SUBJECT_ID.value] = subs


    # if specific subject, then read in that row
    if subject is not None:
        subject = subject.upper()
        if subject not in df[ClinicalColumnns.SUBJECT_ID.value].tolist():
            logger.error(f"Subject {subject} not in Clinical data sheet.")
            return None

        if keep_as_df:
            return df.loc[df[ClinicalColumnns.SUBJECT_ID.value] == subject]
        else:
            return df.loc[df[ClinicalColumnns.SUBJECT_ID.value] == subject].to_dict(
                "records"
            )[0]
    if keep_as_df:
        return df
    else:
        return df.to_dict("dict")