from pathlib import Path
import pandas as pd
import json

from utils import update_participants_info

from mne_bids import get_entity_vals


def main():
    bids_root = Path(
        '/Users/adam2392/Johns Hopkins/Scalp EEG JHH - Documents/90Hz-10/')
    # bids_root = Path(
    #     '/Users/adam2392/Johns Hopkins/Patrick E Myers - scalp_jhu_bids/')
    source_dir = Path(
        '/Users/adam2392/Johns Hopkins/Patrick E Myers - scalp_jhu_bids/') / 'sourcedata'

    record_id_map_fname = source_dir / 'jhu_pt_map.json'
    excel_metadata_fpath = source_dir / 'JHU-metadata_June2021.xlsx'

    # read in record ID mapping
    with open(record_id_map_fname, 'r') as fin:
        record_map = json.load(fin)

    # read in metadata as dataframe
    meta_df = pd.read_excel(excel_metadata_fpath)

    columns = {
        'First EEG (normal vs abnormal)': 'Whether the first EEG showed normal or abnormal activity',
        'epileptiform EEG': 'Whether or not there was epileptoform activity in EEG',
        'number of AEDs during first eeg': 'Number of Anti-epileptic drugs during first EEG',
        'Final diagnosis': 'What was the final clinical diagnosis',
        'type of epilepsy': 'Focal or generalized epilepsy',
        'type of focal epilepsy': 'Clinical complexity (four categories)',
    }

    subjects = get_entity_vals(bids_root, 'subject')
    print('All the subjects are: ', subjects)

    for idx, row in meta_df.iterrows():
        record_id = row['Record ID']
        subject = record_map[str(record_id)]

        if subject not in subjects:
            continue

        for key, description in columns.items():
            value = row[key]
            if pd.isnull(value):
                value = 'n/a'
            update_participants_info(root=bids_root, subject=subject,
                                     key=key, value=value, description=description)


if __name__ == '__main__':
    main()
