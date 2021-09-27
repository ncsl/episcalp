function meta = handle_metadata(meta_fpath)
    if ~isempty(meta_fpath)
        meta_table = readtable(meta_fpath);
        patient_ids = table2array(meta_table(1:86, 'hospital_id'));
        bads = table2array(meta_table(1:86, 'bad_contacts'));
        best_windows = table2array(meta_table(1:86, 'best_window'));
        meta = struct();
        meta.patient_ids = patient_ids;
        meta.bads = bads;
        meta.best_windows = best_windows;
    else
        meta = struct();
    end
end