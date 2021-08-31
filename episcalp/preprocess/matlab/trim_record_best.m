function record = trim_record_best(record, fs,subject_id, meta)
    subject_ids = meta.patient_ids;
    best_windows = meta.best_windows;
    best_window = '';
    max_end_win = size(record,2);
    for i = 1:length(subject_ids)
        if subject_ids(i) == double(subject_id)
            best_window = char(best_windows(i));
        end
    end
    if strcmp("N/A", best_window)
        start_win=1;
        end_win=300*fs;
        max_win = size(record, 2);
        if end_win >= max_win
            end_win = max_win -1;
        end
        fprintf('Clipping %s from %d-%d', subject_id, start_win, end_win)
        record = record(:, start_win:end_win);
    elseif contains(best_window, ",")
        segment_splits = strsplit(best_window, ",");
        record_cat = [];
        for s = 1:length(segment_splits)
            time_split = strsplit(segment_splits{s}, "-");
            start_win = str2num(time_split{1}) * fs;
            if start_win == 0
                start_win = 1;
            end
            end_win = str2num(time_split{2}) * fs;
            record_cat = [record_cat, record(:, start_win:end_win)];
        end
        fprintf('Clipping %s discontinuously', subject_id)
        record = record_cat;
    else
        time_split = strsplit(best_window, "-");
        start_win = str2num(time_split{1}) * fs;
        if start_win == 0
            start_win = 1;
        end
        end_win = str2num(time_split{2}) * fs;
        if (end_win > max_end_win)
            end_win = max_end_win
        end
        fprintf('Clipping %s from %d-%d', subject_id, start_win, end_win)
        record = record(:, start_win:end_win);
    end
    
end