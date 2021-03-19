function mat_fname = batchFilter(folder, dataset, files, oind)
    % Extension will likely be ".edf" or ".bdf"
    % dataset should be a unique string, which can identify the output mat
    % file
    % include_channels is optional, but is a list of indices of channels to
    % read in (i.e. [1:12, 22:25])
    % output file contains one structure (pt_data) that contains three cell
    % arrays, one that holds the records, one that holds the headers (hdr),
    % and one that holds the original filename for back referencing.
    pt_data = struct();
    pt_data.record = {};
    pt_data.hdr = {};
    pt_data.source_files = {};
    for find = 1:length(files)
        fname = files{1, find};
        disp(fname)
        [hdr, record] = edfread(fname); % This might change depending on the recording montage. 32 here was for an extended standard montage
        fs = hdr.frequency(1);
        keep_inds = ones(size(record,1),1);
        for r = 1:size(record,1)
            row = record(r, :);
            if(~isfinite(row))
                keep_inds(r) = 0;
            end
        end
        record = record(keep_inds==1, :);
        labels = string(hdr.label);
        labels = labels(keep_inds==1)';
        labels = clean_label_names(labels);
        standard_montage = make_standard1020_montage();
        [record, labels] = reduce_montage(record, labels, standard_montage);
        record = trim_record(record, fs, 15*60);
        record = filtSignals(record, fs, 0.5, 90, 2);
        hdr.label = labels;
        % Name the save file as same as source file with .mat extension
        pt_data.source_files{find} = fname;
        pt_data.record{find} = record;
        pt_data.hdr{find} = hdr;
    end
    mat_fname = folder + "\" + dataset + sprintf("%d.mat", oind);
    save(mat_fname, 'pt_data')
end