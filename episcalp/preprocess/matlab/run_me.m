%% Set parameters from the get_run_params file, which should not be vc

[EEGLabPath, root_dir, componentdir, extension, os, bids, filt_range, filt_order, brain_thresh, save_components, window_ica, window_length, concatenate_windows, outputdir] = get_run_params();

%%

% Grab just the names of just the edf files
if (bids)
    search_str = fullfile(root_dir, "**/*.edf");
else
    search_str = fullfile(root_dir, '*.edf');
end
files = dir(search_str);
%files = files(2:6, :);
ext_files = {};
for find = 1:length(files)
    fname = files(find).name;
    fdir = files(find).folder;
    % check if we have already run the file
    expected_name = strrep(fname, extension, '.set');
    expected_path = fullfile(outputdir, expected_name);
    if ~isfile(expected_path)
        %fname = folder + path_separator + fname;
        fname = fullfile(fdir, fname);
%         disp(strcat("About to run this file: ", fname))
        %disp(expected_path);
        if (contains(fname, extension))
            ext_files{end+1} = fname;
        end
    end
end
count = 1;
% iterate over the chunks
for find=1:group_size:length(ext_files)
    if (find+group_size-1>length(ext_files))
        stop_ind = length(ext_files);
    else
        stop_ind = find+group_size-1;
    end
    files = {ext_files{find:stop_ind}};
    % Perform freq-based filtering on the raw edf files
    temp_mat_fpath = batchFilter(root_dir, dataset, files, count, filt_range, filt_order);
    % load in the structure
    load(temp_mat_fpath);
    % Perform ICA on that whole chunk
    batchFilterICA(pt_data, outputdir, componentdir, EEGLabPath, os, brain_thresh, save_components, window_ica, window_length, concatenate_windows);
    count = count+1;
    % delete the temp file
    eval(sprintf("delete %s", temp_mat_fpath))
end
