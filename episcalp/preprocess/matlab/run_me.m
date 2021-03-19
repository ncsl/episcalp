EEGLabPath = "/home/adam2392/Documents/eeglab2021.0"; % change this
addpath(genpath(EEGLabPath))

%folder = "D:/ScalpData/test_parallel/sourcedata/normal";  % change this
folder = "/home/adam2392/hdd3/tuh_normal_vs_abnormal/sourcedata/normal";
dataset = "tuh_normal_vs_abnormal";  % used to save the temporary files, not important
extension = ".edf";
%outputdir = "D:/ScalpData/test_parallel/outdir";  % change this
outputdir = "/home/adam2392/hdd3/tuh_normal_vs_abnormal/derivatives/ICA/normal";

group_size=5; % chunk the data to make computation a little faster
path_separator = "/";  % change for Windows/Linux systems \ or /

% Grab just the names of just the edf files
files = dir(folder);
ext_files = {};
for find = 1:length(files)
    fname = files(find).name;
    fname = folder + path_separator + fname;
    if (contains(fname, extension))
        ext_files{end+1} = fname;
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
    temp_mat_fpath = batchFilter(folder, dataset, files, count);
    % load in the structure
    load(temp_mat_fpath);
    % Perform ICA on that whole chunk
    batchFilterICA(pt_data, outputdir, EEGLabPath)
    count = count+1;
    % delete the temp file
    eval(sprintf("delete %s", temp_mat_fpath))
end
