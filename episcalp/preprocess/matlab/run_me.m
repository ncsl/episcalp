%% Variables to Set Up

EEGLabPath = 'D:/Desktop/eeglab_default/eeglab2020_0'; % change this
addpath(genpath(EEGLabPath))

root_dir = "D:\OneDriveParent\OneDrive - Johns Hopkins\Shared Documents\bids"; % change this
%folder = '/home/adam2392/hdd3/tuh_epileptic_abnormal_vs_normal_EEG/sourcedata/abnormal/';
dataset = 'tuh_epilepsy_vs_normal';  % used to save the temporary files, not important
extension = '.edf';
outputdir = "D:\OneDriveParent\OneDrive - Johns Hopkins\Shared Documents\bids\derivatives\ICA\sourcedata";  % change this
%outputdir = '/home/adam2392/hdd3/tuh_epilepsy_vs_normal/derivatives/ICA/sourcedata/epilepsy/';
%outputdir = '/home/adam2392/hdd3/tuh_epileptic_abnormal_vs_normal_EEG/derivatives/ICA/sourcedata/abnormal/';

group_size=5; % chunk the data to make computation a little faster
% path_separator = '/';  % change for Windows/Linux systems \ or /
os = 'Windows';  % eeglab has additional constraints for os than just path. We will just capture them here

filt_range = [1, 30];  % Frequencies to band-pass filter the data
filt_order = 4;        % Order of the band-pass filter

brain_thresh = 0.30;   % High threshold of brain perc to mark component for removal

save_components = true;  % If true, save the components before/after removal
plot_components = false;

window_ica = true;       % If true, splits the eeg data into windows, computes ICA for the window
window_length = 10;        % Only used if window_ica is true. Window length in seconds.
save_windows = false;
concatenate_windows = true;

bids = true;            % If true, expects data to be in BIDS format rather than just in one sourcedata folder

%%

% Grab just the names of just the edf files
if (bids)
    search_str = fullfile(root_dir, "**/*.edf");
else
    search_str = fullfile(root_dir, '*.edf');
end
files = dir(search_str);
% files = files([5, 35, 64], :);
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
    disp(files)
    % Perform freq-based filtering on the raw edf files
    temp_mat_fpath = batchFilter(root_dir, dataset, files, count, filt_range, filt_order);
    % load in the structure
    load(temp_mat_fpath);
    % Perform ICA on that whole chunk
    batchFilterICA(pt_data, outputdir, EEGLabPath, os, brain_thresh, save_components, plot_components, window_ica, window_length, save_windows, concatenate_windows);
    count = count+1;
    % delete the temp file
    eval(sprintf("delete %s", temp_mat_fpath))
end
