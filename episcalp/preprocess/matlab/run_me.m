%% Variables to Set Up

EEGLabPath = 'D:/Desktop/eeglab_default/eeglab2020_0'; % change this
addpath(genpath(EEGLabPath))

folder = "D:/OneDriveParent/OneDrive - Johns Hopkins/ICATest/sourcedata";  % change this
%folder = '/home/adam2392/hdd3/tuh_epileptic_abnormal_vs_normal_EEG/sourcedata/abnormal/';
dataset = 'tuh_epilepsy_vs_normal';  % used to save the temporary files, not important
extension = '.edf';
outputdir = "D:/OneDriveParent/OneDrive - Johns Hopkins/ICATest/derivatives/ICA";  % change this
%outputdir = '/home/adam2392/hdd3/tuh_epilepsy_vs_normal/derivatives/ICA/sourcedata/epilepsy/';
%outputdir = '/home/adam2392/hdd3/tuh_epileptic_abnormal_vs_normal_EEG/derivatives/ICA/sourcedata/abnormal/';

group_size=5; % chunk the data to make computation a little faster
% path_separator = '/';  % change for Windows/Linux systems \ or /
os = 'Windows';  % eeglab has additional constraints for os than just path. We will just capture them here

filt_range = [1, 30];  % Frequencies to band-pass filter the data
filt_order = 4;        % Order of the band-pass filter

brain_thresh = 0.30;   % High threshold of brain perc to mark component for removal

save_components = true;  % If true, save the components before/after removal

window_ica = true;       % If true, splits the eeg data into windows, computes ICA for the window
window_length = 30;        % Only used if window_ica is true. Window length in seconds.


%%

% Grab just the names of just the edf files
files = dir(folder);
ext_files = {};
for find = 1:length(files)
    fname = files(find).name;
    % check if we have already run the file
    expected_name = strrep(fname, extension, '.set');
    expected_path = fullfile(outputdir, expected_name);
    if ~isfile(expected_path)
        %fname = folder + path_separator + fname;
        fname = fullfile(folder, fname);
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
    temp_mat_fpath = batchFilter(folder, dataset, files, count, filt_range, filt_order);
    % load in the structure
    load(temp_mat_fpath);
    % Perform ICA on that whole chunk
    batchFilterICA(pt_data, outputdir, EEGLabPath, os, brain_thresh, save_components, window_ica, window_length);
    count = count+1;
    % delete the temp file
    eval(sprintf("delete %s", temp_mat_fpath))
end
