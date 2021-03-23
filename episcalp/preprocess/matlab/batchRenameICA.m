%%
% 1. Move all the .set and .fdt files to a new folder
% 2. Change the sourcedir path below to the new folder path
% The .set file needs to point to the .fdt file, so this will modify
% the internal pointer

%%
EEGLabPath = "D:/Downloads/eeglab_default/eeglab2020_0";
%EEGLabPath = '/home/adam2392/Documents/eeglab2021.0'; % change this
addpath(genpath(EEGLabPath))


sourcedir = 'D:/ScalpData/test_convert/derivatives/ICA/sourcedata/normal';
fileList = dir(fullfile(sourcedir, '**/*.set'));
for find = 1:length(fileList)
    fname = fileList(find).name;
    fdir = fileList(find).folder;
    disp(fname)
    EEG = pop_loadset('filename', fname, 'filepath', fdir);
    EEG = eeg_checkset(EEG);
    EEG = pop_saveset(EEG, 'filename', fname, 'filepath', fdir);
end