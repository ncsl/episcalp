sourcedir = 'D:\ScalpData\R01_ministudy\ICA_data';
fileList = dir(fullfile(sourcedir, '*.set'));
for find = 1:length(fileList)
    fname = fileList(find).name;
    disp(fname)
    EEG = pop_loadset('filename', fname, 'filepath', sourcedir);
    EEG = eeg_checkset(EEG);
    EEG = pop_saveset(EEG, 'filename', fname, 'filepath', sourcedir);
end