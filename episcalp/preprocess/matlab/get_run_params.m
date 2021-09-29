function [EEGLabPath, root_dir, componentdir, extension, os, bids, filt_range, filt_order, brain_thresh, save_components, window_ica, window_length, concatenate_windows, outputdir] = get_run_params()
    % Path related params
    EEGLabPath = 'D:/Desktop/eeglab_default/eeglab2020_0';
    root_dir = "D:\OneDriveParent\OneDrive - Johns Hopkins\Shared Documents\bids";
    componentdir = "D:\OneDriveParent\OneDrive - Johns Hopkins\Shared Documents\bids\derivatives\ICA_components\sourcedata";
    extension = '.edf';
    os = 'Windows';
    bids = true;   % Whether the sourcedata is in bids format
    
    % Filtering params
    filt_range = [1, 30];  % Frequencies to band-pass filter the data
    filt_order = 4;        % Order of the band-pass filter
    
    % ICA params
    brain_thresh = 0.30;   % High threshold of brain perc to mark component for removal
    save_components = false;  % If true, save the components before/after removal
    window_ica = true;       % If true, splits the eeg data into windows, computes ICA for the window
    window_length = 20;        % Only used if window_ica is true. Window length in seconds.S
    concatenate_windows = true;    % If true, recombines the windowed ICA into a single file
    
    
    outputdir = sprintf("D:\\OneDriveParent\\OneDrive - Johns Hopkins\\Shared Documents\\bids\\derivatives\\ICA\\%d-%dHz-%0d\\win-%d\\sourcedata", filt_range(1), filt_range(2), 100*brain_thresh, window_length);

end

