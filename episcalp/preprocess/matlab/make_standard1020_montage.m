function standard1020_montage = make_standard1020_montage(extend_montage)
    %% Get the names of the standard 1020 montage channels
    % Ouput:
    %   standard1020_montage: List of channel names
    % Uses the extended_1020 system described here: https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Patrick Myers
    % v1: Feb 2021
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%
    if nargin < 1
        extend_montage = true;
    end
    standard1020_montage = {'Fp1', 'Fp2', 'F3','F4', 'F7', 'F8', 'Fz',... 
        'T3', 'T4', 'C3', 'C4', 'Cz', 'T5', 'T6', 'P3', 'P4', 'Pz', 'O1',...
        'O2', 'T7', 'T8', 'P7', 'P8'};
    if extend_montage
        additional_channels = {'Fpz', 'AF3', 'AF4', 'AF7', 'AF8', 'AFz',...
            'F1', 'F2', 'F5', 'F6', 'F9', 'F10', 'FC1', 'FC2', 'FC3',...
            'FC4', 'FC5', 'FC6', 'FT7', 'FT8', 'FT9', 'FT10', 'FCz',...
            'C1', 'C2', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5',...
            'CP6', 'TP7', 'TP8', 'TP9', 'TP10', 'CPz', 'P1', 'P2', 'P5',...
            'P6', 'P9', 'P10', 'PO3', 'PO4', 'PO7', 'PO8', 'POz', 'Oz'};
        standard1020_montage = horzcat(standard1020_montage,additional_channels);
    end
    
end