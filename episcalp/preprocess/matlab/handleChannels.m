function EEG = handleChannels(EEG, labels, cedName, EEGLabPath, OS)
    %% Wrapper function to clean up channels in EEGlab
    %   EEG: EEG data structure in EEGlab
    %   labels: Total channel label set for the data
    %   cedName: Path to save the ced file
    % Ouput:
    %   EEG: modified version of EEG data structure
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Patrick Myers
    % v1: Feb 2021
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%
    disp(labels)
    labels_clean = {};
    for lab = 1:size(labels,2)
        label = labels(lab);
        label_clean = erase(erase(label,'Ref'), 'EEG');
        labels_clean{lab, 1} = label_clean;
    end
    disp(labels_clean)
    disp(cedName)
    writeCED(labels_clean, cedName);
    elp_path = fullfile(EEGLabPath, 'plugins', 'dipfit', 'standard_BESA', 'standard-10-5-cap385.elp');
    if (strcmp(OS, "Windows"))
        cedName = char(cedName);
        elp_path = char(elp_path);
    end
    EEG = pop_chanedit(EEG, 'lookup', elp_path, ...
        'load', {cedName,'filetype','chanedit'}, ...
        'lookup',elp_path);
    channel_list = getMontageChannels(labels_clean)';
    disp(channel_list)
    EEG = pop_select( EEG, 'channel', channel_list);
end