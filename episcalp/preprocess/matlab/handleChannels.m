function EEG = handleChannels(EEG, labels, cedName)
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
        label_clean = erase(erase(label,"Ref"), "EEG");
        labels_clean{lab, 1} = label_clean;
    end
    disp(labels_clean)
    writeCED(labels_clean, cedName);
    disp(cedName)
    cedNameChar = char(cedName);
    EEG = pop_chanedit(EEG, 'lookup', 'D:\\Downloads\\eeglab_default\\eeglab2020_0\\plugins\\dipfit\\standard_BESA\\standard-10-5-cap385.elp', ...
        'load', {cedNameChar,'filetype','chanedit'}, ...
        'lookup','D:\\Downloads\\eeglab_default\\eeglab2020_0\\plugins\\dipfit\\standard_BESA\\standard-10-5-cap385.elp');
    channel_list = getMontageChannels(labels_clean)';
    disp(channel_list)
    EEG = pop_select( EEG, 'channel', channel_list);
end