function labels_clean = clean_label_names(labels)
    %% Fixes label capitalization inconsitencies
    % Input:
    %   labels: Total channel label set for data
    % Ouput:
    %   labels_clean: Labels with correct capitalization
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Patrick Myers
    % v1: Feb 2021
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    labels_uppercase = ["FP1", "FP2", "CZ", "PZ", "FZ"];
    labels_lowercase = ["Fp1", "Fp2", "Cz", "Pz", "Fz"];
    labels_clean = labels;
    for i = 1:length(labels)
        label_strip = erase(erase(erase(labels(i),"EEG"), "Ref"), "REF");
        ind = find(strcmp(labels_uppercase,label_strip), 1);
        if ~isempty(ind)
            label_case = labels_lowercase(ind);
        else
            label_case = label_strip;
        end
        labels_clean(i) = label_case;
    end

end