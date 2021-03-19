function output = filterICA(EEG, bin)
    %% Filter the data via ICA
    % Input:
    %   EEG: the eeg object used in EEGLab
    % Ouput:
    %   EEG: eeg object filtered
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Jacob Feitelberg
    % v1: Jan 2021
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
    %% Run ICA using the compiled 'binica' version. 
    % This is faster than runica. If binica is not configured correctly, 
    % use runica. The infomax ICA algorithm is random so do not expect to 
    % have exactly the same results when running this multiple times.
    if (nargin < 2)
        bin = true;
    end
    if (bin)
        try
            disp("trying binica")
            EEG = pop_runica(EEG,'icatype','binica', 'extended',1,'interupt','on');
        catch
            disp("using runica")
            EEG = pop_runica(EEG,'icatype','runica', 'extended',1,'interupt','on');
        end
    else
        EEG = pop_runica(EEG,'icatype','runica', 'extended',1,'interupt','on');
    end
    % EEG = pop_runica(EEG);
    
    %% Label the components using the ICLabel classifier
    EEG = iclabel(EEG);
    
    %% Filter out components
    classifications = EEG.etc.ic_classification.ICLabel.classifications;
    [m,n] = size(classifications);
    remove = [];
    for i = 1:m
        if classifications(i,1) < 0.05 || ... % brain
           classifications(i,2) > 0.9 || ... % muscle
           classifications (i,3) > 0.9 % eye
            remove = [remove; i]; % append this component to remove array
        end
    end
    
    %% Remove the components flagged for removal
    output = pop_subcomp(EEG, remove);
    return;
end