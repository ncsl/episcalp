function [output, classifications] = filterICA(EEG, bin, brain_thresh, save_components, dirname, fname)
    %% Filter the data via ICA
    % Input:
    %   EEG: the eeg object used in EEGLab
    %   bin: boolean of whether to try to run the binary ica
    %   brain_thresh: upper cutoff val of brain perc in ICLabel to remove
    %   save_components: boolean of whether to save the individual
    %   components
    %   dirname: Path of directory to save the components
    
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
        if classifications(i,1) < brain_thresh || ... % brain
           classifications(i,2) > 0.9 || ... % muscle
           classifications (i,3) > 0.9 % eye
            remove = [remove; i]; % append this component to remove array
        end
    end

    % Grab the relevant data from the ICA run and save
    if (save_components)
        ica_data = struct();
        ica_data.data = EEG.data;
        ica_data.labels = {EEG.chanlocs(:).labels};
        ica_data.components = classifications;
        ica_data.sfreq = EEG.srate;
        ica_data.times = EEG.times;
        ica_data.icaact = EEG.icaact;
        ica_data.icawinv = EEG.icawinv;
        ica_data.icasphere = EEG.icasphere;
        ica_data.icaweights = EEG.icaweights;
        ica_data.icachansind = EEG.icachansind;
        out_fpath = fullfile(dirname, fname);
        save(out_fpath, 'ica_data');
    end

    %% Remove the components flagged for removal
    EEG = pop_subcomp(EEG, remove);
    
    output = EEG;
    return;
end