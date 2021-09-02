function output = filterICA(EEG, bin, brain_thresh, save_components, plot_components, dirname, pat_name, save_windows)
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
    reject = struct();
    if (save_components)
        suffix = '_full';
        if (plot_components)
            save_ICA_plots(EEG, dirname, suffix, 'png');
        end
        reject = EEG.reject;

        data = EEG.data;
        if (~save_windows)
            EEG.data = [];
        end
        set_fname = pat_name+"_full.set";
        EEG = pop_saveset( EEG, 'filename', char(set_fname),'filepath',char(dirname));
        EEG.data = data;
    end
    
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
    
    %% Remove the components flagged for removal
    EEG = pop_subcomp(EEG, remove);
    if (save_components)
        % temporarily reset reject so that we can run ICLabel
        reject_ = EEG.reject;
        EEG.reject = structfun(@(x) [], reject, 'UniformOutput', false);
        suffix = '_filtered';
        if (plot_components)
            save_ICA_plots(EEG, dirname, suffix, 'png');
        end
        EEG.reject = reject_;
    end
    output = EEG;
    return;
end