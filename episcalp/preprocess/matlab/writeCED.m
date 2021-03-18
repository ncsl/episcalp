function writeCED(labels, fname)
    %% Write a CED file, which is the format EEGlab uses for channel locations
    % Input:
    %   labels: EEG channel names
    %   fname: Path to save the file
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Patrick Myers
    % v1: Feb 2021
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%
    fields = { 'Number' 'label' 'theta' 'radius' 'X' 'Y' 'Z' 'sph_theta' 'sph_phi' 'sph_radius' 'type' };
    fid = fopen(fname, 'w');
    if fid ==-1, error('Cannot open file'); end
    for field = 1:length(fields)
        fprintf(fid, '%s\t', fields{field});
    end
    fprintf(fid, '\n');
    emptyVal = "";
    for label = 1:length(labels)
        fprintf(fid, '%d\t',  label);
        fprintf(fid, '%s\t',   string(labels{label}));
        for ind = 3:length(fields)
            fprintf(fid, '%s\t',   emptyVal);
        end
        fprintf(fid, '\n');
    end
end