function record = trim_record(record, fs, seconds)
    %% Reduce the time of the record
    % Input:
    %   record: EEG data
    %   fs: Sampling frequency of the record
    %   seconds: How many seconds to keep
    % Ouput:
    %   record: trimmed EEG data
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Patrick Myers
    % v1: Feb 2021
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%
    endsamp = fs*seconds;
    shape = size(record);
    if (shape(2) > endsamp)
        record = record(:, 1:endsamp);
    end
end