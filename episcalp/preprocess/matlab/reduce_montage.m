function [reduced_data, reduced_labels] = reduce_montage(data, labels, ideal_labels)
    %% Reduce channels to standard montage
    % Input:
    %   data: EEG data
    %   labels: Total channel label set for data
    %   ideal_labels: Names of the standard 1020 channels
    % Ouput:
    %   reduced_data: subset version of the EEG data
    %   reduced_labels: labels that belong to standard 1020 montage
    %   (maintains original order)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Patrick Myers
    % v1: Feb 2021
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%
    keep_indices = zeros(length(labels), 1);
    for ind = 1:length(labels)
        if ismember(labels(ind), ideal_labels)
            keep_indices(ind) = 1;
        end
    end
    disp(size(keep_indices))
    disp(size(data))
    keep_indices = logical(keep_indices);
    reduced_data = data(keep_indices, :);
    reduced_labels = labels(keep_indices);
end