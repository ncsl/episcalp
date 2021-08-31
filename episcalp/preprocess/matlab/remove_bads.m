function [reduced_data, reduced_labels] = remove_bads(data, labels, subject_id, meta)
    subject_ids = meta.patient_ids;
    bads = meta.bads;
    bad_channels = '';
    for i = 1:length(subject_ids)
        if subject_ids(i) == double(subject_id)
            bad_channels = char(bads(i));
        end
    end
    if ~isempty(bad_channels)
        bad_list = split(bad_channels, ", ");
        keep_indices = ones(length(labels), 1);
        for ind = 1:length(labels)
            if ismember(labels(ind), bad_list)
                keep_indices(ind) = 0;
            end
        end
        keep_indices = logical(keep_indices);
        reduced_data = data(keep_indices, :);
        reduced_labels = labels(keep_indices);
    else
        reduced_data = data;
        reduced_labels = labels;
    end

end