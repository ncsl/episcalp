function channel_list = getMontageChannels(labels)
    montage_channels = {'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', ...
        'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', ...
        'T7', 'T8', 'P7', 'P8'};
    channel_list = {};
    for index = 1:length(labels)
        label = string(labels{index});
        if any(strcmp(montage_channels,label))
            channel_list{end+1} = char(label);
        end
    end
end