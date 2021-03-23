function channel_list = getMontageChannels(labels)
    montage_channels = make_standard1020_montage();
    channel_list = {};
    for index = 1:length(labels)
        label = string(labels{index});
        if any(strcmp(montage_channels,label))
            channel_list{end+1} = char(label);
        end
    end
end