function channel_list = getMontageChannels(labels)
%% Return a cell array of channels that are within the standard montage_channels
%% Inputs:
%		labels: cell array of actual channels to compare against
%% Outputs:
%		channel_list: cell array of standard montage_channels.

% Define possible standard montage channel names
montage_channels = {'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', ...
	'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', ...
	'T7', 'T8', 'P7', 'P8'};
channel_list = {};
% Iterate over the labels input and check if each channel is in the expected montage channels
for index = 1:length(labels)
	label = string(labels{index});
	if any(strcmp(montage_channels,label))
		channel_list{end+1} = char(label);
	end
end
end