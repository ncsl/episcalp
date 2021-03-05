function EEG = handleChannels(EEG, labels, cedName)
%% Call EEGLab's functions for assigning channel locations and subsetting channels.
%  For readability of code, assigining channel locations is done in two steps. Writing 
%  a sparse .ced file (EEGLab format) and then loading that file in to read locations.
%  Otherwise, we are forced to append channels one-by-one in a lengthy command.
%% Inputs:
%		EEG: The EEGLab struct
%		labels: Cell array of channel names
%		cedName: Full path of a temporary file to save channel locations
%% Outputs:
%		EEG: The modified EEGlab struct with channels assigned

% Strip all channel names of "EEG" and "REF"
labels_clean = {};
for lab = 1:size(labels,1)
	label = labels(lab);
	label = erase(label,"EEG");
	label_clean = erase(label,"Ref");
	labels_clean{lab, 1} = label_clean;
end
% Temporarily write channel names to a ced file
writeCED(labels_clean, cedName);

% EEGLab commands require chars instead of strings
cedNameChar = char(cedName);

% EEGLab command to load in the saved .ced file for channel names and then query the database
% in the .elp file to determine channel locations
EEG = pop_chanedit(EEG, 'lookup', 'D:\\Downloads\\eeglab_default\\eeglab2020_0\\plugins\\dipfit\\standard_BESA\\standard-10-5-cap385.elp', ...
	'load', {cedNameChar,'filetype','chanedit'}, ...
	'lookup','D:\\Downloads\\eeglab_default\\eeglab2020_0\\plugins\\dipfit\\standard_BESA\\standard-10-5-cap385.elp');
% Get the subset of channels that belong to the montage.
channel_list = getMontageChannels(labels_clean)';
% Subset the data to be only the montage channels
% This must be done after assinging channel locations, else EEGLab errors
EEG = pop_select( EEG, 'channel', channel_list);
% Delete the temporary .ced file
delete(cedName)
end