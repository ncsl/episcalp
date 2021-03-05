function writeCED(labels, fname)
%% Save a sparse .ced file containing just the channel names.
%  Paired down version of EEGLab's pop_chanedit save function
%% Inputs:
%		labels: cell array containing channel names
%		fname: Full path to the ced file


% Header for the .ced file
fields = { 'Number' 'label' 'theta' 'radius' 'X' 'Y' 'Z' 'sph_theta' 'sph_phi' 'sph_radius' 'type' };
% Create and open the .ced file
fid = fopen(fname, 'w');
if fid ==-1, error('Cannot open file'); end
% Add header row
for field = 1:length(fields)
	fprintf(fid, '%s\t', fields{field});
end
fprintf(fid, '\n');
emptyVal = "";
% For each channel, add its number and name, leaving an empty value for all other fields
for label = 1:length(labels)
	fprintf(fid, '%d\t',  label);
	fprintf(fid, '%s\t',   string(labels{label}));
	for ind = 3:length(fields)
		fprintf(fid, '%s\t',   emptyVal);
	end
	fprintf(fid, '\n');
end
end