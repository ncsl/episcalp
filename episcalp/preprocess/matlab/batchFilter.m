function output = batchFilter(folder, dataset, extension, include_channels)
% Extension will likely be ".edf" or ".bdf"
% dataset should be a unique string, which can identify the output mat
% file
% include_channels is optional, but is a list of indices of channels to
% read in (i.e. [1:12, 22:25])
% output file contains one structure (pt_data) that contains three cell
% arrays, one that holds the records, one that holds the headers (hdr),
% and one that holds the original filename for back referencing.
if nargin < 4
	include_channels = [1:32];
files = dir(folder);
pt_data = struct();
pt_data.record = {};
pt_data.hdr = {};
pt_data.source_files = {};
ind = 1;
for file = 1:length(files)
	fname = files(file).name;
	fname = folder + "\" + fname;
	disp(fname)
	if (contains(fname, extension))
		[hdr, record] = edfread(fname, 'targetSignals',include_channels); % This might change depending on the recording montage. 32 here was for an extended standard montage
		record = filtSignals(record, 200, 0.5, 90, 2);
		labels = string(hdr.label);
		% Name the save file as same as source file with .mat extension
		pt_data.source_files{ind} = fname;
		pt_data.record{ind} = record;
		pt_data.hdr{ind} = hdr;
		ind = ind + 1;
	end
end
mat_fname = folder + "\" + dataset + ".mat";
save(mat_fname, "pt_data",  '-v7.3')
end