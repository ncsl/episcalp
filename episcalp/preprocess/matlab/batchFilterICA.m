function batchFilterICA(pt_data, outdir)
%% Automate EEGLab's ICA process for multiple files at once.
%% Inputs:
%		pt_data: struct containing cell arrays named record, hdr, and source files. This is the output of batchFilter.m
% 		outdir: string path of the desired output directory

%  Patrick Myers
%  Created: 02/02/2021

% Read in the variables from the struct
records = pt_data.record';
hdrs = pt_data.hdr;
fnames = pt_data.source_files;

% Iterate over patients
num_patients = size(records);
for pat = 1:num_patients
	record = records{pat};
	hdr = hdrs{pat};
	fname = fnames{pat};
	labels = hdr.label';
	srate = hdr.frequency(1);
	[~,pat_name,~] = fileparts(fname);
	
	% Initialize the EEGlab structure 
	EEG = eeg_emptyset;
	% Read in the data, specifying the sampling frequency
	EEG = pop_importdata('dataformat','array','nbchan',0,'data',record,'setname',pat_name,'srate',srate,'pnts',0,'xmin',0);
	% Add channel names and locations
	cedName = fullfile(outdir, pat_name+".ced");
	EEG = handleChannels(EEG, labels, cedName);
	% Perform ICA and remove components
	EEG = filterICA(EEG);
	% Save the mat and set files.
	pt_data = struct();
	pt_data.data_filt = EEG.data;
	pt_data.fs = EEG.srate;
	chanlocs = EEG.chanlocs;
	pt_data.labels = {chanlocs(:).labels}';
	results_fname = fullfile(outdir, pat_name+".mat");
	save(results_fname, 'pt_data');
	set_fname = pat_name+".set";
	EEG = pop_saveset( EEG, 'filename', char(set_fname),'filepath',char(outdir));
end
end