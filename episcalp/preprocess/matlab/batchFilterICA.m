function batchFilterICA(pt_data, outdir, EEGLabPath)
    % pt_data is a struct with cell array fields record, hdr, and
    % source_file, which are the outputs of the filtered data
    records = pt_data.record';
    hdrs = pt_data.hdr;
    fnames = pt_data.source_files;
    num_patients = size(records);
    
    % make output directory
    mkdir(outdir);
    
    for pat = 1:num_patients
        % extract the record info
        record = records{pat};
        hdr = hdrs{pat};
        fname = fnames{pat};
        [~,pat_name,~] = fileparts(fname);
        EEG = eeg_emptyset;
        labels = hdr.label';
        disp(labels)
        srate = hdr.frequency(1);
        % import the data into EEGlab
        EEG = pop_importdata('dataformat','array','nbchan',0,'data',record,'setname',pat_name,'srate',srate,'pnts',0,'xmin',0);
        % Subset the channels to the standard 1020 channels
        cedName = fullfile(outdir, pat_name+".ced");
        EEG = handleChannels(EEG, labels, cedName, EEGLabPath);
        % perform the actual ICA. Make sure you have the binICA files set
        % or else this will be very slow
        EEG = filterICA(EEG, true);
        % extract the necessary information
        pt_data = struct();
        pt_data.data_filt = EEG.data;
        pt_data.fs = EEG.srate;
        chanlocs = EEG.chanlocs;
        pt_data.labels = {chanlocs(:).labels}';
        % save both a .mat file and an EEGlab file
        results_fname = fullfile(outdir, pat_name+".mat");
        save(results_fname, 'pt_data');
        set_fname = pat_name+".set";
        EEG = pop_saveset( EEG, 'filename', char(set_fname),'filepath',char(outdir));
        % clean up temp files used in binary ICA computation
        delete binica*
    end
end