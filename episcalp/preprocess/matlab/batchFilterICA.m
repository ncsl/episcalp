function batchFilterICA(pt_data, outdir, componentdir, EEGLabPath, OS, brain_thresh, save_components, window_ica, window_length, concatenate_windows)
    % pt_data is a struct with cell array fields record, hdr, and
    % source_file, which are the outputs of the filtered data
    records = pt_data.record';
    hdrs = pt_data.hdr;
    fnames = pt_data.source_files;
    num_patients = size(records);
    
    % make output directory
    mkdir(outdir);
    mkdir(componentdir);
    
    for pat = 1:num_patients
        % extract the record info
        record = records{pat};
        hdr = hdrs{pat};
        fpath = fnames{pat};
        [~,fname,~] = fileparts(fpath);
        [pat_name, session, task, run] = get_bids_params(fname);
        outname = create_out_name(pat_name, session, task, run);
        dirname = fullfile(outdir, sprintf('sub-%s',pat_name)); 
        mkdir(dirname);
        compdirname = fullfile(componentdir, pat_name);
        mkdir(compdirname);
        EEG = eeg_emptyset;
        labels = hdr.label';
        disp(labels)
        srate = hdr.frequency(1);
        % import the data into EEGlab
        EEG = pop_importdata('dataformat','array','nbchan',0,'data',record,'setname',pat_name,'srate',srate,'pnts',0,'xmin',0);
        % Subset the channels to the standard 1020 channels
        cedName = fullfile(outdir, strcat(pat_name, '.ced'));
     
        EEG = handleChannels(EEG, labels, cedName, EEGLabPath, OS);
        % perform the actual ICA. Make sure you have the binICA files set
        % or else this will be very slow
        if (window_ica)
            EEG_win = EEG;
            EEG_win.data = [];
            EEG_win.times = [];
            EEG_win.pnts = 0;
            [starts, stops] = get_sample_points(EEG, window_length);
            for win = 1:length(starts)
                window = [starts(win), stops(win)];
                EEG_ = EEG;
                EEG_.data = EEG.data(:, window(1):window(2));
                EEG_.times = EEG.times(window(1):window(2));
                EEG_.pnts = window(2)-window(1)+1;
                fname = sprintf("%s_win-%d_components.mat", outname, win);
                EEG_ = filterICA(EEG_, true, brain_thresh, save_components, compdirname, fname);
                % extract the necessary information
                data_ = EEG_.data;
                times_ = EEG_.times;
                pnts_ = EEG.pnts;
                if (~save_components)
                    EEG_.data = [];
                end
                set_fname = sprintf("%s_win-%d.set", outname, win);
                EEG_ = pop_saveset( EEG_, 'filename', char(set_fname),'filepath',char(compdirname));
                EEG_win.data = [EEG_win.data, data_];
                EEG_win.times = [EEG_win.times, times_];
                EEG_win.pnts = length(EEG_win.times);
            end
            if (concatenate_windows)
                set_fname = outname+".set";
                EEG_win = pop_saveset( EEG_win, 'filename', char(set_fname),'filepath',char(dirname));
            end
                       
        else
            fname = sprintf("%s_components.mat", outname);
            EEG = filterICA(EEG, true, brain_thresh, save_components, dirname, fname);
            % extract the necessary information
            pt_data = struct();
            pt_data.data_filt = EEG.data;
            pt_data.fs = EEG.srate;
            chanlocs = EEG.chanlocs;
            pt_data.labels = {chanlocs(:).labels}';
            % save both a .mat file and an EEGlab file
            results_fname = fullfile(dirname, pat_name+".mat");
            save(results_fname, 'pt_data');
            set_fname = outname+".set";
            EEG = pop_saveset( EEG, 'filename', char(set_fname),'filepath',char(dirname));
        end
        % clean up temp files used in binary ICA computation
        delete binica*
    end
end