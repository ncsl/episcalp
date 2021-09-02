function batchFilterICA(pt_data, outdir, EEGLabPath, OS, brain_thresh, save_components, plot_components, window_ica, window_length, save_windows, concatenate_windows)
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
        dirname = fullfile(outdir, pat_name); 
        mkdir(dirname);
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
                dirname_ = fullfile(dirname, "win-"+win);
                mkdir(dirname_);
                EEG_ = filterICA(EEG_, true, brain_thresh, save_components, plot_components, dirname_, pat_name, save_windows);
                % extract the necessary information
                data_ = EEG_.data;
                times_ = EEG_.times;
                pnts_ = EEG.pnts;
                if (~save_components)
                    EEG_.data = [];
                end
                set_fname = pat_name+".set";
                EEG_ = pop_saveset( EEG_, 'filename', char(set_fname),'filepath',char(dirname_));
                EEG_win.data = [EEG_win.data, data_];
                EEG_win.times = [EEG_win.times, times_];
                EEG_win.pnts = length(EEG_win.times);
            end
            if (concatenate_windows)
                set_fname = pat_name+".set";
                EEG_win = pop_saveset( EEG_win, 'filename', char(set_fname),'filepath',char(dirname));
            end
                       
        else
            EEG = filterICA(EEG, true, brain_thresh, save_components, dirname, pat_name);
            % extract the necessary information
            pt_data = struct();
            pt_data.data_filt = EEG.data;
            pt_data.fs = EEG.srate;
            chanlocs = EEG.chanlocs;
            pt_data.labels = {chanlocs(:).labels}';
            % save both a .mat file and an EEGlab file
            results_fname = fullfile(dirname, pat_name+".mat");
            save(results_fname, 'pt_data');
            set_fname = pat_name+".set";
            EEG = pop_saveset( EEG, 'filename', char(set_fname),'filepath',char(dirname));
        end
        % clean up temp files used in binary ICA computation
        delete binica*
    end
end