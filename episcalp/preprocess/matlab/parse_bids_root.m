function files = parse_bids_root(root_dir, pattern, files)
    fpath = fullfile(root_dir, pattern);
    f = dir(fpath);
    if (~isempty(f))
        if (~isempty(files))
            files_ = f;
        else
            files_ = [files; f];
        end
        files = files_;
    end
	for k = 1:length(f)
		fprintf('%s\n',fullfile(root_dir,f(k).name));
    end
 
	f = dir(root_dir);
	n = find([f.isdir]);	
	for k=n(:)'
		if any(f(k).name~='.') 
			parse_bids_root(fullfile(root_dir,f(k).name), pattern, files);
        end
    end
end

