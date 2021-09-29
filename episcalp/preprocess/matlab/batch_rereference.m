function new_data = batch_rereference(set_dir)
set_fpaths = dir(fullfile(set_dir, '*.set'));
nsubs = length(set_fpaths);
for f = 1:nsubs
    set_file = set_fpaths(f);
    set_fpath = fullfile(set_file.folder,set_file.name);
    disp(set_fpath)
end


end
