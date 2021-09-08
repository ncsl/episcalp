function [subject, session, task, run] = get_bids_params(bids_fname)
    subject = "";
    session = "";
    task = "";
    run = "";
    bids_fname_parse = strsplit(bids_fname, "_");
    for idx = 1:length(bids_fname_parse)
        bids_part = bids_fname_parse(idx);
        if contains(bids_part, 'sub')
            subject = strrep(bids_part, "sub-", "");
        elseif contains(bids_part, 'ses')
            session = strrep(bids_part, "ses-", "");
        elseif contains(bids_part, 'task')
            task = strrep(bids_part, "task-", "");
        elseif contains(bids_part, 'run')
            run = strrep(bids_part, "run-", "");
        end
    end
end

