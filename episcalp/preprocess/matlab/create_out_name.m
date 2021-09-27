function out_name = create_out_name(subject, session, task, run)
    combine = [];
    if ~strcmp(subject, "")
        combine = [combine, sprintf("sub-%s", subject)];
    end
    if ~strcmp(session, "")
        combine = [combine, sprintf("ses-%s", session)];
    end
    if ~strcmp(task, "")
        combine = [combine, sprintf("task-%s", task)];
    end
    if ~strcmp(run, "")
        combine = [combine, sprintf("run-%s", run)];
    end
    out_name = strjoin(combine, "_");
end

