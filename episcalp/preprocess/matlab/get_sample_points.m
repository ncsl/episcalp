function [starts, stops] = get_sample_points(EEG, window_length)
    data = EEG.data;
    
    start = 1;
    data_len = size(data, 2);
    winsize = window_length*EEG.srate;
    
    nwins = ceil(data_len / winsize);
    
    starts = zeros(nwins,1);
    stops = zeros(nwins,1);
    win = 1;
    while(start < data_len)
        stop = min(start + winsize, data_len);
        starts(win) = start;
        stops(win) = stop;
        start = start + winsize;
        win = win + 1;
    end
end

