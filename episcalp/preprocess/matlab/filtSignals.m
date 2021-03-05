function [data_filt] = filtSignals(data, fs, fcutlow, fcuthigh, forder)
%% Filter iEEG signals in desired frequency band
% Inputs:
    % signal: dimensions ch x time
    % fs: sampling frequency of signal
    % fcutlow: low cut frequency
    % fcuthigh: high cut frequency
    % forder: filter order
    
% Outputs:
    % signal_filt: the filtered signal (dimensions ch x time)
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Kristin M. Gunnarsdottir
% v1: 2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data_filt = data';

% Powerline rejection at 60 Hz
hpf1 = 59:60:fs/2;  % HalfPowerFrequency1
hpf2 = 61:60:fs/2;  % HalfPowerFrequency2

% Powerline rejection at 50 HZ if data recorded in Europe
% hpf1 = 49:50:fs/2;  % HalfPowerFrequency1
% hpf2 = 51:50:fs/2;  % HalfPowerFrequency2

for hpf = 1:length(hpf1)
    d1 = designfilt('bandstopiir','FilterOrder',2, ...
               'HalfPowerFrequency1',hpf1(hpf),'HalfPowerFrequency2',hpf2(hpf), ...
               'DesignMethod','butter','SampleRate',fs);
    data_filt = filtfilt(d1,data_filt);
end

% Bandpass filter
[b,a]=butter(forder,[fcutlow,fcuthigh]/(fs/2),'bandpass');
% fvtool(b,a)         % visualizing the filter

data_filt = filtfilt (b,a,data_filt);
data_filt = data_filt';
end
