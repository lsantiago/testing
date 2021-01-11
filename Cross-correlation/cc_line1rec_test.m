%%%%% Code for cross-correlation test %%%%%%
% data used - recording from Matera - Line 1 geophones
% recording - 25_10_2019_10_51_00_000


close all
clear all
tic;

addpath '/Users/annamaria/PhD/Data/Matera/20200214_Line1_Matera'
file = load('/Users/annamaria/PhD/Data/Matera/20200214_Line1_Matera/151_25_10_2019_10_51_00_000.txt_line1');

%% loading the data

Fs=250; %samlpling frequency
dt=1/Fs;
nb_sen=20; %number of sensors
n=length(file)/nb_sen; %number of data points
time = linspace(0,n-1,n)*dt; % defining time vector

% separating recordings for each sensor
data_matrix = zeros(n,nb_sen);
for i = 1:nb_sen
    data_matrix(:,i) = file(n*(i-1)+1:n*i);  
end

%% plotting the example data

figure()
for i = 1:nb_sen
    plot(data_matrix(:,i)+0.0005*i);hold on
end


%% detrend and mean removal

for i = 1:nb_sen
    line = data_matrix(:,i);
    %line = detrend(line); 
    % removing mean
    line = line - mean(line);
    
    % removing trend
    trend_coeff = polyfit(time, line , 1);
    sensor_trend = trend_coeff(1)*time + trend_coeff(2);
    line = line - sensor_trend';
    
    % overwriting the data matrix
    data_matrix(:,i) = line;
end

%% cross-correlation

% for one signal
[r, lags] = xcorr(data_matrix(:,1), data_matrix(:,3));

[A, B] = max(r);

max_lag_cc = lags(B);
max_lag_cc_time = max_lag_cc * dt;


% plotting the cross correlation figure
figure()
plot(lags,r); hold on
plot([max_lag_cc, max_lag_cc], [A, A], 'ro')
title(sprintf('Max lag time is %f s', max_lag_cc_time))

% for all signals

for i = 1:nb_sen
    
    [r, lags] = xcorr(data_matrix(:,1), data_matrix(:,i));
    [A, B] = max(r);    % finding max value of cross-correlation and corresponding index
    max_lag_cc = lags(B);  % lag for max value
    max_lag_cc_time = max_lag_cc * dt;    % time of max value
    
    
    % putting everthing in the matrices
    cc_matrix(:,i) = r;
    lags_matrix(:,i) = lags;
    max_lag_cc_vector(i) = max_lag_cc;
    max_lag_cc_time_vector(i) = max_lag_cc_time;
    
end

figure()

for i=1:nb_sen
   plot(lags_matrix(:,i)*dt,  cc_matrix(:,i)+0.00001 * i); hold on
   plot(max_lag_cc_time_vector(i), max(cc_matrix(:,i))+0.00001 * i, 'ro')
end
xlim([-0.5, 0.5]) % define the x limits for zooming on the peak


%% deconvolution

wl = 1; % water level in 
r = 3; % resampling for better peak peaking
dstack = 0.5; % length of the singal for the visualisation of the deconvolution [in  sec]
Fs_r = Fs * r; % resampled sampling frequency   

%-------------------------------------------------------------------------

% for one signal

% signal deconvolution
sig_deco=deconvo(data_matrix(:,1),data_matrix(:,20),wl);
nt=length(sig_deco); % deconvolved signal length

% flipping the signal
sig_deco=[sig_deco(floor(nt/2)+1:end),sig_deco(1:floor(nt/2))];

% signal resampling -> resample(sig, p, q) --> sig * p/q
sig_deco=resample(sig_deco,r,1);
nt2=length(sig_deco);   % length of the resampled signal

% deconvolved and resampled signal - taken just the dstack*2 length
sig_deco_r=sig_deco((floor(nt2/2+1)-dstack*Fs_r):(floor(nt2/2+1)+dstack*Fs_r));


t=-dstack:1/Fs_r:dstack; % time axis for deconvolution plots

% max delay values
[A, B] = max(sig_deco_r);
max_lag_d = t(B);
max_lag_d_time = max_lag_d * dt;


% Plot of the deconvolved signal 
figure()
plot(t,sig_deco_r); hold on
plot([max_lag_d, max_lag_d], [A, A], 'ro')
title(sprintf('Max delay time is %f s', max_lag_d_time))



% for all recordings
sig_deco_matrix = zeros(Fs_r+1,nb_sen);
for i = 1:nb_sen
    
    sig_deco=deconvo(data_matrix(:,1),data_matrix(:,i),wl);
    nt=length(sig_deco); % deconvolved signal length
    
    % flipping the signal
    sig_deco=[sig_deco(floor(nt/2)+1:end),sig_deco(1:floor(nt/2))];
    
    % signal resampling -> resample(sig, p, q) --> sig * p/q
    sig_deco=resample(sig_deco,r,1);
    nt2=length(sig_deco);   % length of the resampled signal
    
    % deconvolved and resampled signal - taken just the dstack*2 length
    sig_deco_matrix(:,i)=sig_deco((floor(nt2/2+1)-dstack*Fs_r):(floor(nt2/2+1)+dstack*Fs_r));
end


% plot for all recordings

figure()
for i=1:nb_sen
   plot(t,  sig_deco_matrix(:,i)+0.1 * i); hold on  
end
