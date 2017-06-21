%
% Test program for Kalman notch filter
%
% Dependencies: The baseline wander toolbox of the Open Source ECG Toolbox
%
% Open Source ECG Toolbox, version 1.0, October 2007
% Released under the GNU General Public License
% Copyright (C) 2007  Reza Sameni
% Sharif University of Technology, Tehran, Iran -- GIPSA-LAB, INPG, Grenoble, France
% reza.sameni@gmail.com

% This program is free software; you can redistribute it and/or modify it
% under the terms of the GNU General Public License as published by the
% Free Software Foundation; either version 2 of the License, or (at your
% option) any later version.

clc
clear all
close all;

% load('SampleECG1.mat'); data = data';
load('SampleECG2.mat'); data = data(1:15000,6)';

fs = 1000;
f0 = 50;

n = (0:length(data)-1);

x = data + (.055+.02*sin(2*pi*n*.1/fs)).*sin(2*pi*n*f0/fs)/std(data);

% [y1,y2,Pbar,Phat,PSmoothed,Kgain] = KFNotch(x,f0,fs);
[y1,y2,Pbar,Phat,PSmoothed,Kgain] = KFNotch(x,f0,fs,1e-3,.1*var(x),.9);


t = n/fs;

figure;
hold on;
plot(t,data,'b');
plot(t,x,'r');
plot(t,y1,'m');
plot(t,y2,'g');
grid;
xlabel('time (sec.)');
legend('original ECG','noisy ECG','Kalman filter','Kalman smoother');

figure;
psd(x,1000,fs);
title('noisy spectrum');

figure;
psd(y1,1000,fs);
title('Kalman filter output spectrum');

figure;
psd(y2,1000,fs);
title('Kalman smoother output spectrum');
