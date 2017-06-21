function [R_loc,QRS_width,Q_loc,S_loc,lp_ecg] = QRSdetection(necg,Fs)
% QRSdetection.m
% QRS detection and Rhythm Analysis  
% QRS Beat Detection using Pan-TompKins Algorithm 
% associated files of the Bayesian P and T wave delineation and waveform
% estimation Toolbox
% Inputs:
%
% Outputs:
%     necg: signal ecg sans pic d'alimentation
%     lp_ecg: ECG signal after low-pass filtering
%     hp_ecg: ECG signal after high-pass filtering
%     deriv_ecg: ECG signal after derivation filtering
%     sq_ecg: ECG signal after quadratic filtering
%     moy_ecg: ECG signal after moving average filtering
%     R_loc : R peak location vector
%     QRS_width: QRS complexe length vector
%     Q_loc : Q peak location vector
%     S_loc : S peak location vector
%
% remarks : QRS_width_buff=S_loc-Q_loc
% We need the wavelet Toolbox to perform all the calculations
%
% Reference:
% [1] C. Lin, C. Mailhes and J.-Y. Tourneret, "P- and T-wave delineation in
%     ECG signals using a Bayesian approach and a partially collapsed Gibbs 
%     sampler", IEEE Trans. Biomed. Eng., vol. 57, no. 12, pp. 2840 - 2849, 
%     Dec. 2010.
% [2] C. Lin, G. Kail, J.-Y. Tourneret, C. Mailhes and F. Hlawatsch, "P and
%     T-wave Delineation and waveform estimation in ECG Signals Using a 
%     Block Gibbs Sampler," IEEE Int. Conf. on Acoust., Speech and Sig. Proc. 
%     (ICASSP'11), 2011, 
% 
% Written by Chao LIN, chao.lin@tesa.prd.fr and made available under the 
% GNU general public license. If you have not received a copy of this 
% license, please download a copy from http://www.gnu.org/
% Please distribute (and modify) freely, commenting
% where you have added modifications. 
% The author would appreciate correspondence regarding
% corrections, modifications, improvements etc.
% This program is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
% 
% Acknowledgment: 
% The author would like to acknowledge 
% G. Clifford for providing BaselineTOS.m
% R. Sameni for providing Baseline1.m and BPFilter.m
% (C) C. LIN 2011 
% TeSA lab, University of Toulouse, France
 %% Loading the ECG file  %%
Ns = length(necg);
time = [1 : Ns]/Fs;

% lowpass filtering 
b1 = [1 0 0 0 0 0 -2 0 0 0 0 0 1]/32;
a1 = [1 -2 1];

lpecg = filter(b1, a1, necg);
lp_ecg=lpecg;

%% Low pass
b2 = zeros(1,33);
b2(1) = 1;
b2(33) = -1;
a2 = [1 -1];
h2 = filter(b2,a2,lpecg);  %% lowpass

%% All pass
b3 = zeros(1,17);
b3(17) = 1;
a3 =1;
h3 = filter(b3,a3,lpecg);  %% allpass

%% highpass = allpass-lowpass
p2 = h3 - h2/32;   
hp_ecg=p2;


%% signal derivation
b4 = [1 2 0 -2 -1];
a4 = 1;
h4=filter(b4,a4,p2);   
deriv_ecg=h4;

%%% Squaring %%%
h4 = h4.^2;
sq_ecg=h4;

% Integration
b5=ones(1,30)/30;
a5= 1;
op = filter(b5,a5,h4);
moy_ecg=op;

% QRS detection
[R_loc,QRS_width,Q_loc,S_loc]=peak_features(op,b1,b2,b3,b4,b5,necg,Fs); 





