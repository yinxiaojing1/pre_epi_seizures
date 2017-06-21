function [QRS_loc,QRS_width_buff,QRS_buff_on,QRS_buff_off]=peak_features(preproc_sig,b1,b2,b3,b4,b5,ecg,Fs)
% peak_features.m
% Peaks selection 
% QRS Beat Detection using Pan-TompKins Algorithm 
% associated files of the Bayesian P and T wave delineation and waveform
% estimation Toolbox
% Inputs:
%
% Outputs:
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

bpb=[b1, b2, b3];
[signal_recherche,loc_peak,recal]=marque(bpb,b4,b5,preproc_sig,Fs);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%          Exact R wave localisation            
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
[QRS_loc]=Rwavexact(b1,b2,b3,b4,b5,loc_peak,recal,ecg,Fs);
                         
%%**************************************************************
%%*
%%* Measure of the features that characterise the detected QRS
%%*  
%%**************************************************************  

 [QRS_width_buff,QRS_buff_on,QRS_buff_off]=QRSestim(QRS_loc,ecg,Fs);

    
    
function [signal_recherche,loc_peak,Ndeb]=marque(b1,b2,b3,op,Fs)
        
%% On laissera tomber les Ndeb premiers points pour la recherche des ECG
Ndeb=(length(b1)+length(b2)+length(b3));
signal_recherche=op(Ndeb+1:end);
MS200=round(0.2*Fs);
MS300=round(0.3*Fs);
fact=0.3;
loc_peak=zeros(0,0);
peak=signal_recherche(1);
timesincemax=0;
TH_init=max(signal_recherche(1:300));
buff_QRS=[TH_init TH_init TH_init TH_init];
buff_bruit=[0 0 0 0];
qmean=mean(buff_QRS);
nmean=mean(buff_bruit);
TH=thresholding(qmean, nmean, fact);
i=1;
loc_peak_temp=i;
titi=0;

while i<length(signal_recherche)
    i=i+1;
    timesincemax=timesincemax+1;
    
    if (signal_recherche(i)>peak) & timesincemax<MS300
        peak=signal_recherche(i);
        loc_peak_temp=i;
        timesincemax=0;
        
    elseif (signal_recherche(i)<peak) & timesincemax>=MS300
       
        %% test pour savoir si le peak est du bruit ou bien un QRS
        if peak>TH
            %%disp('yes');
            
            buff_QRS=[buff_QRS(2:end) peak];
            qmean=mean(buff_QRS);
            TH=thresholding(qmean, nmean, fact);
            
            loc_peak=[loc_peak loc_peak_temp]; 
            
            if (loc_peak(end)+MS200)<length(signal_recherche)
              i=loc_peak(end)+MS200;
            end
            
            peak=signal_recherche(i);
            timesincemax=0;
            
        else 
            
            buff_bruit=[buff_bruit(2:end) peak];
            nmean=mean(buff_bruit);
            TH=thresholding(qmean, nmean, fact);
            if (loc_peak_temp+MS200)<length(signal_recherche)
              i=loc_peak_temp+MS200;
            end
            peak=signal_recherche(i);
            loc_peak_temp=i;
            timesincemax=0;
        end
       
    end
    
end
        
       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% R wave extraction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [QRS_loc]=Rwavexact(b1,b2,b3,b4,b5,loc_peak,recal,necg,Fs)


FILTER_DELAY=fix((length(b1)-1)/2)+fix((length(b2)-1)/2)+fix((length(b3)-1)/2)+fix((length(b4)-1)/2)+fix((length(b5)-1)/2);
ind_recal=loc_peak+recal-FILTER_DELAY;


MS_PER_SAMPLE=fix(1000/Fs+0.5);
MS200=fix(200/MS_PER_SAMPLE+0.5);
MS400=fix(400/MS_PER_SAMPLE+0.5);  

QRS_loc=zeros(0,0);

for i=1:length(loc_peak)                            
    
    if ind_recal(i)>MS400 & ind_recal(i)+MS400<length(necg)
                                
        WIN_SEARCH=MS200;
    
        %% this signal can be saved ain a circular buffer of length=2*MS400; then an indice should be
        %% given as an input
        S_Rw_loc=necg(ind_recal(i)-MS400+1:ind_recal(i)+MS400);
                            
        [R_loc,R_pos,R_neg]=R_wave_loc(S_Rw_loc,ind_recal(i),Fs,WIN_SEARCH);
                     
        QRS_loc=[QRS_loc R_loc-MS400];
        
    elseif ind_recal(i)>MS200 & ind_recal(i)+MS200<length(necg)
    
        WIN_SEARCH=MS200;
                     
        %% this signal can be saved ain a circular buffer of length=2*MS400; then an indice should be
        %% given as an input
    
        S_Rw_loc=necg(ind_recal(i)-WIN_SEARCH+1:ind_recal(i)+WIN_SEARCH);
                            
        [R_loc,R_pos,R_neg]=R_wave_loc(S_Rw_loc,ind_recal(i),Fs,WIN_SEARCH);
                     
        QRS_loc=[QRS_loc R_loc-WIN_SEARCH];

    
    else
    
        disp('precise R wave location is not possible')
        QRS_loc=[QRS_loc ind_recal(i)];
        
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%                              R_wave_loc
%%  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [QRS_loc,QRS_pos,QRS_neg]=R_wave_loc(varargin)

%% R wave location after QRS detection
%% Once a QRS has been detected using the Output signal from the pre-processing
%% we have to carefully locate it. Indeed the QRS detection is based on the maxima location
%% of a transformed signal, according to the  
%%
%% INPUTS
%%
%%  signal : portion of the raw signal, centred aroud the detected QRS
%%  ind_recal : re-centred location of the detected QRS   
%%  fs : sampling frequency
%%  WIN_SEARCH : Length of the search interval for the R location
%%
%% OUTPUTS
%% 
%%  QRS_loc : R_location : indice of the true R wave location
%%  QRS_pos : If R is up then QRS_pos=QRS_loc
%%  QRS_neg : If R is down then QRS_neg=QRS_loc


if nargin==0 | nargin==1
    disp('pas d''entree correcte pour la location de l''onde R')
    return
elseif nargin==2
    signal=varargin{1};
    ind_recal=varargin{2};
    % default sampling frequency
    fs=250;
    %% Default : the search window is firstly 200ms around the proposed QRS location
    WIN_SEARCH=fix(0.2*fs+0.5);
    
elseif nargin==3
    signal=varargin{1};
    ind_recal=varargin{2};
    fs=varargin{3};
    %% Default : the search window is firstly 200ms around the proposed QRS location
    WIN_SEARCH=fix(0.2*fs+0.5);
    
elseif nargin==4
    signal=varargin{1};
    ind_recal=varargin{2};
    fs=varargin{3};
    WIN_SEARCH=varargin{4};
    
end

%%-----------------------------------------------------------
%%
%%      Signal segment pre-proc for R wave location
%%
%%-----------------------------------------------------------

%% 
interv_search=2*fix(WIN_SEARCH/2);     
                                            
%% For the R wave location we use the derivative of the 
%% (raw) data (TBC if we use the raw or some pre-proc before the search

%% Pre_proc : noise filtering using dyadic wavelet
[s1,s2,s3,s4]=Beat_analysis_preproc(signal,[]);
signal_f=s3;

%% Low pass filtering for noise removing
[b1] = ones(1,6);%0;%[1 0 0 0 0 0 -2 0 0 0 0 0 1]/32;
[a1] = 1;%[1 -2 1];
signal_f=filter(b1,a1,signal_f);
delay=fix((length(b1)-1)/2);

%% bandpass filtering

%FC=12;
%Npoint=8;
%[signal_pb,delay_bp] =banpass(signal_f,FC,fs,Npoint);

%% on met ou non ce filtrage en place
signal_pb=signal_f;
delay_bp=delay;

%% Derivative of the signal : high pass filtering

%% filtering features
b_der = [1 0 -1];
b_der = b_der/4;
a_der = 1;

derivative=filter(b_der,a_der,signal_pb);

deriv_delay=fix((length(b_der)-1)/2)+delay_bp;
    
derivation_analysed=derivative(length(signal)/2+deriv_delay-interv_search/2:length(signal)/2+deriv_delay+interv_search/2);

%derivation_analysed=derivee1(ind_recal-interv_search/2:ind_recal+interv_search/2);

%% Min/Max location around the located R wave

[val_max,loc_max] = max(derivation_analysed);
[val_min,loc_min] = min(derivation_analysed);
                           
%% Comparison of the MIN/MAX magnitudes
%% ecart max for the Rwave location
%% may put it as an input
    
WIN_MAX=fix(0.15*fs+0.5);
                            
if ((abs(val_min/2) > abs(val_max) ) | (abs(val_max/2) > abs(val_min))) %& (abs(loc_max-loc_min)<WIN_SEARCH)
                         
    %disp('sur cet intervalle pas de couple min max correspondant')
    %% on tente d'elargir l'intervalle
    interv_search=2*fix(1.5*WIN_SEARCH/2); 
    derivation_analysed=derivative(length(signal)/2+deriv_delay-interv_search/2:length(signal)/2+deriv_delay+interv_search/2);
                            
    [val_max,loc_max] = max(derivation_analysed);
    [val_min,loc_min] = min(derivation_analysed);
    
    if abs(loc_max-loc_min)>WIN_MAX
        disp('max correspondant trop ecartes... peut pas un QRS')
    end
                               
end
   

%% we compute the true position of the detected R wave
%% according to the diffrent delays

%% indice recalage
if loc_max>=loc_min
                                
    [val, ind_loc]=min(abs(derivation_analysed(loc_min:loc_max)));
    
    %% -2 is the number of sample recaling (one for loc_min, one for length(signal)/2 one for interv_search/2)
    %% no filter delay included : already taken into account in the segmentation
    ind_loc=ind_loc+loc_min+ind_recal+length(signal)/2-interv_search/2-2; 
    
    %ind_loc=ind_loc+loc_min+ind_recal-interv_search/2-FILTER_DELAY+PRE_BLANK;
     
    QRS_neg=zeros(0,0);
    QRS_pos=ind_loc;
    QRS_loc=ind_loc;
                                
else 
                                
    [val, ind_loc]=min(abs(derivation_analysed(loc_max:loc_min)));

    %% -2 is the number of sample recaling (one for loc_min, one for length(signal)/2 and interv_search/2)
    ind_loc=ind_loc+loc_max+length(signal)/2+ind_recal-interv_search/2-2;   
    %ind_loc=ind_loc+loc_min+ind_recal-interv_search/2-FILTER_DELAY+PRE_BLANK;
                          
    QRS_neg=ind_loc;
    QRS_pos=zeros(0,0);
    QRS_loc=ind_loc;
                                 
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%*
%%* Measure of the features that characterise the detected QRS
%%*  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 

   
function [QRS_width_buff,QRS_buff_on,QRS_buff_off]=QRSestim(QRS_loc,necg,Fs)

%% buffer initialisation
QRS_width_buff=zeros(0,0); 
QRS_buff_on=zeros(0,0);
QRS_buff_off=zeros(0,0);
%% Constantes pour la segmentation des beats detectes
MS_PER_SAMPLE=fix(1000/Fs+0.5);
MS200=fix(200/MS_PER_SAMPLE+0.5);
MS400=fix(400/MS_PER_SAMPLE+0.5);  

%% we compute the pre-processing specific to QRS analysis
half_beat_length=2^(nextpow2(round(1.2*MS400))+1);

for i=1:length(QRS_loc)
    
    if QRS_loc(i)-half_beat_length+1>0 & QRS_loc(i)+half_beat_length<length(necg)
                                
        %% segmentation around each detected R wave    
        interv=QRS_loc(i)-half_beat_length+1:QRS_loc(i)+half_beat_length;
        beat_segment=necg(interv);
    
        %% pre-processing for noise reduction
        [s1,s2,s3,s4]=Beat_analysis_preproc(beat_segment);
                                
        %% segmentation prper to QRS width examination
        interv_recherche=-MS200-1:1:MS200;
        signal_qrsw=s3(length(beat_segment)/2+interv_recherche);
    
        [QRS_width,QRS_on,QRS_off]=QRS_width_func(signal_qrsw,'no');
                                
    
    elseif (QRS_loc(end)-MS200+1)>0 & (QRS_loc(end)+MS200)<length(necg)  %% that shall be thruth according to the principle
                                                                               %% of the QRS detection (filtering delay...) and time waiting
                                                                               %% after a potential detection
        interv_recherche=-MS200+1:1:MS200;
                                
        signal_qrsw=necg(QRS_loc(end)+interv_recherche);
        %% here the pre-processing is performed in the QRS_width module
        [QRS_width,QRS_on,QRS_off]=QRS_width_func(signal_qrsw,'ok');
                                
    else
        disp('no way to segment the beat')
        QRS_width=0;
        QRS_on=0;
        QRS_off=0;
    end

    %% sauvegarde des position et largeur dans des buffers
    QRS_width_buff=[QRS_width_buff QRS_width]; 
                            
    QRS_buff_on=[QRS_buff_on QRS_on+QRS_loc(i)-MS200];
    QRS_buff_off=[QRS_buff_off QRS_off+QRS_loc(i)-MS200];
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%          QRS_width determination
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [QRS_width,QRS_on,QRS_off]=QRS_width_func(varargin)

%% QRS width estimation after QRS detection
%% Once a QRS has been detected using the Output signal from the pre-processing
%% we may have to estimate its width for further classification. 
%%  rem : this one could be implanted during the R wave location process
%% INPUTS
%%
%%  signal : portion of the (raw) signal (we have to see if the use of the
%%  roughly filtered signal (before moving integration would be better), (800ms TBC) around the R wave
%%  pre_proc : pre_proc performed inside or outside of this module
%%  FC : central frequency for the bandpass filter   
%%  Npoint : nb point of the bandpass filter RI
%%  fs : sampling frequency
%%
%% OUTPUTS
%% 
%%  QRS_width : length (in samples) of the QRS
%%  QRS_on : location of the onset
%%  QRS_off : location of the offset


if nargin==0 
    disp('pas d''entree correcte pour la location de l''onde R')
    return
elseif nargin==1 
    signal=varargin{1};
    pre_proc='no';
    FC=12;
    Npoint=8;
    % default sampling frequency
    fs=250;
elseif nargin==2
    signal=varargin{1};
    pre_proc=varargin{2};
    FC=12;
    Npoint=8;
    % default sampling frequency
    fs=250;
elseif nargin==3
    signal=varargin{1};
    pre_proc=varargin{2};
    FC=varargin{3};
    Npoint=8;
    % default sampling frequency
    fs=250;
elseif nargin==4
    signal=varargin{1};
    pre_proc=varargin{2};
    FC=varargin{3}; 
    Npoint=varargin{4};
    % default sampling frequency
    fs=250;
elseif nargin==5
    signal=varargin{1};
    pre_proc=varargin{2};
    FC=varargin{3};
    Npoint=varargin{4};
    fs=varargin{5};
end

%% can be transferred as an input
fact=10; %15; 40;

%%-----------------------------------------------------------
%%
%%      Signal segment pre-proc for QRS width
%%
%%-----------------------------------------------------------

%% due to some high grade noise effect we need some strong preprocessing for
%% QRS-width determination

pre_proc='ok';
%% For the QRS width determination we use the derivative of the signal
%% or the dyadic wavelet analysis model

%% Pre_proc : noise filtering using dyadic wavelet

if strcmp(pre_proc,'ok')
    [s1,s2,s3,s4]=Beat_analysis_preproc(signal,[]);
    signal_f=s3;

    %% low pass filtering for high frequency noise removing
    [b1] = ones(1,10);%%[1 0 0 0 0 0 -2 0 0 0 0 0 1]/32;
    [a1] = 1;%[1 -2 1];
    signal_pb=filter(b1,a1,signal_f);
    delay_bp=fix((length(b1)-1)/2);
    %[signal_pb,delay_bp] = banpass(signal_f,FC,fs,Npoint);
else
    signal_pb=signal;
    delay_bp=0;
end

%%-----------------------------------------------------------
%%
%%      Use of slope criteria to recompute the QRS width
%%
%%-----------------------------------------------------------

%% Derivative of the signal : high pass filtering

%% filtering features   
b_der = [1 0 -1];
b_der = b_der/6;
a_der = 1;

derivative=filter(b_der,a_der,signal_pb);

deriv_delay=fix((length(b_der)-1)/2)+delay_bp;

derivation_analysed=derivative;

%keyboard
%% we look for local min-max of the derivative
%derivation_analysed=derivee1(ind_recal-interv_search/2:ind_recal+interv_search/2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Min/Max location around the located R wave
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %[val_max,loc_max] = max(derivation_analysed);
    %[val_min,loc_min] = min(derivation_analysed);
    % with this technic, some edge effect can occur
    % so we have to look for the first local min max.

portion1=derivation_analysed(length(signal)/2+deriv_delay+1:end);
portion2=derivation_analysed(length(signal)/2+deriv_delay-1:-1:1);

if prod(portion1(1:3)-derivation_analysed(length(signal)/2+deriv_delay))>0 | prod(portion2(1:3)-derivation_analysed(length(signal)/2+deriv_delay))<0
    %% in that case the positive slope is after the R point
    %% search of the first local min-max
    i=2;
    while i<(length(portion1)-3) & ((portion1(i)<portion1(i-1)) | (portion1(i)<portion1(i+1) | portion1(i)<portion1(i+2) | portion1(i)<portion1(i+3)))
        i=i+1;
    end
    loc_max=i+length(signal)/2+deriv_delay;
    val_max=derivation_analysed(loc_max);
        
    i=2;
    while i<(length(portion2)-3) & ((portion2(i)>portion2(i-1)) | (portion2(i)>portion2(i+1) | portion2(i)>portion2(i+2) | portion2(i)>portion2(i+3)))
        i=i+1;
    end
    loc_min=length(signal)/2+deriv_delay-i;
    val_min=derivation_analysed(loc_min);
            
elseif prod(portion1(1:3)-derivation_analysed(length(signal)/2+deriv_delay))<0 | prod(portion2(1:3)-derivation_analysed(length(signal)/2+deriv_delay))>0
        
    %% in that case the positive slope is befor the R point
    %% search of the first local min-max
    i=2;
    while i<(length(portion2)-3) & ((portion2(i)<portion2(i-1)) | (portion2(i)<portion2(i+1) | portion2(i)<portion2(i+2) | portion2(i)<portion2(i+3)))
        i=i+1;
    end
    loc_max=length(signal)/2+deriv_delay-i;
    val_max=derivation_analysed(loc_max);
        
    i=2;
    while i<(length(portion1)-3) & ((portion1(i)>portion1(i-1)) | (portion1(i)>portion1(i+1) | portion1(i)>portion1(i+2) | portion1(i)>portion1(i+3)))
        i=i+1;
    end
    %% in that case loc_min>loc_max
    loc_min=i+length(signal)/2+deriv_delay;
    val_min=derivation_analysed(loc_min);

else
    disp('attention, du bruit doit perturber l''utilisation de la derivee pour la mesure du QRS width');
    val_min=0;
    val_max=0;
    loc_min=0;
    loc_max=0;
end

%% other computation of the loc max
[val_max2,loc_max2] = max(derivation_analysed);
[val_min2,loc_min2] = min(derivation_analysed);

if val_max2~=val_max & (sign(loc_max-loc_min)*loc_max+sign(loc_min-loc_max)*loc_max2)>=0 & (sign(loc_min-loc_max)*loc_min+sign(loc_max-loc_min)*loc_max2)>=0
    val_max=val_max2;
    loc_max=loc_max2;
end

if val_min2~=val_min & (sign(loc_min-loc_max)*loc_min+sign(loc_max-loc_min)*loc_min2)>=0 & (sign(loc_max-loc_min)*loc_max+sign(loc_min-loc_max)*loc_max2)>=0
    val_min=val_min2;
    loc_min=loc_min2;
end
    
slope_max=val_max;
slope_min=val_min;

%% Once we get the slope min and max, we can search for the QRS_onset and offset

%% we can reduce the factor if a onset or offset is (are) not found

if loc_max>=loc_min
    i=loc_min;                            
    while i>3 & ~exist('QRS_on')
        i=i-1;
        if abs(prod(derivation_analysed(i-2:i)))<abs((slope_min/fact)^3)
            QRS_on=i;
        end
    end
        
    i=loc_max;                            
    while i<(length(derivation_analysed)-2) & ~exist('QRS_off')
        i=i+1;
        if abs(prod(derivation_analysed(i:i+2)))<abs((slope_max/fact)^3)
            QRS_off=i;
        end
    end
else 
    i=loc_max;                            
    while i>3 & ~exist('QRS_on')
        i=i-1;
        if abs(prod(derivation_analysed(i-2:i)))<abs((slope_max/fact)^3)
            QRS_on=i;
        end
    end
        
    i=loc_min;                            
    while i<(length(derivation_analysed)-2) & ~exist('QRS_off')
        i=i+1;
        if abs(prod(derivation_analysed(i:i+2)))<abs((slope_min/fact)^3)
            QRS_off=i;
        end
    end
end

if ~exist('QRS_on') | ~exist('QRS_off') | QRS_off==2 | QRS_on==2 | QRS_off==(length(derivation_analysed)-1) | QRS_on==(length(derivation_analysed)-1)
    %disp('attention souci dans le calul du qrs width')
    %% we change the factor if  QRS_on or (and) QRS_off are not found
    
    while (~exist('QRS_on') | ~exist('QRS_off')) & fact>1e-2
        fact=fact/2;
        if loc_max>=loc_min
            i=loc_min;                            
            while i>3 & ~exist('QRS_on')
                i=i-1;
                if abs(prod(derivation_analysed(i-2:i)))<abs((slope_min/fact)^3)
                    QRS_on=i;
                end
            end
        
            i=loc_max;                            
            while i<(length(derivation_analysed)-2) & ~exist('QRS_off')
                i=i+1;
                if abs(prod(derivation_analysed(i:i+2)))<abs((slope_max/fact)^3)
                    QRS_off=i;
                end
            end
        else 
            i=loc_max;                            
            while i>3 & ~exist('QRS_on')
                i=i-1;
                if abs(prod(derivation_analysed(i-2:i)))<abs((slope_max/fact)^3)
                    QRS_on=i;
                end
            end
        
            i=loc_min;                            
            while i<(length(derivation_analysed)-2) & ~exist('QRS_off')
                i=i+1;
                if abs(prod(derivation_analysed(i:i+2)))<abs((slope_min/fact)^3)
                    QRS_off=i;
                end
            end
        end
    end

    if ~exist('QRS_on') 
        QRS_on=1;
    end
    if ~exist('QRS_off') 
        QRS_off=length(derivation_analysed);
    end
end

QRS_width=QRS_off-QRS_on;
QRS_on=QRS_on-deriv_delay;
QRS_off=QRS_off-deriv_delay;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%          QRS_width pre-process determination
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

function [varargout]=Beat_analysis_preproc(varargin)

%% Pre-processing of the beat segment for analysis (identification)
%% De-noising with wavelet analysis, Baseline drift removing by long term evolution
%%
%% INPUTS
%%
%%  signal : portion of the raw signal, centred aroud the R wave
%%  level_max_BLD : Max Decomposition level for baseline drift   
%%  wavelet_BLD : Wavelet type for baseline drift
%%  build_level_BLD : Reconstruction level for approximation
%%  level_max_DEN : Max Decomposition level for de-noising
%%  wavelet_DEN : Wavelet type for de-noising
%%
%% OUTPUTS
%% 
%%  signal_basrem : signal without baseline
%%  baseline
%%  sig_basrem_d1 : denoised signal without baseline, with threshold selection 'sqtwolog'
%%  sig_basrem_d2 : denoised signal without baseline, with threshold selection 'minmax'

%% default parameters

if nargin==0 
    disp('pas d''entree correcte pour la location de l''onde R')
    return
elseif nargin==1
    signal=varargin{1};
    level_max_BLD=8;
    wavelet_BLD='sym4'; 
    build_level_BLD=8;
    level_max_DEN=3;
    wavelet_DEN='sym4';
elseif nargin==2
    signal=varargin{1};
    level_max_BLD=varargin{2}; 
    wavelet_BLD='sym4'; 
    build_level_BLD=varargin{2};
    level_max_DEN=3;
    wavelet_DEN='sym4';
elseif nargin==3
    signal=varargin{1};
    level_max_BLD=varargin{2}; 
    wavelet_BLD=varargin{3}; 
    build_level_BLD=varargin{2};
    level_max_DEN=3;
    wavelet_DEN='sym4';
elseif nargin==4
    signal=varargin{1};
    level_max_BLD=varargin{2}; 
    wavelet_BLD=varargin{3}; 
    build_level_BLD=varargin{4};
    level_max_DEN=3;
    wavelet_DEN='sym4';
elseif nargin==5
    signal=varargin{1};
    level_max_BLD=varargin{2}; 
    wavelet_BLD=varargin{3}; 
    build_level_BLD=varargin{4};
    level_max_DEN=varargin{5};
    wavelet_DEN='sym4';
else
    signal=varargin{1};
    level_max_BLD=varargin{2}; 
    wavelet_BLD=varargin{3}; 
    build_level_BLD=varargin{4};
    level_max_DEN=varargin{5};
    wavelet_DEN=varargin{6};
end


%% If the decomposition level max for baseline drift removing is set as empty in input
%% then no baseline drift removing is performed

if ~isempty(level_max_BLD)
    
    %%***********************************************
    %%
    %% Baseline Drift removing
    %% By digital wavelet decomposition
    %% (not a real time implantation to this point)
    %%
    %%***********************************************

    %% Approximation and detail coeff from level 1 to level_max_BLD
    [C,L] = wavedec(signal,level_max_BLD,wavelet_BLD);

    %% re-buiding of the signal from the approximation coefficients of level niv_reconst
    baseline = wrcoef('a',C,L,wavelet_BLD,build_level_BLD);
    
    %% baseline removing from the original signal
    signal_basrem=signal-baseline;
    
    varargout{1}=signal_basrem;
    varargout{2}=baseline;
    
else
    signal_basrem=signal;
    varargout{1}=signal_basrem;
    varargout{2}=zeros(size(signal));
end

%% If the decomposition level max de-noising is set as empty in input
%% then no de-noising is performed

if ~isempty(level_max_DEN)
    
    %%*****************************************************
    %%
    %% Noise filtering with wavelet analysis thresholding
    %%
    %%*****************************************************
    % We need the wavelet Toolbox here!!!!

    % Find first value in order to avoid edge effects. 
    deb = signal(1);

    % De-noise signal using soft fixed form thresholding 
    % and unknown noise option. 
    % xd = wden(signal_basrem-deb,'sqtwolog','s','mln',level_max,wavelet)+deb;
    sig_basrem_d1 = wden(signal_basrem-deb,'sqtwolog','s','sln',level_max_DEN,wavelet_DEN)+deb;
    %sig_basrem_d1 = signal_basrem;

    % sd = wden(signal_basrem,'minimaxi','s','mln',level_max,wavelet);
    sig_basrem_d2 = wden(signal_basrem-deb,'minimaxi','s','sln',level_max_DEN,wavelet_DEN)+deb;
    %sig_basrem_d2 = signal_basrem;
    %% Outputs
    varargout{3}=sig_basrem_d1;
    varargout{4}=sig_basrem_d2;
    
else
    %% Outputs
    varargout{3}=sig_basrem;
    varargout{4}=sig_basrem;
end

%%***************************************************
%% thresh() calculates the detection threshold 
%% from the qrs mean and noise mean estimates.
%%***************************************************       
%
% INPUTS
% qmean = moyenne du buffer contenant les amplitudes des QRS detectes
% nmean = moyenne du buffer contenant les amplitudes des peaks de bruit detectes
% TH = facteur sur le calcul du threqhold (empirique)
%
% OUTPUTS
% Thresh : seuil pour la detection des QRS


function thresh =thresholding(qmean, nmean, TH)
	
	dmed = qmean - nmean ;
    temp=TH*dmed;
	dmed = temp ;
	thresh = nmean + dmed ; 
