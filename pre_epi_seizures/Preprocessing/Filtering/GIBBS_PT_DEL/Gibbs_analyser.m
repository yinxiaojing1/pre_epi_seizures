
function [b_hat_T,r_hat_T,f_hat_T,b_hat_P,r_hat_P,f_hat_P,baseline_total,x_hat_T,x_hat_P]=Gibbs_analyser(Iterat,x,N,pi_1,sigma2_a,sigma2_alpha,sigma2_gamma,sigma2_gamma2,eta,xi,fshift,onset_list_T,end_list_T,onset_list_P,end_list_P,handles)
% Gibbs_analyser.m
% a PCGS for P and T wave analysis
% associated files of the Bayesian P and T wave delineation and waveform
% estimation Toolbox
% Inputs:
%     est_f --->
%     1 ... with estimation of alpha
%     0 ... with knowledge of alpha
%     Iterat ---> number of iterations of the sampler
%     x ---> observations
%     N ---> wave coeff length
%     pi_1 ---> 1-probability of P and T waves
%     d_min ---> local constrain
%     sigma2_a ---> 
%     sigma2_alpha ---> 
%     eta ---> IG(eta, xi)
%     xi ---> IG(eta, xi)
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
%     (ICASSP'11), 2011, to appear. 
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

% V3.0 modified 11/2011, optimization

sigma2_gamma3 = 0.01*1;
est_iter = Iterat; % vector of iterations at which we estimate
Nr_est = length(est_iter);
K = length(x);
q = K;  % length of f
qc = round(N/2); % the effective nonzero length of f is only qc*2 (or less)
G = length(onset_list_P);
est_f = 1; % 1-- estimate, 0-- do not estimate
% We propose three models for local baseline, 1-- applied, 0-- not applied
est_g3 = 1; % 4 degree polynomial for local baseline (each RR interval)
ec=0; % error counter
clear *_hat coll* *_avg *_fshift 

%%%%%%%%%%%% Initialize Sampler %%%%%%%%%%%%%%%%
p0factor = log((1-pi_1) /10 * sqrt(sigma2_a)); % auxiliary for p_pl_est
b_avg_T = zeros(K,Nr_est);
b_avg_P = zeros(K,Nr_est);
b_hat_T = zeros(K,Nr_est);
b_hat_P = zeros(K,Nr_est);
a_hat_T = zeros(K,Nr_est);
a_hat_P = zeros(K,Nr_est);
a_fshift_T = zeros(K,Nr_est);
a_fshift_P = zeros(K,Nr_est);
r_hat_T = zeros(K,Nr_est);
r_hat_P = zeros(K,Nr_est);
s_hat = zeros(1,Nr_est);
f_hat_T = zeros(q,Nr_est);
f_hat_P = zeros(q,Nr_est);
f_fshift_T = zeros(q,Nr_est);
f_fshift_P = zeros(q,Nr_est);
%err_g = zeros(1,Iterat);
%err_g2 = zeros(1,Iterat);
coll_b_T = zeros(Iterat,K);
coll_b_P = zeros(Iterat,K);
coll_a_T = zeros(Iterat,K);
coll_a_P = zeros(Iterat,K);
coll_s = zeros(Iterat,1);
coll_f_T = zeros(Iterat,q);
coll_f_P = zeros(Iterat,q);

coll_gamma3 = zeros(Iterat,G*6);

load B; 
N=20;   % number of Hermite functions
H_0 = zeros((q-2*qc)/2,N);
H = [H_0; B(512/2+(-qc+1:qc),1:N); H_0]/100;
% make basis smaller to avoid breakoff at inversion sqrt_sg2_inv

M_P = zeros(K,G);
for i=1:G
    M_P(onset_list_P(i):end_list_P(i),i)=1;
    M_P(:,i) = M_P(:,i)/norm(M_P(:,i));
end
M_T = zeros(K,G);
for i=1:G
    M_T(onset_list_T(i):end_list_T(i),i)=1;
    M_T(:,i) = M_T(:,i)/norm(M_T(:,i));
end
M = M_P + M_T;
non_QRS = sum(M'>0)';

M3 = zeros(K,G*6);
index = 1;
for i=1:6:G*6
    interval_len = end_list_P(index) - onset_list_T(index)+1;
    M3(onset_list_T(index):end_list_P(index),i) = 1;  % degree 0
    M3(onset_list_T(index):end_list_P(index),i+1) = (linspace(1,interval_len,interval_len)); % degree 1
    M3(onset_list_T(index):end_list_P(index),i+2) = (linspace(1,interval_len,interval_len)).^2; % degree 2
    M3(onset_list_T(index):end_list_P(index),i+3) = (linspace(1,interval_len,interval_len)).^3; % degree 3
    M3(onset_list_T(index):end_list_P(index),i+4) = (linspace(1,interval_len,interval_len)).^4; % degree 4
    M3(onset_list_T(index):end_list_P(index),i+5) = (linspace(1,interval_len,interval_len)).^5; % degree 5    
    M3(:,i) = M3(:,i)/norm(M3(:,i));
    M3(:,i+1) = M3(:,i+1)/norm(M3(:,i+1));
    M3(:,i+2) = M3(:,i+2)/norm(M3(:,i+2));
    M3(:,i+3) = M3(:,i+3)/norm(M3(:,i+3));
    M3(:,i+4) = M3(:,i+3)/norm(M3(:,i+4));
    M3(:,i+5) = M3(:,i+3)/norm(M3(:,i+5));
    index = index +1;
end
gamma3 = zeros(G*6,1);
RH = zeros(q,N);
RH2 = zeros(q,N);
b_est_T = zeros(K,1);
b_est_P = zeros(K,1);
a_est_T = sqrt(sigma2_a/2) * (randn(K,1)); % initialize a
a_est_P = sqrt(sigma2_a/2) * (randn(K,1)); % initialize a
r_est_T = a_est_T .* b_est_T;
r_est_P = a_est_P .* b_est_P;

gamma3_est = sqrt(sigma2_gamma3/2) * randn(G*6,1);
sigma2_n_est = gamrnd(xi, 1/eta, [1 1]); % initialize sigma2_n
sigma2_n_est = 1 / sigma2_n_est;
f_est_P = zeros(q,1);
f_est_P(q/2) = 1;
f_est_T = zeros(q,1);
f_est_T(q/2) = 1;
x_est_P = zeros(K,1); %remove
x_est_T = zeros(K,1); %remove

warning off MATLAB:divideByZero

%prepare LED progress bar
LED=10;  %You can use more LED lights. Create them in 'guide' for main GUI
for n=1:LED
    temp = findobj('Tag',['textLED' num2str(n)]);
    LEDhandles(n)=temp(1);  
end
stepsize=1/LED;  %fraction indicated by each LED light
progressmark=stepsize; %initialization 
set(handles.editLEDbkground,'Visible','on'); 
%set(handles.textLEDpercentDone,'Visible','on');
percentDone_string=[num2str(round(0/Iterat*100)),'% Done'];
set(handles.textLEDpercentDone,'String',percentDone_string,'Visible','on'); %LED %done
pause(0.00000001)  %need this so their showing up will not have a delay
%LED progress bar preparation finished

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     BEGIN PCG SAMPLER
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:Iterat
    
    %%%%%%%%%%% begin {sample alpha_T} %%%%%%%%%%%%
    for rh_ind = 1:N
        rh = conv(r_est_T,H((q-2*qc)/2+(1:2*qc),rh_ind));
        RH(1:q,rh_ind) = rh(K-1-((q)/2-1)-((q-2*qc)/2)+(1:q));
    end   
    RH = diag(non_QRS) * RH;  
    RHHR = real(RH' * RH);
    sg2_inv =  (1/sigma2_n_est * RHHR + diag(1/sigma2_alpha * ones(1,N)) );
    sqrt_sg2_inv = chol(sg2_inv);
    if size(sqrt_sg2_inv)==[N N] % check if factorization was successful
        my_f =  ( RH' * (x - x_est_P - M3 * gamma3_est) ) / sigma2_n_est;
        alpha_est_T =  (sqrt_sg2_inv\(sqrt_sg2_inv')\my_f + sqrt_sg2_inv\(randn(N,1))); % sample alpha
        
        %%%%%% begin {time-shift compensation} to make sure b_T located at its peak %%%%%%
        thrld = 1/2; % propose a shift at every iteration
        if rand<2*thrld
            [val_alpha pos_alpha] = max(abs(H*alpha_est_T));
            uset=-[q/2-pos_alpha]; % set of possible shifts
            u = uset(min(round(length(uset)*rand+0.5),length(uset))); % choose one shift
            b_est2 = circshift(b_est_T,u);
            b_est2(mod([0:u-1 u:-1],K)+1) = 0;
            a_est2 = circshift(a_est_T,u);
            r_est2 = b_est2'.*(a_est2).' ;
            for rh_ind = 1:N
                rh2 = conv(r_est2,H((q-2*qc)/2+(1:2*qc),rh_ind));
                RH2(1:q,rh_ind) = rh2(K-1-((q)/2-1)-((q-2*qc)/2)+(1:q));
            end
            RHHR2 = real(RH2' * RH2);
            sg2_inv2 =  (1/sigma2_n_est * RHHR2 + diag(1/sigma2_alpha * ones(1,N)) );
            sqrt_sg2_inv2 = chol(sg2_inv2);
            if 1
                b_est_T = b_est2;
                a_est_T = a_est2;
                RH = RH2;
                sqrt_sg2_inv = sqrt_sg2_inv2;
            end
        end
        %%%%%%% end {time-shift compensation} %%%%%%%
        my_f =  ( RH' * (x - x_est_P - M3 * gamma3_est) ) / sigma2_n_est;
        alpha_est_T =  (sqrt_sg2_inv\(sqrt_sg2_inv')\my_f + sqrt_sg2_inv\(randn(N,1))); % sample alpha
        f_est_T = (H * alpha_est_T);
        [v p] = max(abs(f_est_T)); % now normalize to avoid scale ambiguity
        a_est_T = a_est_T * f_est_T(p);
      %  alpha_est_T = alpha_est_T / f_est_T(p);
        f_est_T = f_est_T / f_est_T(p);
    else
        ec = ec+1 % counter for problems with cholesky
    end
    if est_f==0
        f_est_T = f_T; % skip estimation of alpha
    end
    %%%%%%%%%%%% end {sample alpha_T} %%%%%%%%%%%%%%
    
    x_est_T = [zeros(q/2-qc,1); conv(f_est_T(q/2+(-qc+1:qc)),r_est_T); zeros(q/2-qc,1)];
    x_est_T = x_est_T(q/2+(1:K)) .* non_QRS;
    
    %%%%%%%%%%% begin {sample alpha_P} %%%%%%%%%%%%
    for rh_ind = 1:N
        rh = conv(r_est_P,H((q-2*qc)/2+(1:2*qc),rh_ind));
        RH(1:q,rh_ind) = rh(K-1-((q)/2-1)-((q-2*qc)/2)+(1:q));
    end
    RH = diag(non_QRS) * RH;
    RHHR = real(RH' * RH);
    sg2_inv =  (1/sigma2_n_est * RHHR + diag(1/sigma2_alpha * ones(1,N)) );
    sqrt_sg2_inv = chol(sg2_inv);
    if size(sqrt_sg2_inv)==[N N] % check if factorization was successful
        my_f =  ( RH' * (x - x_est_T - M3 * gamma3_est) ) / sigma2_n_est;
        alpha_est_P =  (sqrt_sg2_inv\(sqrt_sg2_inv')\my_f + sqrt_sg2_inv\(randn(N,1))); % sample alpha
        %%%%%% begin {time-shift compensation} to make sure b_P located at its peak %%%%%%
        thrld = 1/2; % propose a shift at every iteration
        if rand<2*thrld
            [val_alpha pos_alpha] = max(abs(H*alpha_est_P));
            uset=-[q/2-pos_alpha]; % set of possible shifts
            u = uset(min(round(length(uset)*rand+0.5),length(uset))); % choose one shift
            b_est2 = circshift(b_est_P,u);
            b_est2(mod([0:u-1 u:-1],K)+1) = 0;
            a_est2 = circshift(a_est_P,u);
            r_est2 = b_est2'.*(a_est2).' ;
            for rh_ind = 1:N
                rh2 = conv(r_est2,H((q-2*qc)/2+(1:2*qc),rh_ind));
                RH2(1:q,rh_ind) = rh2(K-1-((q)/2-1)-((q-2*qc)/2)+(1:q));
            end
            RHHR2 = real(RH2' * RH2);
            sg2_inv2 =  (1/sigma2_n_est * RHHR2 + diag(1/sigma2_alpha * ones(1,N)) );
            sqrt_sg2_inv2 = chol(sg2_inv2);
            if 1
                b_est_P = b_est2;
                a_est_P = a_est2;
                RH = RH2;
                sqrt_sg2_inv = sqrt_sg2_inv2;
            end
        end
        %%%%%%% end {time-shift compensation} %%%%%%%
        my_f =  ( RH' * (x - x_est_T - M3 * gamma3_est) ) / sigma2_n_est;
        alpha_est_P =  (sqrt_sg2_inv\(sqrt_sg2_inv')\my_f + sqrt_sg2_inv\(randn(N,1))); % sample alpha
        f_est_P = (H * alpha_est_P);
        [v p] = max(abs(f_est_P)); % now normalize to avoid scale ambiguity
        a_est_P = a_est_P * f_est_P(p);
        %alpha_est_P = alpha_est_P / f_est_P(p);
        f_est_P = f_est_P / f_est_P(p);
    else
        ec = ec+1 % counter for problems with cholesky
    end
    if est_f==0
        f_est_P = f_P; % skip estimation of alpha
    end
    %%%%%%%%%%%% end {sample alpha_P} %%%%%%%%%%%%%%
    
    x_est_P = [zeros(q/2-qc,1); conv(f_est_P(q/2+(-qc+1:qc)),r_est_P); zeros(q/2-qc,1)];
    x_est_P = x_est_P(q/2+(1:K)) .* non_QRS;
    eng1=cumsum(abs(f_est_T(end:-1:1)).^2);
    eng_T=[eng1((q+2)/2:q); eng1(q)*ones(K-(q-2)/2,1)]; % auxiliary for sg
    
    %%%%%%%%%%%%% begin {sample b_T,a_T} %%%%%%%%%%%%%
    for k_ind = 1:length(onset_list_T);
        k = onset_list_T(k_ind);
        k_max = end_list_T(k_ind);
        b_est_T(k:k_max)=0;
        b1_est = b_est_T;
        sg = 1 ./ sqrt(    eng_T(    (k:k_max)-k+1 )       / sigma2_n_est + 1/sigma2_a   );	% auxiliary for p_est
        r1_est = a_est_T.*b1_est;
        x_est = [zeros(q/2-qc,1); conv(f_est_T(q/2+(-qc+1:qc)),r1_est); zeros(q/2-qc,1)];
        eps = ((x - x_est_P - M3 * gamma3_est) - x_est(q/2+(1:K)) .* non_QRS);
        my= [zeros(q/2-qc,1); conv( conj(f_est_T(q/2+1-1+(qc:-1:-qc+1))), eps); zeros(q/2-qc,1)] / sigma2_n_est;
        my = sg.^2 .* my(q/2+(k:k_max)); % auxiliary for p_est
        temp=abs(my).^2./sg.^2/2; % auxiliary for p_est
        mmt=max(max(temp));       % auxiliary for p_est
        p_est = sg.^2 .* exp((temp)-mmt)*pi_1  ; % p(b_{k:k_max})
        p_est = [exp(p0factor-mmt); p_est] / (sum(sum(p_est)) + exp( p0factor-mmt));
        u = rand(1);
        k_rel = sum( u>cumsum(p_est)) - 1;
        if k_rel >= 0
            b_est_T(k+k_rel) = 1;
            % !!!positivity ambiguity!!!
            if mean(a_est_T) >=0
                my(1+k_rel) = abs(my(1+k_rel));
            end
            a_est_T(k+k_rel) = my(1+k_rel) + sg(1+k_rel) * randn(1);  % sample a_k
        end
    end
    %%%%%%%%%%%%% end {sample b_T,a_T} %%%%%%%%%%%%%%%
    
    r_est_T = b_est_T.*(a_est_T) ;
    x_est_T = [zeros(q/2-qc,1); conv(f_est_T(q/2+(-qc:qc-1)),r_est_T); zeros(q/2-qc,1)];
    x_est_T = x_est_T(q/2+(1:K)) .* non_QRS;
    eng1=cumsum(abs(f_est_P).^2);
    eng_P=[ eng1(q)*ones(K-(q-2)/2,1); eng1(q:-1:(q+2)/2)]; % auxiliary for sg
    
    %%%%%%%%%%%%% begin {sample b_P,a_P} %%%%%%%%%%%%%
    for k_ind = 1:length(onset_list_P);
        k = onset_list_P(k_ind)+10;
        k_max = end_list_P(k_ind);
        b_est_P(k:k_max)=0;
        b1_est = b_est_P;
        sg = 1 ./ sqrt(    eng_P(    (k:k_max)-k_max+K )       / sigma2_n_est + 1/sigma2_a   );	% auxiliary for p_est
        r1_est = a_est_P.*b1_est;
        x_est = [zeros(q/2-qc,1); conv(f_est_P(q/2+(-qc+1:qc)),r1_est); zeros(q/2-qc,1)];
        eps = ((x - x_est_T - M3 * gamma3_est) - x_est(q/2+(1:K)) .* non_QRS);
        my= [zeros(q/2-qc,1); conv( conj(f_est_P(q/2+1-1+(qc:-1:-qc+1))), eps); zeros(q/2-qc,1)] / sigma2_n_est;
        my = sg.^2 .* my(q/2+(k:k_max)); % auxiliary for p_est
        temp=abs(my).^2./sg.^2/2; % auxiliary for p_est
        mmt=max(max(temp));       % auxiliary for p_est
        p_est = sg.^2 .* exp((temp)-mmt)*pi_1  ; % p(b_{k:k_max})
        p_est = [exp(p0factor-mmt); p_est] / (sum(sum(p_est)) + exp( p0factor-mmt));
        u = rand(1);
        k_rel = sum( u>cumsum(p_est)) - 1;
        if k_rel >= 0
            b_est_P(k+k_rel) = 1;
            % !!!positivity ambiguity!!!
            if mean(a_est_P) >=0
                my(1+k_rel) = abs(my(1+k_rel));
            end
            a_est_P(k+k_rel) = my(1+k_rel) + sg(1+k_rel) * randn(1);  % sample a_k
        end
    end
    %%%%%%%%%%%%% end {sample b_P,a_P} %%%%%%%%%%%%%%%
    r_est_P = b_est_P.*(a_est_P) ;
    %x_est_P = [zeros(q/2-qc,1); conv(f_est_P(q/2+(-qc+1:qc)),r_est_P); zeros(q/2-qc,1)];
    x_est_P = [zeros(q/2-qc,1); conv(f_est_P(q/2+(-qc:qc-1)),r_est_P); zeros(q/2-qc,1)];
    x_est_P = x_est_P(q/2+(1:K)) .* non_QRS;
        
    %%%%%%%%%%% begin {sample gamma3} %%%%%%%%%%%%
    M3M3 = real(M3' * M3);
    sg2_g3_inv =  (1/sigma2_n_est * M3M3 + diag(1/sigma2_gamma3 * ones(1,G*6)) );
    sqrt_sg2_g3_inv = chol(sg2_g3_inv);
    if size(sqrt_sg2_g3_inv)==[G*6 G*6] % check if factorization was successful
        my_g3 =  ( M3' * (x - x_est_T - x_est_P) ) / sigma2_n_est;
        gamma3_est =  (sg2_g3_inv\my_g3 + sqrt_sg2_g3_inv\(randn(G*6,1))); % sample alpha
    else
        ec = ec+1 % counter for problems with cholesky
    end
    if est_g3==0
        gamma3_est = gamma3; % skip estimation of gamma2
    end
    %%%%%%%%%%%% end {sample gamma3} %%%%%%%%%%%%%%
    
    %%%%%%%%%% begin {sample sigma2_n} %%%%%%%%%%%
    sigma2_n_est = gamrnd(xi+K/2, 1/(eta + sum(abs( (x - M3 * gamma3_est )-x_est_T - x_est_P ).^2)/2), [1 1]);
    sigma2_n_est = 1 / sigma2_n_est;
    %%%%%%%%%%%% end {sample sigma2_n} %%%%%%%%%%%
    
    coll_b_T(i,:) = b_est_T';
    coll_b_P(i,:) = b_est_P';
    coll_a_T(i,:) = r_est_T.';
    coll_a_P(i,:) = r_est_P.';
    coll_s(i,:) = sigma2_n_est;
    coll_f_T(i,:) = f_est_T.';
    coll_f_P(i,:) = f_est_P.';
    coll_gamma3(i,:) = gamma3_est.';
    
    %set LED progress bar below
    percentDone_string=[num2str(round(i/Iterat*100)),'% Done'];
    set(handles.textLEDpercentDone,'String',percentDone_string,'Visible','on'); %LED %done
    if i/Iterat >= progressmark*0.99   %Use a smaller number (e.g., 0.8) to show LED earlier
        set(LEDhandles(round(progressmark/stepsize)),'Visible','on');
        pause(0.000000001)  %need this so that LED lighting will not fall behind %done!
        progressmark=progressmark+stepsize;
    end
    %LED progress bar block ends
    
end

%turn off LED progress bar
for n=1:LED; set(LEDhandles(n),'Visible','off'); end %turn of LED bar    
set(handles.editLEDbkground,'Visible','off'); 
set(handles.textLEDpercentDone,'Visible','off');
%finished turning off LED progress bar

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     END PCG SAMPLER
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     BEGIN DETECTOR/ESTIMATOR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
burnin = round(Iterat*0.6);
iii=0;
gamma3_hat = zeros(G*6,est_iter);

for i = est_iter
    iii=iii+1;
    f_fshift_T(:,iii) = mean(coll_f_T(burnin:i,:),1).';
    f_hat_T(:,iii) = f_fshift_T(:,iii);
    b_avg_T(1:K,iii) = mean(coll_b_T(burnin:i,:),1)';
    
%%%%%%%%%% begin {block detector for b_T} %%%%%%%%%%%
    b_hat_T(1:K,iii) = 0; 
    for ib = 1:length(onset_list_T)
    [v p] = max(b_avg_T(onset_list_T(ib):end_list_T(ib),iii));
        if v> 1-sum(b_avg_T(onset_list_T(ib):end_list_T(ib),iii))
            b_hat_T( p + onset_list_T(ib)-1 ,iii) = 1;
        else
            disp('the probability of haveing a T-wave in this interval is smaller than the threshold.')
            sum(b_avg_T(onset_list_T(ib):end_list_T(ib),iii))
        end
    end
%%%%%%%%%%% end {block detector for b_T} %%%%%%%%%%%%
    
    a_sum=sum(coll_a_T(burnin:i,:)~=0,1);
    a_fshift_T(1:K,iii) = (sum(coll_a_T(burnin:i,:),1)./(a_sum + (a_sum==0) )).';
    a_hat_T(1:K,iii) = a_fshift_T(1:K,iii);
    r_hat_T(1:K,iii) = b_hat_T(:,iii).*a_hat_T(:,iii);
    s_hat(1:K,iii) = mean(coll_s(burnin:i),1)';
    gamma3_hat(1:G*6,iii) = mean(coll_gamma3(burnin:i,:),1)';
end

iii=0;
for i = est_iter    
    iii=iii+1;
    f_fshift_P(:,iii) = mean(coll_f_P(burnin:i,:),1).';
    f_hat_P(:,iii) = f_fshift_P(:,iii);
    b_avg_P(1:K,iii) = mean(coll_b_P(burnin:i,:),1)';
    
%%%%%%%%%% begin {block detector for b_P} %%%%%%%%%%%
    b_hat_P(1:K,iii) = 0;    
    for ib = 1:length(onset_list_P)
    [v p] = max(b_avg_P(onset_list_P(ib):end_list_P(ib),iii));
        if v> 1-sum(b_avg_P(onset_list_P(ib):end_list_P(ib),iii))
            b_hat_P( p + onset_list_P(ib)-1 ,iii) = 1;
        else
            disp('the probability of haveing a P-wave in this interval is smaller than the threshold.')
            sum(b_avg_P(onset_list_P(ib):end_list_P(ib),iii))            
        end
    end 
%%%%%%%%%%% end {block detector for b_P} %%%%%%%%%%%%
    
    a_sum=sum(coll_a_P(burnin:i,:)~=0,1);
    a_fshift_P(1:K,iii) = (sum(coll_a_P(burnin:i,:),1)./(a_sum + (a_sum==0) )).';
    a_hat_P(1:K,iii) = a_fshift_P(1:K,iii);
    r_hat_P(1:K,iii) = b_hat_P(:,iii).*a_hat_P(:,iii);
end

x_hat_T = [zeros(q/2-qc,1); conv(f_hat_T(q/2+(-qc:qc-1)),r_hat_T); zeros(q/2-qc,1)];
x_hat_T = x_hat_T(q/2+(1:K)) .* non_QRS;
x_hat_P = [zeros(q/2-qc,1); conv(f_hat_P(q/2+(-qc:qc-1)),r_hat_P); zeros(q/2-qc,1)];
x_hat_P = x_hat_P(q/2+(1:K)) .* non_QRS;

% signal reconstruction
%figure, plot(1:K, x, 'g', 1:K, M_T * gamma_T_hat + M_P * gamma_P_hat, '--b', 1:K, M2 * gamma2_hat, '--r',1:K , M3 * gamma3_hat, '--k')
%figure, plot(1:K, x ,'--k', 1:K, x_hat_T + x_hat_P +  M_T * gamma_T_hat + M_P * gamma_P_hat + M2 * gamma2_hat + M3 * gamma3_hat, 'r',...
%    1:K, M_T * gamma_T_hat + M_P * gamma_P_hat + M2 * gamma2_hat + M3 * gamma3_hat, 'b'), title('signal reconstruction inside'),legend('target signal','estimated P and T-waves','estimated baseline') 
baseline_total = M3 * gamma3_hat;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     END DETECTOR/ESTIMATOR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%