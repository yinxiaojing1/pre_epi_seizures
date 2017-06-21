function varargout = test(varargin)
% TEST M-file for test.fig
% The main GUI for the Bayesian P and T wave delineation and waveform
% estimation Toolbox
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
% Brandon Kuczenski for providing vline.m
% (C) C. LIN 2011
% TeSA lab, University of Toulouse, France

% Version 0.3, Nov. 2011
% Modifications: 
% (1) Wave delination with curvatures
% (2) Optimization of the function Gibbs_sampler

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @test_OpeningFcn, ...
    'gui_OutputFcn',  @test_OutputFcn, ...
    'gui_LayoutFcn',  [] , ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

% --- Executes just before test is made visible.
function test_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to test (see VARARGIN)

% Choose default command line output for test
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% This sets up the initial plot - only do when we are invisible
% so window can get raised using test.
% if strcmp(get(hObject,'Visible'),'off')
%    % plot(rand(5));
% end
% Custom initializations
%clc
%oldstr = get(handles.listbox2,'string'); % The string as it is now.
addstr = {'Welcome!'}; % The string to add to the stack.
set(handles.listbox2,'str',addstr);  % Put the new string on top


% UIWAIT makes test wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = test_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global target_signal_original target_signal_total
global T_peak_total P_peak_total T_onset_total P_onset_total T_end_total P_end_total

Fs = get(handles.edit3,'string');   % Signal processing window length
Fs = str2num(Fs);

axes(handles.axes1);
cla;

popup_sel_index = get(handles.popupmenu1, 'Value');
switch popup_sel_index
    case 1
        plot(1/Fs:1/Fs:length(target_signal_original)/Fs,target_signal_original,'b','LineWidth',2);
    case 2
        plot(1/Fs:1/Fs:length(target_signal_total)/Fs,target_signal_total,'b','LineWidth',2);
    case 3
        axes(handles.axes1);
        cla;
        hold on
        plot(1/Fs:1/Fs:length(target_signal_total)/Fs,target_signal_total,'b','LineWidth',2);
        plot(T_peak_total/Fs,target_signal_total(T_peak_total),'o','MarkerEdgeColor','k','MarkerFaceColor',[.49 1 .63],'MarkerSize',7)
        plot(T_onset_total/Fs,target_signal_total(T_onset_total),'o','MarkerEdgeColor','k','MarkerFaceColor',[.49 1 .63],'MarkerSize',7)
        plot(T_end_total/Fs,target_signal_total(T_end_total),'o','MarkerEdgeColor','k','MarkerFaceColor',[.49 1 .63],'MarkerSize',7)
        plot(P_peak_total/Fs,target_signal_total(P_peak_total),'o','MarkerEdgeColor','k','MarkerFaceColor',[.49 1 .63],'MarkerSize',7)
        plot(P_onset_total/Fs,target_signal_total(P_onset_total),'o','MarkerEdgeColor','k','MarkerFaceColor',[.49 1 .63],'MarkerSize',7)
        plot(P_end_total/Fs,target_signal_total(P_end_total),'o','MarkerEdgeColor','k','MarkerFaceColor',[.49 1 .63],'MarkerSize',7)
        hold off
end


% --------------------------------------------------------------------
function FileMenu_Callback(hObject, eventdata, handles)
% hObject    handle to FileMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% --------------------------------------------------------------------
function OpenMenuItem_Callback(hObject, eventdata, handles)
% hObject    handle to OpenMenuItem (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% --------------------------------------------------------------------
function PrintMenuItem_Callback(hObject, eventdata, handles)
% hObject    handle to PrintMenuItem (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
printdlg(handles.figure1)

% --------------------------------------------------------------------
function CloseMenuItem_Callback(hObject, eventdata, handles)
% hObject    handle to CloseMenuItem (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
selection = questdlg(['Close ' get(handles.figure1,'Name') '?'],...
    ['Close ' get(handles.figure1,'Name') '...'],...
    'Yes','No','Yes');
if strcmp(selection,'No')
    return;
end

delete(handles.figure1)


% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns popupmenu1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu1


% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

set(hObject, 'String', {'original ECG signal', 'preprocessed ECG signal','delineation results'});


% --- Executes on selection change in listbox2.
function listbox2_Callback(hObject, eventdata, handles)
% hObject    handle to listbox2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns listbox2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox2


% --- Executes during object creation, after setting all properties.
function listbox2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global target_signal_original target_signal_total
global T_peak_total P_peak_total T_onset_total P_onset_total T_end_total P_end_total
% baseline removal, QRS detection
% =============
D_str = get(handles.edit4,'string'); % processing window length
D = str2num(D_str);
Fs = get(handles.edit3,'string');   % Signal processing window length
Fs = str2num(Fs);

popup_sel_index = get(handles.popupmenu2, 'Value');
switch popup_sel_index
    case 2
        [ecgout b1] = BaseLineTOS(target_signal_original', [], Fs, round(Fs/200));  % 3 order Spline
        target_signal = ecgout;
        target_signal_show = ecgout;
    case 1
        b2 = BaseLine1(target_signal_original,Fs*.3,'md');    % median filtering
        target_signal = target_signal_original-b2;
        target_signal_show = target_signal_original-b2;
    case 3
        b3 = BPFilter(target_signal_original,0,.7/Fs);  % Band pass filter
        target_signal = target_signal_original-b3;
        target_signal_show = target_signal_original-b3;
end
target_signal = target_signal/max(abs(target_signal));
oldstr = get(handles.listbox2,'string'); % The string as it is now.
addstr = {'preprocessing done!'}; % The string to add to the stack.
set(handles.listbox2,'str',{addstr{:},oldstr{:}});  % Put the new string on top

% QRS Complex detection, low-pass filtering
% =============
[R_loc_total,QRS_width,Q_loc_total,S_loc_total,target_signal_filtered]=QRSdetection(target_signal, Fs);
Ndeb = 5;
target_signal_total = target_signal_filtered(Ndeb+1:end);
total_beat = length(R_loc_total);
total_process = floor(total_beat/(D-1));
Q_loc_1_total = Q_loc_total(2:end);
S_loc_2_total = S_loc_total(1:end-1);
QS_interval = Q_loc_1_total - S_loc_2_total;
moy_QS_interval_N = round(mean(QS_interval)/3.5);
if mod(moy_QS_interval_N,2)
    moy_QS_interval_N = moy_QS_interval_N+1;
end
oldstr = get(handles.listbox2,'string'); % The string as it is now.
addstr = {'QRS detection done!'}; % The string to add to the stack.
set(handles.listbox2,'str',{addstr{:},oldstr{:}});  % Put the new string on top

% creat buffers to save overall delineation results
% =============
T_hat_total = zeros(total_process,moy_QS_interval_N);
P_hat_total = zeros(total_process,moy_QS_interval_N);
T_peak_total = [];
P_peak_total = [];
T_onset_total = [];
P_onset_total = [];
T_end_total = [];
P_end_total = [];

% =============
% =============
% Main processing loop
for processing_ind = 1:total_process
    oldstr = get(handles.listbox2,'string'); % The string as it is now.
    addstr = {['Processing block index No. ' num2str(processing_ind)]}; % The string to add to the stack.
    set(handles.listbox2,'str',{addstr{:},oldstr{:}});  % Put the new string on top
    % D-beat window construction
    % =============
    offset = (D-1)*(processing_ind-1);
    if offset+D <= length(Q_loc_total)
        target_signal = target_signal_total(Q_loc_total(offset+1):Q_loc_total(offset+D));
    else
        display('Warning: the ending beats are not processed.')
        break
    end
    K = length(target_signal);
    if mod(K,2)
        K = K+1;
    end
    R_loc = R_loc_total(offset+1:offset+D)-Q_loc_total(offset+1);
    S_loc = S_loc_total(offset+1:offset+D)-Q_loc_total(offset+1);
    Q_loc = Q_loc_total(offset+1:offset+D)-Q_loc_total(offset+1)+1;
    Q_loc_1 = Q_loc(2:end);
    S_loc_2 = S_loc(1:end-1);
    QS_interval = Q_loc_1 - S_loc_2;
    moy_QS_interval = round(QS_interval/2);
    
    % generat pure P and T-wave signal
    % =============
    ST = 15;  % QRS location adjustment
    T_onset_list = [];
    T_end_list = [];
    for i=1:length(S_loc_2)
        T_onset_list = [T_onset_list S_loc(i)+ST];
        T_end_list = [T_end_list S_loc(i)+moy_QS_interval(i)+ST-1];
    end
    PQ = 10;  % QRS location adjustment
    P_onset_list = [];
    P_end_list = [];
    P_onset_list = T_end_list + 1 ;
    for i=1:length(Q_loc_1)
        P_end_list = [P_end_list Q_loc_1(i)-PQ];
    end
    pure_signal = zeros(K,1);
    for i=1:length(P_onset_list)
        pure_signal(T_onset_list(i):P_end_list(i)) = target_signal(T_onset_list(i):P_end_list(i));
    end
    amp_factor = 5;
    pure_signal = pure_signal*amp_factor;
    
    oldstr = get(handles.listbox2,'string'); % The string as it is now.
    addstr = {'P and T research region extraction done!'}; % The string to add to the stack.
    set(handles.listbox2,'str',{addstr{:},oldstr{:}});  % Put the new string on top
    
    % T-wave and P-wave analysis using MCMC
    % =============
    % parameters
    N_T_wave = moy_QS_interval_N;    %
    N_P_wave = moy_QS_interval_N;    %
    pi_1 = 0.005;  % prior of Bernoulli-Gaussian
    sigma2_a = 0.1; % prior of amplitude
    sigma2_alpha = 0.001*1;  % prior of Hermits coefficients, smaller-->more sensitive, bigger-->noise resistence
    sigma2_gamma = 0.001*1;  % prior of local baseline
    sigma2_gamma2 = 0.001*1;
    eta = 0.5;               % prior of noise
    xi = 3;
    fshift = 0; %adjustment of frequency mismatch in real data
    Iterat_str = get(handles.edit5,'string'); % number of iterations of the sampler
    Iterat = str2num(Iterat_str);
    % Gibbs analysis
    oldstr = get(handles.listbox2,'string'); % The string as it is now.
    addstr = {'Wave detection...'}; % The string to add to the stack.
    set(handles.listbox2,'str',{addstr{:},oldstr{:}});  % Put the new string on top
    [b_hat_T,r_hat_T,f_hat_T,b_hat_P,r_hat_P,f_hat_P,baseline_total,x_hat_T,x_hat_P]=Gibbs_analyser(Iterat,pure_signal,N_T_wave,pi_1,sigma2_a,sigma2_alpha,sigma2_gamma,sigma2_gamma2,eta,xi,fshift,T_onset_list,T_end_list,P_onset_list,P_end_list,handles);
    oldstr = get(handles.listbox2,'string'); % The string as it is now.
    addstr = {'Wave detection done!'}; % The string to add to the stack.
    set(handles.listbox2,'str',{addstr{:},oldstr{:}});  % Put the new string on top
    inst_T_hat=b_hat_T;
    inst_P_hat=b_hat_P;
    T_hat=f_hat_T;
    P_hat=f_hat_P;
    T_loc = find(inst_T_hat)';
    P_loc = find(inst_P_hat)';
    
    % Wave delineation
    % =============
    
    switch findobj(get(handles.uipanel5,'selectedobject'))
        case handles.radiobutton4  % manual supervised waveform analysis
            T_hat = T_hat(length(T_hat)/2+(-moy_QS_interval_N/2+1:moy_QS_interval_N/2));
            [value,T_peak] = max(T_hat);
            xc=1:length(T_hat);
            yc=T_hat';
            axes(handles.axes2);
            cla;
            l=plot(xc,yc);
            hold on
            title('T waveform estimation')
            [x,y]=ginput(1);
            [a,b]=min((xc-x).^2+(yc-y).^2);
            T_limit_left = xc(b(1));
            vline(T_limit_left,'r','On set')
            [x,y]=ginput(1);
            [a,b]=min((xc-x).^2+(yc-y).^2);
            T_limit_right = xc(b(1));
            vline(T_limit_right,'r','Off set')
            hold off
            T_limit_left_dis_ref = abs(T_peak-T_limit_left);
            T_limit_right_dis_ref = abs(T_peak-T_limit_right);
            T_limit_dis_factor = nonzeros(inst_T_hat)/max(nonzeros(inst_T_hat));
            T_limit_left_dis = round(T_limit_left_dis_ref*T_limit_dis_factor)';
            T_limit_right_dis = round(T_limit_right_dis_ref*T_limit_dis_factor)';
            
            P_hat = P_hat(length(P_hat)/2+(-moy_QS_interval_N/2+1:moy_QS_interval_N/2));
            [value,P_peak] = max(P_hat);
            xc=1:length(P_hat);
            yc=P_hat';
            axes(handles.axes3);
            cla;
            l=plot(xc,yc);
            hold on
            title('P waveform estimation')
            [x,y]=ginput(1);
            [a,b]=min((xc-x).^2+(yc-y).^2);
            P_limit_left = xc(b(1));
            vline(P_limit_left,'r','On set')
            [x,y]=ginput(1);
            [a,b]=min((xc-x).^2+(yc-y).^2);
            P_limit_right = xc(b(1));
            vline(P_limit_right,'r','Off set')
            hold off
            P_limit_left_dis_ref = abs(P_peak-P_limit_left);
            P_limit_right_dis_ref = abs(P_peak-P_limit_right);
            P_limit_dis_factor = nonzeros(inst_P_hat)/max(nonzeros(inst_P_hat));
            P_limit_left_dis = round(P_limit_left_dis_ref*P_limit_dis_factor)';
            P_limit_right_dis = round(P_limit_right_dis_ref*P_limit_dis_factor)';
            
        case handles.radiobutton3  % automatic waveform analysis with local minima
            T_hat = T_hat(K/2-N_T_wave/2:K/2+N_T_wave/2);
            P_hat = P_hat(K/2-N_P_wave/2:K/2+N_P_wave/2);
            % T-wave
            T_hat_proc = T_hat(1:end);
            [value,T_peak] = max(T_hat_proc);
            threshold_left = get(handles.edit10,'string');
            threshold_left = str2num(threshold_left);
            threshold_right = get(handles.edit11,'string');
            threshold_right = str2num(threshold_right);
            % T-wave onset
            [T_min_left_value,T_min_left_loc] = min(T_hat_proc(1:T_peak));
            if T_min_left_value >= threshold_left
                T_limit_left_dis_ref = abs(T_peak-T_min_left_loc);
            else
                [T_nearest_left_value,T_nearest_left_loc] = min(abs(T_hat_proc(T_min_left_loc:T_peak)-threshold_left));
                T_limit_left_dis_ref = abs(T_peak-T_nearest_left_loc-T_min_left_loc);
            end
            % T-wave end
            [T_min_right_value,T_min_right_loc_relative] = min(T_hat_proc(T_peak+1:end));
            T_min_right_loc = T_min_right_loc_relative + T_peak;
            if T_min_right_value >= threshold_right
                T_limit_right_dis_ref = abs(T_min_right_loc-T_peak);
            else
                [T_nearest_right_value,T_nearest_right_loc] = min(abs(T_hat_proc(T_peak+1:T_min_right_loc)-threshold_right));
                T_limit_right_dis_ref = T_nearest_right_loc;
            end
            T_limit_left_dis = T_limit_left_dis_ref;
            T_limit_right_dis = T_limit_right_dis_ref;
            % P wave
            P_hat_proc = P_hat(1:end);
            [value,P_peak] = max(P_hat_proc);
            threshold_left = get(handles.edit12,'string');
            threshold_left = str2num(threshold_left);
            threshold_right = get(handles.edit13,'string');
            threshold_right = str2num(threshold_right);
            % P-wave onset
            [P_min_left_value,P_min_left_loc] = min(P_hat_proc(1:P_peak));
            if P_min_left_value >= threshold_left
                P_limit_left_dis_ref = abs(P_peak-P_min_left_loc);
            else
                [P_nearest_left_value,P_nearest_left_loc] = min(abs(P_hat_proc(P_min_left_loc:P_peak)-threshold_left));
                P_limit_left_dis_ref = abs(P_peak-P_nearest_left_loc-P_min_left_loc);
            end
            % P-wave end
            [P_min_right_value,P_min_right_loc_relative] = min(P_hat_proc(P_peak+1:end));
            P_min_right_loc = P_min_right_loc_relative + P_peak;
            if P_min_right_value >= threshold_right
                P_limit_right_dis_ref = abs(P_min_right_loc-P_peak);
            else
                [P_nearest_right_value,P_nearest_right_loc] = min(abs(P_hat_proc(P_peak+1:P_min_right_loc)-threshold_right));
                P_limit_right_dis_ref = P_nearest_right_loc;
            end
            P_limit_left_dis = P_limit_left_dis_ref;
            P_limit_right_dis = P_limit_right_dis_ref;
            
            xc=1:length(T_hat);
            yc=T_hat';
            axes(handles.axes2);
            cla;
            plot(xc,yc);
            hold on
            title('T waveform estimation')
            vline(T_peak-T_limit_left_dis,'r','On set')
            vline(T_peak+T_limit_right_dis,'r','Off set')
            hold off
            
            xc=1:length(P_hat);
            yc=P_hat';
            axes(handles.axes3);
            cla;
            plot(xc,yc);
            hold on
            title('P waveform estimation')
            vline(P_peak-P_limit_left_dis,'r','On set')
            vline(P_peak+P_limit_right_dis,'r','Off set')
            hold off
            
        case handles.radiobutton5  % automatic waveform analysis with curvature
            T_hat = T_hat(K/2-N_T_wave/2:K/2+N_T_wave/2);
            P_hat = P_hat(K/2-N_P_wave/2:K/2+N_P_wave/2);
            % T wave delineation
            T_hat_proc = T_hat(1:end);
            %[value,T_peak] = max(T_hat_proc);
            T_peak = round(length(T_hat_proc)/2);  % thanks to the constraint to avoid ambiguity
            deno=(sqrt(1+diff(T_hat_proc).^2).^3);
            curv=abs(diff(T_hat_proc,2))./deno(1:end-1);
            curv = curv';
            curv(1) = curv(2); % the first point is meaningless
            coef_b=fir1(10,0.3); % smoothing filter
            curv_s = conv(coef_b,curv);
            curv_s = curv_s(6:end);
            %local_maxima_list = find(curv>=[curv(2:end) inf] & curv>[inf curv(1:end-1)]);
            local_maxima_list = find(curv_s>=[curv_s(2:end) inf] & curv_s>[inf curv_s(1:end-1)]);
            forbidden_zone = round(length(T_hat_proc)/8);
            local_maxima_list_pos_left = find(local_maxima_list<=T_peak-forbidden_zone);
            if isempty(local_maxima_list_pos_left)   % in case that no local maxima of curvature is found
                T_limit_left_dis = round(length(T_hat_proc)/3);
            else
                [v,p]=max(curv_s(local_maxima_list(local_maxima_list_pos_left)));
                T_limit_left_curv = local_maxima_list(local_maxima_list_pos_left(p));   % the most significant local maxima on the left
               % T_limit_left_curv =  local_maxima_list(local_maxima_list_pos_left(end));  % the nearest local maxima on the left
                T_limit_left_dis = abs(T_peak - T_limit_left_curv) ;
            end
            local_maxima_list_pos_right = find(local_maxima_list>T_peak+forbidden_zone);
            if isempty(local_maxima_list_pos_right)   % in case that no local maxima of curvature is found
                T_limit_right_dis = round(length(T_hat_proc)/3);
            else
                [v,p]=max(curv_s(local_maxima_list(local_maxima_list_pos_right)));
                T_limit_right_curv = local_maxima_list(local_maxima_list_pos_right(p));   % the most significant local maxima on the right
               % T_limit_right_curv = local_maxima_list(local_maxima_list_pos_right(1));    % the nearest local maxima on the right
                T_limit_right_dis = abs(T_limit_right_curv - T_peak) ;               
            end
            
            % P wave delineation
            P_hat_proc = P_hat(1:end);
            %[value,P_peak] = max(P_hat_proc);
            P_peak = round(length(P_hat_proc)/2);  % thanks to the constraint to avoid ambiguity
            deno=(sqrt(1+diff(P_hat_proc).^2).^3);
            curv=abs(diff(P_hat_proc,2))./deno(1:end-1);
            curv = curv';
            curv(1) = curv(2); % the first point is meaningless
            coef_b=fir1(10,0.3); % smoothing filter
            curv_s = conv(coef_b,curv);
            curv_s = curv_s(5:end);
            local_maxima_list = find(curv_s>=[curv_s(2:end) inf] & curv_s>[inf curv_s(1:end-1)]);
            forbidden_zone = round(length(P_hat_proc)/8);
            local_maxima_list_pos_left = find(local_maxima_list<=P_peak-forbidden_zone);
            if isempty(local_maxima_list_pos_left)   % in case that no local maxima of curvature is found
                P_limit_left_dis = round(length(P_hat_proc)/3);
            else
                [v,p]=max(curv_s(local_maxima_list(local_maxima_list_pos_left)));
                P_limit_left_curv = local_maxima_list(local_maxima_list_pos_left(p));   % the most significant local maxima on the left
              %  P_limit_left_curv = local_maxima_list(local_maxima_list_pos_left(end)); % the nearest local maxima on the left
                P_limit_left_dis = abs(P_peak - P_limit_left_curv);
            end
            local_maxima_list_pos_right = find(local_maxima_list>P_peak+forbidden_zone);
            if isempty(local_maxima_list_pos_right)   % in case that no local maxima of curvature is found
                P_limit_right_dis = round(length(P_hat_proc)/3);
            else
                [v,p]=max(curv_s(local_maxima_list(local_maxima_list_pos_right)));
                P_limit_right_curv = local_maxima_list(local_maxima_list_pos_right(p));   % the most significant local maxima on the left
             %   P_limit_right_curv = local_maxima_list(local_maxima_list_pos_right(1));    % the nearest local maxima on the right
                P_limit_right_dis = abs(P_limit_right_curv - P_peak);
            end
            
            xc=1:length(T_hat);
            yc=T_hat';
            axes(handles.axes2);
            cla;
            plot(xc,yc);
            hold on
            title('T waveform estimation')
            vline(T_peak-T_limit_left_dis,'r','On set')
            vline(T_peak+T_limit_right_dis,'r','Off set')
            hold off
            
            xc=1:length(P_hat);
            yc=P_hat';
            axes(handles.axes3);
            cla;
            plot(xc,yc);
            hold on
            title('P waveform estimation')
            vline(P_peak-P_limit_left_dis,'r','On set')
            vline(P_peak+P_limit_right_dis,'r','Off set')
            hold off
            
        otherwise
            set(S.ed,'string','None!') % Very unlikely I think.
    end
    % Overall results saving
    T_hat_total(processing_ind,:)= T_hat(1:moy_QS_interval_N)'/max(T_hat);
    P_hat_total(processing_ind,:)= P_hat(1:moy_QS_interval_N)'/max(P_hat);
    T_peak_total = [T_peak_total Q_loc_total(offset+1)+T_loc];
    P_peak_total = [P_peak_total Q_loc_total(offset+1)+P_loc];
    T_onset_total = [T_onset_total Q_loc_total(offset+1)+T_loc-T_limit_left_dis];
    P_onset_total = [P_onset_total Q_loc_total(offset+1)+P_loc-P_limit_left_dis];
    T_end_total = [T_end_total Q_loc_total(offset+1)+T_loc+T_limit_right_dis];
    P_end_total = [P_end_total Q_loc_total(offset+1)+P_loc+P_limit_right_dis];
end  % the end of the test processing loop
axes(handles.axes1);
cla;
hold on
plot(1/Fs:1/Fs:length(target_signal_total)/Fs,target_signal_total,'b','LineWidth',2);
plot(T_peak_total/Fs,target_signal_total(T_peak_total),'o','MarkerEdgeColor','k','MarkerFaceColor',[.49 1 .63],'MarkerSize',7)
plot(T_onset_total/Fs,target_signal_total(T_onset_total),'o','MarkerEdgeColor','k','MarkerFaceColor',[.49 1 .63],'MarkerSize',7)
plot(T_end_total/Fs,target_signal_total(T_end_total),'o','MarkerEdgeColor','k','MarkerFaceColor',[.49 1 .63],'MarkerSize',7)
plot(P_peak_total/Fs,target_signal_total(P_peak_total),'o','MarkerEdgeColor','k','MarkerFaceColor',[.49 1 .63],'MarkerSize',7)
plot(P_onset_total/Fs,target_signal_total(P_onset_total),'o','MarkerEdgeColor','k','MarkerFaceColor',[.49 1 .63],'MarkerSize',7)
plot(P_end_total/Fs,target_signal_total(P_end_total),'o','MarkerEdgeColor','k','MarkerFaceColor',[.49 1 .63],'MarkerSize',7)
hold off
h = pan;
set(h,'Motion','horizontal','Enable','on'); % pan on the plot in the horizontal direction.


function edit3_Callback(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit3 as text
%        str2double(get(hObject,'String')) returns contents of edit3 as a double


% --- Executes during object creation, after setting all properties.
function edit3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.edit3,'string','250');   % Fs
set(handles.edit4,'string','5');   % processing window length
set(handles.edit5,'string','80');   % iteration
set(handles.popupmenu2, 'value',1);   % baseline removal technique
set(handles.radiobutton4,'value',0);
set(handles.radiobutton3,'value',1);


function edit4_Callback(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit4 as text
%        str2double(get(hObject,'String')) returns contents of edit4 as a double


% --- Executes during object creation, after setting all properties.
function edit4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in popupmenu2.
function popupmenu2_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns popupmenu2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from
%        popupmenu2


% --- Executes during object creation, after setting all properties.
function popupmenu2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit5_Callback(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit5 as text
%        str2double(get(hObject,'String')) returns contents of edit5 as a double


% --- Executes during object creation, after setting all properties.
function edit5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit6_Callback(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit6 as text
%        str2double(get(hObject,'String')) returns contents of edit6 as a double


% --- Executes during object creation, after setting all properties.
function edit6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global target_signal_original
Fs = get(handles.edit3,'string');   % Signal processing window length
Fs = str2num(Fs);

[nomfichier,PathName] = uigetfile('*.mat','Choose the file ...'); % load the file
load([PathName nomfichier]);
%target_signal_original = ecg';
target_signal_original = ECG_1';  % choose one lead
oldstr = get(handles.listbox2,'string'); % The string as it is now.
addstr = {[ nomfichier,' successfully loaded!']}; % The string to add to the stack.
set(handles.listbox2,'str',{addstr{:},oldstr{:}});  % Put the new string on top
set(handles.edit6,'string',nomfichier);
axes(handles.axes1);
cla;
plot(1/Fs:1/Fs:length(target_signal_original)/Fs,target_signal_original,'b','LineWidth',2);
set(handles.slider1,'max',length(target_signal_original)/Fs+1)

function edit7_Callback(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit7 as text
%        str2double(get(hObject,'String')) returns contents of edit7 as a double


% --- Executes during object creation, after setting all properties.
function edit7_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on slider movement.
function slider1_Callback(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
global target_signal_original
% Fs_str = get(handles.edit3,'string');   % Signal processing window length
% Fs = str2num(Fs_str);
ax = handles.axes1;
set(ax,'xlim',[-0.00001 get(handles.slider1,'val')])



% --- Executes during object creation, after setting all properties.
function slider1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on button press in radiobutton3.
function radiobutton3_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton3


% --- Executes on button press in radiobutton4.
function radiobutton4_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton4



function edit10_Callback(hObject, eventdata, handles)
% hObject    handle to edit10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit10 as text
%        str2double(get(hObject,'String')) returns contents of edit10 as a double


% --- Executes during object creation, after setting all properties.
function edit10_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit11_Callback(hObject, eventdata, handles)
% hObject    handle to edit11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit11 as text
%        str2double(get(hObject,'String')) returns contents of edit11 as a double


% --- Executes during object creation, after setting all properties.
function edit11_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit12_Callback(hObject, eventdata, handles)
% hObject    handle to edit12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit12 as text
%        str2double(get(hObject,'String')) returns contents of edit12 as a double


% --- Executes during object creation, after setting all properties.
function edit12_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit13_Callback(hObject, eventdata, handles)
% hObject    handle to edit13 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit13 as text
%        str2double(get(hObject,'String')) returns contents of edit13 as a double


% --- Executes during object creation, after setting all properties.
function edit13_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit13 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function editLEDbkground_Callback(hObject, eventdata, handles)
% hObject    handle to editLEDbkground (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of editLEDbkground as text
%        str2double(get(hObject,'String')) returns contents of editLEDbkground as a double


% --- Executes during object creation, after setting all properties.
function editLEDbkground_CreateFcn(hObject, eventdata, handles)
% hObject    handle to editLEDbkground (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function textLED1_Callback(hObject, eventdata, handles)
% hObject    handle to textLED1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of textLED1 as text
%        str2double(get(hObject,'String')) returns contents of textLED1 as a double


% --- Executes during object creation, after setting all properties.
function textLED1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to textLED1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function textLED10_Callback(hObject, eventdata, handles)
% hObject    handle to textLED10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of textLED10 as text
%        str2double(get(hObject,'String')) returns contents of textLED10 as a double


% --- Executes during object creation, after setting all properties.
function textLED10_CreateFcn(hObject, eventdata, handles)
% hObject    handle to textLED10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function textLED3_Callback(hObject, eventdata, handles)
% hObject    handle to textLED3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of textLED3 as text
%        str2double(get(hObject,'String')) returns contents of textLED3 as a double


% --- Executes during object creation, after setting all properties.
function textLED3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to textLED3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function textLED4_Callback(hObject, eventdata, handles)
% hObject    handle to textLED4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of textLED4 as text
%        str2double(get(hObject,'String')) returns contents of textLED4 as a double


% --- Executes during object creation, after setting all properties.
function textLED4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to textLED4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function textLED5_Callback(hObject, eventdata, handles)
% hObject    handle to textLED5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of textLED5 as text
%        str2double(get(hObject,'String')) returns contents of textLED5 as a double


% --- Executes during object creation, after setting all properties.
function textLED5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to textLED5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit19_Callback(hObject, eventdata, handles)
% hObject    handle to edit19 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit19 as text
%        str2double(get(hObject,'String')) returns contents of edit19 as a double


% --- Executes during object creation, after setting all properties.
function edit19_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit19 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function textLED6_Callback(hObject, eventdata, handles)
% hObject    handle to textLED6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of textLED6 as text
%        str2double(get(hObject,'String')) returns contents of textLED6 as a double


% --- Executes during object creation, after setting all properties.
function textLED6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to textLED6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function textLED7_Callback(hObject, eventdata, handles)
% hObject    handle to textLED7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of textLED7 as text
%        str2double(get(hObject,'String')) returns contents of textLED7 as a double


% --- Executes during object creation, after setting all properties.
function textLED7_CreateFcn(hObject, eventdata, handles)
% hObject    handle to textLED7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function textLED8_Callback(hObject, eventdata, handles)
% hObject    handle to textLED8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of textLED8 as text
%        str2double(get(hObject,'String')) returns contents of textLED8 as a double


% --- Executes during object creation, after setting all properties.
function textLED8_CreateFcn(hObject, eventdata, handles)
% hObject    handle to textLED8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function textLED9_Callback(hObject, eventdata, handles)
% hObject    handle to textLED9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of textLED9 as text
%        str2double(get(hObject,'String')) returns contents of textLED9 as a double


% --- Executes during object creation, after setting all properties.
function textLED9_CreateFcn(hObject, eventdata, handles)
% hObject    handle to textLED9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function textLED2_Callback(hObject, eventdata, handles)
% hObject    handle to textLED2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of textLED2 as text
%        str2double(get(hObject,'String')) returns contents of textLED2 as a double


% --- Executes during object creation, after setting all properties.
function textLED2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to textLED2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function textLEDpercentDone_Callback(hObject, eventdata, handles)
% hObject    handle to textLEDpercentDone (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of textLEDpercentDone as text
%        str2double(get(hObject,'String')) returns contents of textLEDpercentDone as a double


% --- Executes during object creation, after setting all properties.
function textLEDpercentDone_CreateFcn(hObject, eventdata, handles)
% hObject    handle to textLEDpercentDone (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
