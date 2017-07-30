# @profile
def main(arg):

    # _logger.debug("Starting Gaussian Fit...")
    # _logger.debug("vlfsn")

# #*****************phisionet data*******************************
    # path = '~/Desktop/phisionet_dataset.h5'
    # name = ['sz_'+str(arg)]
    # group = ['medianFIR', 'raw']

#     # # baseline_removal(path, name, 'raw')
#     # # create_rpeak_dataset(path, name, group)
    # X = load_signal(path, name, group)

#     # path = '~/Desktop/dummy.h5'
#     # # rpeaks = load_signal(path,'rpeaks_'+name, group)

#     _logger.debug(X[0]['signal'])
#     _logger.debug(X[0]['mdata'])

    # heart_beat, rpeak = create_heart_beat_dataset(path=path,
    #                                               name=name,
    #                                               group=group,
    #                                               save_dfile=None)

    # _logger.debug(np.asmatrix(rpeaks['signal'].T))
    # models = [X['signal'][40*200:60*200], X_raw['signal'][40*200:60*200]]
    # names = ['Filtered', 'Raw' ]
    # colors = ['red']
    # plot_models(models, names, colors)

#****************************************************************
    path = '~/Desktop/HSM_data.h5'
    name = ['FA77748T']
    group = ['/PATIENT1/free']

    # raw = load_signal(path, name, group)

    # stop
    # begin_seizure_seconds = raw[0]['mdata']['crysis_time_seconds']
    # begin_seizure_sample = int(1000*begin_seizure_seconds[0])

    # sampling_rate_hertz = 1000

    # signal = raw[0]['signal']
    # no_seizure_ecg_raw = signal[0:begin_seizure_sample,0]

    # ecg_10min_5min_raw = signal[begin_seizure_sample
    #                              -sampling_rate_hertz*10*60:
    #                              begin_seizure_sample
    #                              + sampling_rate_hertz*5*60]


    path_to_save = '~/Desktop/seizures_datasets_new.h5'
    # mdata_list = [raw[0]['mdata']]
    # signal_list = [ecg_10min_5min_raw]
    name_list = ['10_15', '10_15', 'rpeaks_10_15']
    group_list= ['raw', 'medianFIR', 'medianFIR']

    baseline_removal(path_to_save, name_list,0 group_list)
    create_rpeak_dataset(path_to_save, name_list[0], group_list[1])
    # save_signal(path_to_save, signal_list,
                # mdata_list, name_list, group_list)
    # ecg_10min_5min_raw = X[0]['signal']
    # ecg_10min_5min_medianFIR = X[1]['signal']


    # create_rpeak_dataset(path_to_save, name, group)
    # _logger.debug(begin_seizure_sample)

    print zip(group_list, name_list)

    X = load_signal(path_to_save, zip(group_list, name_list))
    
    _logger.debug(X)

    signal_raw_model = X[0]['signal'].T
    signal_model = X[1]['signal'].T
    rpeaks_model = X[2]['signal'].T
    signal = signal_model[0,:]
    rpeaks = rpeaks_model[0,:]

    # _logger.debug(rpeaks)
    # _logger.debug(signal)


    # phase = get_phase(signal[0,:], rpeaks[0,:])
    # _logger.debug(phase)
    # # mean_extraction(signal, phase)

    # baseline_removal(path_to_save, name_list, group_list)
    # create_rpeak_dataset(path_to_save, name_list[0], group_list[1])

    # decimated = sp.signal.decimate(raw[0]['signal'], 2)
    # stop
    

    # baseline_removal(path_to_save, ['10_15'], ['raw'])

    Fs = X[1]['mdata']['sample_rate']
    N = len(signal_model[0,:])
    T = (N - 1) / Fs

    time = np.linspace(0, T, N, endpoint=False)
    time_rpeaks = time[rpeaks]

    _logger.debug(len(time))
    _logger.debug(N)

    time_model = np.array([time])
    time_rpeaks_model = np.array([time_rpeaks])
    signal_rpeaks = signal[rpeaks]
    signal_rpeaks_model = np.array([signal_rpeaks])
    # _logger.debug('time %s', time)
    # _logger.debug('time %s', signal)

    signal = filter_signal(signal=signal, ftype='FIR', band='lowpass',
                  order=100, frequency=40,
                  sampling_rate=Fs)

    signal_model = shape_array(signal)

    _logger.debug(signal_model)

    start = 400
    end = 405
    # time_rpeaks_models = [time_rpeaks_model.T]
    # rpeaks_models = [signal_rpeaks_model.T]
    times_models = [time_model.T, time_model.T]
    models = [signal_model, signal_raw_model.T]
    names = ['medianFIR', 'raw']
    colors = ['red']
    # plot_models_scatter(time_rpeaks_models, rpeaks_models,
    #                     times_models, models, names,
    #                     colors, start, end)
    plot_models(times_models, models, names, colors, start, end)
    fft_plot(times_models, models, names, colors)

    plt.show()
