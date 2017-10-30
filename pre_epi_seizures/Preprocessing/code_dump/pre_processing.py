




    win_params
    baseline_removal_name = raw_dataset_name + get_str_from_params() 

    raw_dataset_name = dataset_name + '/' + raw_name
    baseline_removal_dataset_name = raw_dataset_name + '/' + baseline_removal_name
    decimated_dataset_name = baseline_removal_dataset_name + '/' + 'decimation'
    eks_dataset_name = decimated_dataset_name + '/' + 'eks_smoothing'
    interpolated_dataset_name = eks_dataset_name + '/' + 'interpolation'
    # group_name_list = list_group_signals(path_to_load, group_list[0])['signals']
    # compress(path_to_load, group_name_list)

    # Load existing features --------------------------------------
    # raw = load_feature(path_to_load, raw_name, files='existent', feature_group_to_process=dataset_name)[0]


    # Extract the Features -----------------------------------------
    # rpeaks = load_feature(path_to_load, 'rpeak_detection', files='all_new', feature_group_to_process=raw_dataset_name,
                          # param_method='hamilton')

    baseline_removal = load_feature(path_to_load, baseline_removal_name, files='all_new', feature_group_to_process=raw_dataset_name)

    # decimated = load_feature(path_to_load, 'decimation', files='existent', feature_group_to_process=baseline_removal_dataset_name)
    # rpeaks = load_feature(path_to_load, 'rpeak_detection', files='all_new', feature_group_to_process=baseline_removal_dataset_name)

    hrv = load_feature(path_to_load, 'hrv_computation', files='all_new', feature_group_to_process=baseline_removal_dataset_name, rpeak_group_to_process=baseline_removal_dataset_name + '/' + 'rpeak_detection', window=10)

    # hrv_time_domain = load_feature(path_to_load,
                                   # 'hrv_time_domain_features',
                                   # files='existent',
                                   # feature_group_to_process=baseline_removal_dataset_name + '/' + 'hrv_computation_w:10',
                                   # window=10)








# QRS_fixed = load_feature(path_to_load, 'QRS_fixed_segmentation', files='all_new', feature_group_to_process=baseline_removal_dataset_name, rpeak_group_to_process=baseline_removal_dataset_name + '/' + 'rpeak_detection')

    # eks = load_feature(path_to_load, 'eks_smoothing', files='existent', feature_group_to_process=baseline_removal_dataset_name + '/' + 'decimation', rpeak_group_to_process=baseline_removal_dataset_name + '/' + 'decimation' + '/' + 'rpeak_detection')[0]
    # # stop
    # # time_array_to_interpolate = np.linspace(0, 40*60 - 1.0/500, 40*60*500)
    # # print time_array_to_interpolate
    # interpolated = load_feature(path_to_load, 'interpolation', sampling_rate=500, files='existent', feature_group_to_process=eks_dataset_name)[0]
    # rpeaks = load_feature(path_to_load, 'rpeak_detection', files='existent', feature_group_to_process=interpolated_dataset_name)[0]
    # hrv = load_feature(path_to_load, 'hrv_computation',
    #                    files='all_new',
    #                    feature_group_to_process=interpolated_dataset_name,
    #                    rpeak_group_to_process=interpolated_dataset_name + '/' + 'rpeak_detection',
                       
                       
    # # beat = load_feature(path_to_load, 'beat_phase_segmentation', files='existent', feature_group_to_process=interpolated_dataset_name, rpeak_group_to_process=interpolated_dataset_name + '/' + 'rpeak_detection')[0]
    # pca = load_feature(path_to_load, 'pca_beat_amp_computation',
    #                    files='all_new',
    #                    feature_group_to_process=baseline_removal_dataset_name + '/' + 'QRS_fixed_segmentation',
    #                    window=baseline_removal_dataset_name + '/rpeaks',
    #                    begin=5)[0]

    # # sameni = load_feature(path_to_load, 'sameni_evolution', files='all_new', feature_group_to_process=interpolated_dataset_name + '/' + 'beat_phase_segmentation')[0]
    # # rqa = load_feature(path_to_load, 'rqa_computation', files='all_new', feature_group_to_process=interpolated_dataset_name + '/' + 'QRS_fixed_segmentation')[0]
    # # stop
    # # print rqa
    # # stop

    # # ploting
    # # print sameni
    # # stop
    # start = 10*60
    # end = start + 10 
    # sz_nr = 0
    # signal = raw
    stop
    print signal
    # stop
    # signal_t = hrv
    # n = np.linspace(0, (len(signal[sz_nr])-1)/1000, len(signal[sz_nr]))
    # n_t = np.linspace(0, (len(signal_t[sz_nr])-1)/1000, len(signal_t[sz_nr]))

    plt.figure()
    # plt.subplot(1,2,1)
    # plt.title('interpolated ECG')
    plt.plot(signal[sz_nr])
    # plt.plot(n[rpeaks[sz_nr]], signal[sz_nr][rpeaks[sz_nr]], 'o', color='g')
    plt.xlim([start*sampling_rate, end*sampling_rate])
    # plt.xlabel('time[s]')
    # plt.subplot(1,2,2)
    # plt.title('Detrended and Denoised ECG')
    # plt.plot(signal_t[sz_nr])
    # # plt.xlim([start, end])
    # plt.xlabel('time[s]')
    plt.savefig(raw_name + '.png')
    stop
    # stop
    # #phase 
    # signal = signal[sz_nr]

    # features = ecg.ecg(signal=signal, sampling_rate=1000.0, show=True)

    # # print len(idx_up

    # beats = zip(idx_up[0:-1], idx_up[1:])

    # data = [signal[i:f] for i,f in beats]
    # print len(data)
    # print len(rpeaks)


    # sample = len(data) - 1
    # plt.plot(data[sample]*0.05)
    # plt.plot(phase[beats[sample][0]:beats[sample][1]])
    # # plt.xlim([start*1000, end*1000])
    # plt.show()

    # fig_phase = plt.figure()t
    # phase = get_phase(interpolated[sz_nr], rpeaks[sz_nr])
    # print phase[0]


    # Sameni parameters Evaluation 
    # bins = len(x)
    #     rloc = int(bins/2) # r is assumed to be at the center
    #     thetai = np.zeros(5) # phase loc
    #     thetai[0] = phase[int(.2*bins)+np.argmax(x[int(.2*bins):int(.45*bins)])]
    #     idx = int(.44*bins) + np.argmin(x[int(.44*bins):int(.5*bins)])
    #     thetai[1] = phase[idx]
    #     thetai[2] = phase[rloc]
    #     thetai[3] = phase[2*rloc - idx]
    #     thetai[4] = phase[int(5/8*bins)+np.argmax(x[int(5/8*bins):int(7/8*bins)])]
    #     bi = np.array([.1, .05, .05, .05, .2]) # width
    #     ai = np.zeros(5) # amplitude
    #     ai[0] = np.abs(np.max(x[int(.2*bins):int(.45*bins)]))
    #     ai[1] = -np.abs(np.min(x))
    #     ai[2] = np.abs(np.max(x))
    #     ai[3] = -np.abs(np.min(x))
    #     ai[4] = np.abs(np.max(x[int(5/8*bins):int(7/8*bins)]))
    #     values0 = np.hstack((ai, bi, thetai))
    # parameters = sameni[sz_nr]
    # plt.figure()
    # plt.title('T-wave Amplitude/Width/Phase')
    # plt.subplot(3,1,1)
    # plt.plot(parameters[:, 4])
    # plt.subplot(3,1,2)
    # plt.plot(parameters[:, 9])
    # plt.subplot(3,1,3)
    # plt.plot(parameters[:, 14])
    # plt.show()
    # plt.title('R-peaks Amplitude/Width/Phase')
    # plt.subplot(3,1,1)
    # plt.plot(parameters[:, 2])
    # plt.subplot(3,1,2)
    # plt.plot(parameters[:, 7])
    # plt.subplot(3,1,3)
    # plt.plot(parameters[:, 12])
    # plt.show()
    # plt.title('P-wave Amplitude/Width/Phase')
    # plt.subplot(3,1,1)
    # plt.plot(parameters[:, 0])
    # plt.subplot(3,1,2)
    # plt.plot(parameters[:, 5])
    # plt.subplot(3,1,3)
    # plt.plot(parameters[:, 10])
    # plt.show()
    # plt.title('Q-wave Amplitude/Width/Phase')
    # plt.subplot(3,1,1)
    # plt.plot(parameters[:, 1])
    # plt.subplot(3,1,2)
    # plt.plot(parameters[:, 6])
    # plt.subplot(3,1,3)
    # plt.plot(parameters[:, 11])
    # plt.show()

