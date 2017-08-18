    # Smoothing --dummy
    # start = 10
    # end = 20
    # decimation_factor = 4
    # print end

    # one_min_before = records_baseline_removal[:, start * sampling_rate : end * sampling_rate]
    # print one_min_before
    # one_min_before = sp.signal.decimate(one_min_before, decimation_factor)
    # print one_min_before

    # rpeaks = map(functools.partial(detect_rpeaks,
    #             sampling_rate=sampling_rate), one_min_before)

    # # x=(time_before_seizure * 60 * sampling_rate - start) / decimation_factor

    # # print np.shape(rpeaks)
    # # g = lambda a: a/4
    # # rpeaks = np.asarray(map(g, rpeaks_noise_removal))

    # # print rpeaks
    # # rpeaks_one_minute_before = find_rpeaks(rpeaks, start * sampling_rate,
    # #     end * sampling_rate) - start * sampling_rate
    # # rpeaks_one_minute_before = rpeaks_one_minute_before/decimation_factor

    # # print rpeaks

    # # print 'KALMAN ...'
    # tmp = time.time()
    # filtered = EKSmoothing(one_min_before, rpeaks,
    #     fs=sampling_rate, bins=250, verbose=False, 
    #     oset=False, savefolder=None)
    # s = time.time() - tmp

    # print s, 
    # print ' seconds'

    # print np.shape(filtered)

    # # plt.plot(filtered[0])
    # # plt.plot(x=x, color='g')
    # # plt.show()


    # Smoothing --final
    start = 0
    end = 1800+600
    decimation_factor = 4
    print end

    one_min_before = records_baseline_removal[:, start * sampling_rate : end * sampling_rate]
    print one_min_before
    one_min_before = sp.signal.decimate(one_min_before, decimation_factor)
    print one_min_before

    x=(time_before_seizure * 60 * sampling_rate - start) / decimation_factor

    print np.shape(rpeaks)
    g = lambda a: a/4
    rpeaks = map(g, rpeaks_noise_removal)

    # rpeaks_one_minute_before = find_rpeaks(rpeaks, start * sampling_rate,
    #     end * sampling_rate) - start * sampling_rate
    # rpeaks_one_minute_before = rpeaks_one_minute_before/decimation_factor

    # print rpeaks

    path_to_save = '~/Desktop/seizure_datasets.h5'
    name_list = [str(time_before_seizure*60) + '_' + str(time_after_seizure*60)]
    group_list_esksmooth = ['esksmooth']
    mdata = {'fs':sampling_rate/decimation_factor}

    delete_signal(path_to_save, name_list, group_list_esksmooth)

    print 'KALMAN ...'
    tmp = time.time()
    filtered = EKSmoothing(one_min_before, rpeaks,
        fs=sampling_rate, bins=250, verbose=False, 
        oset=False, savefolder=None)
    s = time.time() - tmp

    print s, 
    print ' seconds'


    path_to_save = '~/Desktop/seizure_datasets.h5'
    name_list = [str(time_before_seizure*60) + '_' + str(time_after_seizure*60)]
    group_list_esksmooth = ['esksmooth']
    mdata = {'fs':sampling_rate/decimation_factor}



    save_signal(path=path_to_save, signal_list=[filtered],
                mdata_list=[mdata], name_list=name_list, group_list=group_list_esksmooth)