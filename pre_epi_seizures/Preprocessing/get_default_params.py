def input_default_params(input_params_dict, **default_params_dict):
    final_win_params = dict()

    for k in default_params_dict.keys():
        try:
            final_win_params[k] = input_params_dict[k]
        except Exception as e:
            final_win_params[k] = default_params_dict[k]

    return final_win_params



def get_input_and_default_params(win_params, add_params, feature_name):

    if feature_name == 'baseline_removal':

        final_win_params = input_default_params(win_params,
                            win=0.001,
                            init=0,
                            finish=4200,
                            samplerate=1000)

        final_add_params = input_default_params(add_params,
                            filt='medianFIR')

    if feature_name == 'rpeak_detector':

        final_win_params = input_default_params(add_params,
                            win='rpeaks',
                            samplerate=1000)

        final_add_params = input_default_params(add_params,
                            method='hamilton')

    if feature_name == 'hrv_computation':
        print 'here'
        final_win_params = input_default_params(win_params,
                            win=0.001,
                            init='rpeaks[0]',
                            finish='rpeaks[-1]',
                            samplerate=1000)

        final_add_params = input_default_params(add_params,
                            method='hamilton',
                            resampling='spline')

    if feature_name == 'hrv_features':
        print 'hrv_features'
        final_win_params = input_default_params(win_params,
                            win=3*60*1000,
                            init=0,
                            finish=4200)

        final_add_params = input_default_params(add_params,
                            method='hamilton',
                            resampling='spline')


    return final_win_params, final_add_params
