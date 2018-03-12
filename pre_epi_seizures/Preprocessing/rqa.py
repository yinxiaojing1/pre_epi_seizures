from pyrqa.settings import Settings
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from pyrqa.computation import RecurrencePlotComputation
from pyrqa.image_generator import ImageGenerator

import numpy as np

def rqa_computation(signal_arguments,
                    win_params, add_params,
                    win_param_to_process, param_to_process):
    signal_list = signal_arguments['feature_group_to_process']
    rpeaks_list = signal_arguments['rpeak_group_to_process']    
    rqa_list = [np.array(compute_rqa(beats_seizure)) for beats_seizure in signal_list]
    mdata = [
             {'feature_legend': ['entropy_white_vertical_lines', 'number_of_vertical_lines'
                                'number_of_white_vertical_lines', 'entropy_diagonal_lines',
                                'longest_white_vertical_line', 'longest_vertical_line',
                                'entropy_vertical_lines', 'longest_diagonal_line',
                                'number_of_diagonal_lines']}
                                ]*len(rqa_list)

    window_list = [rpeaks[0][1:-1] for rpeaks in rpeaks_list]

    return rqa_list, mdata, window_list


def compute_rqa(list_beats_sz):
    rqa_seizure = map(compute_rqa_beat, list_beats_sz)
    return rqa_seizure


def compute_rqa_beat(beat):
    settings = Settings(beat,
                            embedding_dimension=3,
                            time_delay=20,
                            neighbourhood=FixedRadius(0.1),
                            similarity_measure=EuclideanMetric,
                            theiler_corrector=1,
                            min_diagonal_line_length=2,
                            min_vertical_line_length=2,
                            min_white_vertical_line_length=2)
    quantitative = RQAComputation.create(settings, verbose=False)
    result = quantitative.run()
    features = result.__dict__
    
    stop
    # print np.asarray([k for k in features.keys()
    #                   if 'distribution' not in k
    #                   if 'points' not in k
    #                   if 'settings' not in k
    #                   if 'runtimes' not in k])
    return np.asarray([features[k] for k in features.keys()
                      if 'distribution' not in k
                      if 'points' not in k
                      if 'settings' not in k
                      if 'runtimes' not in k])
    # stop
    # stop
