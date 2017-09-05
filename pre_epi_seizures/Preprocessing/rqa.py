from pyrqa.settings import Settings
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from pyrqa.computation import RecurrencePlotComputation
from pyrqa.image_generator import ImageGenerator


def rqa_computation(signal_arguments, sampling_rate):
    beats_list = signal_arguments['feature_group_to_process']
    rqa_list = [compute_rqa(beats_seizure) for beats_seizure in beats_list]
    mdata = ['']*len(rqa_list)

    return rqa_list, mdata

def compute_rqa(list_beats_sz):
    rqa_seizure = map(compute_rqa_beat, list_beats_sz)
    return rqa_seizure


def compute_rqa_beat(beat):
    settings = Settings(beat,
                            embedding_dimension=10,
                            time_delay=1,
                            neighbourhood=FixedRadius(0.1),
                            similarity_measure=EuclideanMetric,
                            theiler_corrector=1,
                            min_diagonal_line_length=2,
                            min_vertical_line_length=2,
                            min_white_vertical_line_length=2)
    quantitative = RQAComputation.create(settings, verbose=False)
    result = quantitative.run()
    return result.determinism