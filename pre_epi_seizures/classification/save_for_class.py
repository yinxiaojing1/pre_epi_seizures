import pandas as pd
from uuid import *

def get_name_to_save(path_to_save, seizure_nr, trial):
    test_set = '---Test:' + str(seizure_nr)
    return path_to_save + test_set


def parse_pipeline_str(pipeline):
    steps = pipeline.steps
    final_str = ''
    for item in steps:
        final_str = final_str + str(item) + '/'
    return final_str[:-1]


def save_model(clf):
    import pickle
    with open('our_estimator.pkl', 'wb') as fid:
        pickle.dump(clf, fid)

    print 'Saved!!'
    with open('our_estimator.pkl', 'rb') as fid:
        clf = pickle.load(fid)

    print 'Load!!!'



def get_full_pipeline_name(path_to_save,
                       file_to_save,
                       pipeline, 
                       scoring,
                       param_grid,
                       feature_names,
                       cv_out,
                       cv_in,
                       trial):
  

    pipeline = parse_pipeline_str(pipeline)
    pipeline = '/Model:' + pipeline 
    scoring = '/Scoring:' + str(scoring)
    param_grid = '/Params:' + str(param_grid)
    feature_names = '/Feature_Names:' + str(feature_names)
    cv_out = '/CV_out:' + str(cv_out)
    cv_in = '/CV_in:' + str(cv_in)
    trial = '/Trial:' + str(trial)
    path = path_to_save + pipeline + scoring + feature_names + cv_out + cv_in + trial + file_to_save
    path = path.replace('\n', '')
    path = path.replace(' ', '')

    return path


def make_dir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    
def make_corresp_unique_file(index_file):
    if not os.path.exists(index_file):
        os.makedirs(index_file)
        index = pd.DataFrame( {'file_id':'NA', 'filename':'NA'})
        index.to_hdf(index_file, key=None, format='w', mode='w')


def insert_corresp_unique_file(path, filename):
    index_file = path + 'index.h5'
    make_corresp_unique_file(index_file)
    index = pd.read_hdf(index_file, key=None, mode='r+')
    
    # make unique id
    file_id = uuid.uuid4()
    index.append([file_id, filename], columns=['file_id', 'filename'])
    
    index.to_hdf(index_file, key=None, format='w', mode='w')
    
    
def h5store(filename, df, **kwargs):
    store = pd.HDFStore(filename)
    store.put('mydata', df)
    store.get_storer('mydata').attrs.metadata = kwargs
    store.close()

    
def h5load(filename):
    with pd.HDFStore(filename) as store:
        data = store['mydata']
        metadata = store.get_storer('mydata').attrs.metadata
        return data, metadata


def h5loadmodel(filename):
    with pd.HDFStore(filename) as store:
        metadata = store.get_storer('mydata').attrs.metadata
        return metadata
                       
