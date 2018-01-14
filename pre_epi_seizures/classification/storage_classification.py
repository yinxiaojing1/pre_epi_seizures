

def load_classification_data_frame(path):
    try:
        results = pd.read_hdf(file_to_save + filename + '.h5', 'test') 
        print 'Optimization Already in disk!'
        return results

    except Exception as e:
        print e
        return False