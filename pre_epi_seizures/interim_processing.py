import numpy as np


def assign_random_baseline_seizure(baseline_data,
                    seizure_data,
                    seizure_nr_id,
                    patient_id):
    
    # Get patients in seizure dataframe
    patients = seizure_data[patient_id].unique()
    
    # Loop over patients in acquisition
    for patient in patients:
    
        # Get seizures
        seizures = seizure_data[seizure_nr_id].loc(seizure_data[patient_id]==patient)
        seizures = seizures.unique()
   
        # randomly assign seizures
        np.random.shuffle(seizures)
        frac = len(baseline_data)/len(seizures)

        for i, seizure in enumerate(seizures):
            baseline_data.iloc[i*frac:(i+1)*frac][seizure_nr_id] = seizure
        
        return baseline_data
    
    
def assign_equal_baseline_seizure(baseline_data,
                    seizure_data,
                    seizure_nr_id,
                    patient_id):
    
    # Get patients in seizure dataframe
    patients = seizure_data[patient_id].unique()
    
    # Loop over patients in acquisition
    for patient in patients:
    
        # Get seizures
        seizures = seizure_data[seizure_nr_id].loc[seizure_data[patient_id]==patient]
        
        seizures = seizures.unique()
        
        # Get baseline
        baseline_data_ix = baseline_data.index[baseline_data[patient_id]==patient].tolist()
       
        frac = len(baseline_data_ix)/len(seizures)
        
        for i, seizure in enumerate(seizures):
            
            print patient
            print seizure
            
            print 'changing ix'
            print baseline_data_ix[i*frac:(i+1)*frac]
            
            baseline_data.loc[baseline_data_ix[i*frac:(i+1)*frac], seizure_nr_id] = seizure
            
    return baseline_data