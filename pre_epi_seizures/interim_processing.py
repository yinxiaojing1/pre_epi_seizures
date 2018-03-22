import numpy as np


def assign_baseline_seizure_random(baseline_data,
                    seizure_data,
                    seizure_nr_id):
    
 
    # Get seizures
    seizures = seizure_data[seizure_nr_id].unique()
   
    # randomly assign seizures
    np.random.shuffle(seizures)
    frac = len(baseline_data)/len(seizures)
    print 'frac'
    print frac
    
    for i, seizure in enumerate(seizures):
        print ''
        print i*frac
        print (i+1)*frac
        print ''
        baseline_data.iloc[i*frac:(i+1)*frac][seizure_nr_id] = seizure
        
    return baseline_data