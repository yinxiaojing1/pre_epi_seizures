

def evaluate_overall_labels(table):
    per_patient = table[['precision',
                     'recall',
                     'f1-score',
                     'patient_nr',
                     'types_of_seizure',
                     'Labels']].groupby(['patient_nr', 'Labels']).describe()
    
    per_patient = per_patient.loc[:, (['precision', 'recall', 'f1-score'], ['mean', 'std'])]
    return per_patient


def evaluate_per_types_of_seizure(table):
    per_patient = table[['precision',
                         'recall',
                         'f1-score',
                         'patient_nr',
                         'Labels',
                         'types_of_seizure',
                         'location']].groupby(['patient_nr',
                                               'Labels',
                                               'types_of_seizure',
                                               'location']).describe()
    
    per_patient_per_seizure_df = per_patient.loc[:, (['precision', 'recall', 'f1-score'], ['mean', 'std'])]
    
    return per_patient_per_seizure_df

