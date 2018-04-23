# Import for I/O 
from storage_utils_nihon_khoden import *
from create_datasets_nk import *
from create_free_datasets_nk import *



# State the parameters of the pipeline
patient_list = [10]
disk = '/mnt/Seagate/pre_epi_seizures/'
baseline_files = 'h5_files/processing_datasets/baseline_datasets_new'
seizure_files = 'h5_files/processing_datasets/seizure_datasets_new'
time_before_seizure = 50 * 60
time_after_seizure = 20 * 60
time_baseline = 4 * 60 * 60

# Create Raw ECG datasets


create_raw=False
if create_raw:
    create_datasets_nk(disk, time_before_seizure, time_after_seizure,
                       patient_list)

    create_free_datasets_nk(disk, time_baseline, patient_list)


from pre_processing import _main
_main(disk, seizure_files)
_main(disk, baseline_files)