import h5py
import matlab.engine

disk = '/mnt/Seagate/'
files_dir = 'h5_files_backup/processing_datasets/'
file_name = 'data_for_matlab.h5'
file_path = disk + files_dir + file_name



def save_window_sec_analysis(window, 
                             L,
                             name_window_signal,
                             window_signal,
                             name_window_time_domain,
                             window_time_domain):
    save_all_new_for_matlab = True
    if save_all_new_for_matlab:
        # Declare globaL file path
        global file_path

        # open file
        file = h5py.File(file_path, 'w')
        
        # get correct datasets information
        window_signal_name = '/window_sec{}__L/'.format(window)
        signal_name = name_window_signal
        time_domain_name = name_window_time_domain

        # Overwrite the data
        try:
            del data_for_matlab_f[window_signal_name + signal_name]
        except Exception as e:
            print e
        try:
            del data_for_matlab_f[window_signal_name + time_domain_name]
        except Exception as e:
            print e
            
        file.create_dataset(window_signal_name + signal_name, data=window_signal)
        file.create_dataset(window_signal_name + time_domain_name, data=window_time_domain)
        file.close()

        
def load_window_sec_analysis(window,
                             L,
                             name_window_signal,
                             name_window_time_domain):
   
    # Declare globaL file path
    global file_path
    
    # open file
    file = h5py.File(file_path, 'r')
    
    # get correct datasets information
    window_signal_name = '/window_sec{}/'.format(down, up)
    signal_name = name_window_signal
    time_domain_name = name_window_time_domain
    
    # Load the hdf5 datasets 
    signal_dataset = file[window_signal_name + signal_name]
    time_domain_dataset = file[window_signal_name + time_domain_name]
    
    # Load the datasets into memory
    signal = signal_dataset[:]
    time_domain = time_domain_dataset[:]
    
    
    return signal, time_domain
    