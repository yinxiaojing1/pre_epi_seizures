import pandas as pd
import os
import uuid


def generate_string_identifier(**kwargs):
    # Convert parameter dict into string
    params_str = get_str_eda_params(**kwargs)
    
    return params_str


#def create_file_identifier(disk, eda_dir, **kwargs):
    

def get_str_eda_params(**kwargs):
    """
    Generates a string identifying the parameters of a pipeline.
    Name to be changed.
    
    kwargs: parameters of the pipeline
    """
    # Create the string
    str_params = ''
    for item in kwargs.iteritems():
        str_params = str_params + '' + str(item)
     
    return str_params


def generate_txt_file_params(path, str_params):
    """
    Generates a small txt file containing the a string of the 
    parameters.
    
    kwargs: parameters of the pipeline
    """    
    
    print path
    # Open and write to file
    text_file = open(path + "params.txt", "w")
    text_file.write(str_params)
    text_file.close()
    

def get_eda_params_path(disk, directory, params_str):
    
    print params_str

    table_path = disk + directory + 'eda_mdata.h5'
 
    # Load metadata
    if not os.path.exists(table_path):
        
        # Create new table
        print 'The table doesnt exist! Making a new one!'
        new_id = uuid.uuid1()  # generate new id
        table = pd.DataFrame([[params_str, new_id]],
                             columns=['params', 'id'])  # Create new table
        table.to_hdf(table_path, '/mdata')   # save table                   
        final_id = str(new_id)
        print 'New table done !'
                            
    else:
        # load metadata table in hdf5
        table = pd.read_hdf(table_path, '/mdata')
        print ''
        print 'This is a list of the parameters'
        print list(table['params'])
        print ''
        print 'This is the new one'
        print params_str
        print ''
        print 'id params in disk'
        id_params = table.loc[table['params'] == params_str]
        print id_params
        print ''
        print 'Check if they are the same'
        print ''
        print id_params==params_str

        
        if len(id_params):
            final_id = str(id_params['id'].loc[0])
            print ''
            print 'final id -- check disk'
            print final_id
            
         
        else:
            # generate new id 
            new_id = uuid.uuid1()
            new_row = pd.DataFrame([[params_str, new_id]],
                                 columns=['params', 'id'])  # Create new table
            table = pd.concat([table, new_row])
            table.to_hdf(table_path, '/mdata')   # save table                   
            final_id = str(new_id)
    
    print ''
    print 'Check for this path on disk .. should be there'
    print final_id
    
   
        
    return final_id


def read_only_table(disk, eda_dir, **kwargs):
    # Convert parameter dict into string
    params_str = get_str_eda_params(**kwargs)
    
    print params_str
    
    table_path = disk + eda_dir + 'eda_mdata.h5'
 
    # Load metadata
    if not os.path.exists(table_path):
        print 'The table does Not exist!'
        stop
    
    else:
        # load metadata table in hdf5
        table = pd.read_hdf(table_path, '/mdata')
        print ''
        print 'This is a list of the parameters'
        print list(table['params'])
        print ''
        print 'This is the new one'
        print params_str
        print ''
        print 'id params in disk'
        id_params = table.loc[table['params'] == params_str]
        print id_params
        print ''
        print 'Check if they are the same'
        print ''
        print id_params==params_str

        
        if len(id_params):
            final_id = str(id_params['id'].loc[0])
            print ''
            print 'final id -- check disk'
            print final_id
            
        else:
            print 'The parameters do not exist'
            stop
     
    return final_id
    


def delete_entry(table_path, table, idx):

    print ''
    print 'Check for table before deletion'
    
    print 'This is the id to delete'
    print idx
    print ''
    print table
    print 'This is the entry to delete it should not be empty'
    print table.loc[table['id'] == idx]
    table.drop(table[table['id'] == idx].index, inplace=True)
          
    print 'Should be empty'
    print table.loc[table['id'] == idx]
    print ''
                     
    table.to_hdf(table_path, '/mdata')
    

def check_eda_dir(disk, eda_dir):
       
    table_path = disk + eda_dir + 'eda_mdata.h5'    
    table_eda = pd.read_hdf(table_path, '/mdata')
    
    eda_parameters_list = list(table_eda['params'])
    eda_id_list = list(table_eda['id'])
    
    print eda_id_list
    
    stop
    
    for eda_parameters, eda_idx in zip(eda_parameters_list, eda_id_list):
        print eda_parameters
        print ''
        print eda_idx
        print ''
        print '------'
        id_df = table_eda.loc[table_eda['id'] == eda_idx]
        if id_df.empty:
            print 'Halt'
            print 'The id seems to be in the table, but cannot be accessed'
            stop
    
    path = disk + eda_dir + str(eda_idx) + '/'

    print 'The EDA path exists?'
    print os.path.exists(path)

    # Check if path exist
    if not os.path.exists(path):
        delete_entry(table_path, table_eda, eda_idx)

    elif os.listdir(path) == []:
        print ''
        print 'Directory is empty'
        print 'Remove the dir'
        os.rmdir(path)
        print 'Done'

    
    

def check_table(disk, eda_dir):
    
    table_path = disk + eda_dir + 'eda_mdata.h5'    
    table_eda = pd.read_hdf(table_path, '/mdata')
    
    eda_parameters_list = list(table_eda['params'])
    eda_id_list = list(table_eda['id'])
   
    
    
    for eda_parameters, eda_idx in zip(eda_parameters_list, eda_id_list):
        print '--------------'
        print ''
        print 'This is a list of the EDA parameters present in the table'
        print eda_parameters
        print ''
        print eda_idx
        print ''
        
        id_df = table_eda.loc[table_eda['id'] == eda_idx]
        if id_df.empty:
            print 'Halt'
            print 'The id seems to be in the table, but cannot be accessed'
            stop
        
        path = disk + eda_dir + str(eda_idx) + '/'
        
        print 'The EDA path exists?'
        print os.path.exists(path)
        
        # Check if path exist
        if not os.path.exists(path):
            delete_entry(table_path, table_eda, eda_idx)
            
        elif os.listdir(path) == []:
            print ''
            print 'Directory is empty'
            print 'Remove the dir'
            os.rmdir(path)
            print 'Done'

            
        else:
            
            table_path = disk + eda_dir + str(eda_idx) + '/' + 'eda_mdata.h5'    
            table = pd.read_hdf(table_path, '/mdata')
            parameters_list = list(table['params'])
            id_list = list(table['id'])
            print ''

            for parameters, idx in zip(parameters_list, id_list):
                print 'This is a list of the classification list present'
                print parameters
                print ''
                print idx
                class_path = path + str(idx) + '/'
                
                id_df_class = table.loc[table_eda['id'] == idx]
                if id_df_class.empty:
                    print 'Halt'
                    print 'The id seems to be in the table, but cannot be accessed'
                    stop
                    
                
                print ''
                print 'The classification path exists?'
                print os.path.exists(class_path)
                
                if not os.path.exists(class_path):
                    delete_entry(table_path, table, idx)
            
                else:

                    # Check for optimization files, if not delete
                    results = [name
                               for name in os.listdir(class_path)
                               if 'hp_opt_results' in name]

                    print results

                    if not len(results):
                        print 'Delete'
                        delete_entry(table_path, table, idx)

            print '----------------'
            print ''
            print ''