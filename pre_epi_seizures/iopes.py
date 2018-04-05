import pandas as pd
import os
import uuid


def get_str_eda_params(**kwargs):
    str_params = ''
    for item in kwargs.iteritems():
        str_params = str_params + '' + str(item)
     
    return str_params


def get_eda_params_path(disk, eda_dir, **kwargs):
    
    # Convert parameter dict into string
    params_str = get_str_eda_params(**kwargs)
    
    print params_str
    
    table_path = disk + eda_dir + 'eda_mdata.h5'
 
    # Load metadata
    if not os.path.exists(table_path):

        new_id = uuid.uuid1()  # generate new id
        table = pd.DataFrame([[params_str, new_id]],
                             columns=['params', 'id'])  # Create new table
        table.to_hdf(table_path, '/mdata')   # save table                   
        final_id = str(new_id)
                            
    else:
        # load metadata table in hdf5
        table = pd.read_hdf(table_path, '/mdata')
      
        id_params = table.loc[table['params'] == params_str]
                             
        if len(id_params):
            final_id = str(id_params['id'].loc[0])
         
        else:
            # generate new id 
            new_id = uuid.uuid1()
            table = pd.DataFrame([[params_str, new_id]],
                                 columns=['params', 'id'])  # Create new table
            table.to_hdf(table_path, '/mdata')   # save table                   
            final_id = str(new_id)
   
        
    return final_id

        
    