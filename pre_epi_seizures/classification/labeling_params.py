# labeling structure

label_struct = {
                    'baseline_files':{
                        'baseline':{
                            'label': -1,
                            'color': 'g',
                            'time_intervals_sec': [(0, 1000 * 60 * 60 )]
                        },
                    },
                    'seizure_files':{
                        'pre_ictal':{
                            'label': 1,
                            'color': 'y',
                            'time_intervals_sec': [(0, 1000 * 50 * 60 )]
                        },
                        'ictal':{
                            'label': 0,
                            'color': 'r',
                            'time_intervals_sec': [(1000 * 50 * 60, 1000 * 70 * 60 )]
                        },
                    },
               }

