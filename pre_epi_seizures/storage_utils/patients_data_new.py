
from datetime import time
from datetime import date


patients = {
    
   
    
   '8': {'sex': 'm',
              'age': 55,
              'ictal_clinical_on_time': [time(19, 53, 57),
                                        time(5, 29, 13),
                                        time(18, 59, 42),
                                        time(3, 15, 59),
                                        time(12, 1, 48),
                                        time(12, 1, 48),
                                        time(12, 4, 46),
                                        time(12, 9, 13),
                                        time(12, 12, 30),
                                        time(18, 25, 9), 
                                        time(20, 59, 16),
                                        time(10, 50, 23),
                                        time(10, 51, 56),
                                        time(11, 7, 16),
                                        time(11, 9, 16)
                                        
                                       ],
              'post_ictal_time': [time(19, 54, 22),
                                  time(5, 29, 52),
                                  time(19, 0, 23),
                                  time(3, 15, 59),
                                  time(12, 2, 26),
                                  time(12, 5, 52),
                                  time(12, 9, 13),
                                  time(12, 13, 10),
                                  time(18, 27, 40),
                                  time(21, 0, 5),
                                  time(10, 50, 44),
                                  time(10, 52, 42),
                                  time(11, 7, 16),
                                  time(11, 10, 35)
                                  
                                 ],
              'ictal_on_time': [time(19, 53, 56),
                                time(5, 29, 13),
                                time(18, 59, 42),
                                time(3, 15, 57),
                                time(12, 1, 48),
                                time(12, 4, 39),
                                time(12, 9, 13),
                                time(12, 12, 37),
                                time(18, 25, 9),
                                time(20, 59, 16),
                                time(10, 50, 24),
                                time(10, 51, 56),
                                time(11, 7, 16),
                                time(11, 9, 15)
                                
                                
                               ],
              'types_of_seizure': ['F',
                                   'F',
                                   'F',
                                   'F',
                                   'F'
                                  ],

              'location': [''],
              'propagation': [], 
              'dates_of_seizure': [date(2016, 5, 9),
                                   date(2016, 5, 10),
                                   date(2016, 5, 12),
                                   date(2016, 5, 13),
                                   date(2016, 5, 13),
                                   date(2016, 5, 13),
                                   date(2016, 5, 13),
                                   date(2016, 5, 13),
                                   date(2016, 5, 13),
                                   date(2016, 5, 13),
                                   date(2016, 5, 14),
                                   date(2016, 5, 17),
                                   date(2016, 5, 17),
                                   date(2016, 5, 17),
                                  ]
       },  
    
    
    
 '10': {'sex': 'm',
              'age': 55,
              'ictal_clinical_on_time': [time(6, 44, 7),
                                        time(10, 2, 46),
                                        time(12, 4, 6),
                                        time(17, 24, 58),
                                       ],
              'post_ictal_time': [time(6, 46, 29),
                                  time(10, 4, 1),
                                  time(12, 7, 38),
                                  time(17, 25, 12),
                                 ],
              'ictal_on_time': [time(6, 44, 0),
                                time(10, 2, 45),
                                time(12, 4, 16),
                                time(17, 24, 57),
                               ],
              'types_of_seizure': ['F',
                                   'F',
                                   'F',
                                   'F',
                                  ],

              'location': [''],
              'propagation': [], 
              'dates_of_seizure': [date(2016, 9, 15),
                                   date(2016, 9, 15),
                                   date(2016, 9, 15),
                                   date(2016, 9, 15),
                                  ]
       },
    
'7': {'sex': 'm',
             'age': 55, 
             'ictal_clinical_on_time': [
                                        time(4, 20, 47),
                                        time(14, 38, 6),
                                        time(3, 41, 57),
                                        time(5, 59, 16),
                                        time(7, 57, 13),
                                        time(9, 3, 10),
                                        time(10, 47, 24),
                                        time(12, 16, 26),
                                      ],
             'post_ictal_time': [
                                 None,
                                 None,
                                 None,
                                 None,
                                 None,
                                 None,
                                 None,
                                 None
                                ],
             'ictal_on_time': [
                               time(4, 20, 52),
                               time(1, 41, 5),
                               time(2, 19, 28)
                               ],
             'types_of_seizure': [
                                  'FS',
                                  'FS',
                                  'FS'
                                 ],
             'location_of_seizure': [
                                      'FTR',
                                      'FTLR',
                                      'FTLR'
                                     ], 
             'dates_of_seizure': [date(2016, 7, 6),
                                  date(2016, 7, 6),
                                  date(2016, 7, 8),
                                 ]
             },

'5': {'sex': 'f',
             'age': 55, 
             'ictal_clinical_on_time': [time(9, 1, 41),
                                        time(14, 38, 6),
                                        time(3, 41, 57),
                                        time(5, 59, 16),
                                        time(7, 57, 13),
                                        time(9, 3, 10),
                                        time(10, 47, 24),
                                        time(12, 16, 26),
                                      ],
             'post_ictal_time': [time(9, 3, 8),
                                 None,
                                 None,
                                 None,
                                 None,
                                 None,
                                 None,
                                 None
                                ],
             'ictal_on_time': [time(9, 1, 42),
                               time(14, 38, 6),
                               time(3, 41, 52),
                               time(5, 58, 29),
                               time(7, 57, 13),
                               time(9, 3, 10),
                               time(10, 47, 24),
                               time(12, 16, 26)
                              ],
             'types_of_seizure': ['FS',
                                  'FS',
                                  'FS',
                                  'GS',
                                  'FS',
                                  'FS',
                                  'FS',
                                  'FS'
                                 ],

              'location_of_seizure': ['FTLR',
                                      'FTLR',
                                      'FTLR',
                                      'FTLR',
                                      'FTLR',
                                      'FTLR',
                                      'FTLR',
                                      'FTLR',
                                     ], 

             'dates_of_seizure': [date(2016, 8, 23),
                                  date(2016, 8, 24),
                                  date(2016, 8, 25),
                                  date(2016, 8, 25),
                                  date(2016, 8, 25),
                                  date(2016, 8, 25),
                                  date(2016, 8, 25),
                                  date(2016, 8, 28),
                                 ]
             },
                               
    
    
'4': {'sex': 'f',
             'age': 55, 
             'ictal_clinical_on_time': [time(15, 48, 16),
                                       time(18, 34, 13),
                                       time(10, 22, 26)
                                      ],
             'post_ictal_time': [time(15, 49, 10),
                                 time(18, 36, 1),
                                 time(10, 24, 14)
                                ],
             'ictal_on_time': [time(15, 48, 13),
                               time(18, 34, 14),
                               time(10, 22, 27)
                              ],
             'types_of_seizure': ['FC',
                                  'FC',
                                  'FC'
                                 ],

              'location_of_seizure': ['FTLR',
                                      'FTLR',
                                      'FTLR'
                                      ], 

             'dates_of_seizure': [date(2017, 1, 10),
                                  date(2017, 1, 10),
                                  date(2017, 1, 12),
                                 ]
            },

'3': {'sex': 'f',
             'age': 55, 
             'ictal_clinical_on_time': [time(8, 25, 44),
                                       time(16, 42, 53),
                                       time(21, 45, 12)
                                      ],
             'post_ictal_time': [time(8, 26, 27),
                                 time(16, 45, 14),
                                 time(21, 46, 12)
                                ],
             'ictal_on_time': [time(8, 25, 41),
                               time(16, 41, 35),
                               time(21, 43, 55)
                              ],
             'types_of_seizure': ['GS',
                                  'FC',
                                  'FS'
                                 ],

              'location_of_seizure': ['FTL',
                                      'FTL',
                                      'FTL',
                                     ], 

             'lateralization_onset':['RL',
                                     'L',

                                ],
             'dates_of_seizure': [date(2017, 4, 20),
                                  date(2017, 4, 20),
                                  date(2017, 4, 20),
                                 ]
            },

'2': {'sex': 'f',
             'age': 55, 
             'ictal_clinical_on_time': [time(15, 48, 16),
                                       time(18, 34, 13),
                                       # time(10, 22, 26)
                                      ],
             'post_ictal_time': [time(15, 49, 10),
                                 time(18, 36, 1),
                                 # time(10, 24, 14)
                                ],
             'ictal_on_time': [time(15, 48, 13),
                               time(18, 34, 14),
                               # time(10, 22, 27)
                              ],
             'types_of_seizure': ['FC',
                                  'FC',
                                  'FC'
                                 ],

              'location_of_seizure': ['FTLR',
                                      'FTLR',
                                      'FTLR'
                                      ], 

             'dates_of_seizure': [date(2017, 1, 10),
                                  date(2017, 1, 10),
                                  # date(2017, 1, 12),
                                 ]
            }


}