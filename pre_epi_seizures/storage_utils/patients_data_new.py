
from datetime import time
from datetime import date


patients = {
# patient10 = {'sex': 'm',
#              'age': 55,
#              'ictal_clinical_on_time' [time(06, 44, 07),
#                                        time(10, 02, 46),
#                                        time(12, 04, 06),
#                                        time(17, 24, 58),
#                                       ],
#              'post_ictal_time': [time(06, 46, 29),
#                                  time(10, 04, 01),
#                                  time(12, 07, 38),
#                                  time(17, 25, 12),
#                                 ],
#              'ictal_on_time': [time(06, 44, 00),
#                                time(10, 02, 45),
#                                time(12, 04, 16),
#                                time(17, 24, 57),
#                               ],
#              'types_of_seizure': ['F',
#                                   'F',
#                                   'F',
#                                   'F'
#                                  ],

#              'location': ['']
#              'propagation': [], 
#              'dates_of_seizure': [date(2016, 9, 15),
#                                   date(2016, 9, 15),
#                                   date(2016, 9, 15),
#                                   date(2016, 9, 15),
#                                  ]

'4': {'sex': 'f',
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