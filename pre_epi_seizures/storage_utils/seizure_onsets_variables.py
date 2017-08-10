seizure_onsets_HH_MM_SS = ['00:14:36', '01:02:43', '02:55:51',
             '01:24:34', '02:34:27', '00:20:10', '00:24:07',
             '00:51:25', '02:04:45', 
             '01:08:02']


def seizure_onsets_seconds_converter(hh_mm_ss):
    hours = float(hh_mm_ss[0:2])*3600
    minutes = float(hh_mm_ss[3:5]) * 60
    seconds = float(hh_mm_ss[6:8])

    return hours + minutes + seconds


seizure_onsets_seconds = map(seizure_onsets_seconds_converter,
                             seizure_onsets_HH_MM_SS) 

