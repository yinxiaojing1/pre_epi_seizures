"""
.. module:: cvp2biomesh
   :platform: Unix, Windows
   :synopsis: Script to import data from Cruz Vermelha Portuguesa to StorageBIT.

.. moduleauthor:: Filipe Canento, Carlos Carreiras


"""

# imports
import os
import glob
import numpy
import datetime

from database import biomesh
import plux

import xlrd

def age_in_years(from_date, to_date=None, leap_day_anniversary_Feb28=True):
    if to_date == None:
        to_date = datetime.datetime.today()
    age = to_date.year - from_date.year
    try:
        anniversary = from_date.replace(year=to_date.year)
    except ValueError:
        assert from_date.day == 29 and from_date.month == 2
        if leap_day_anniversary_Feb28:
            anniversary = datetime.date(to_date.year, 2, 28)
        else:
            anniversary = datetime.date(to_date.year, 3, 1)
    if to_date < anniversary:
        age -= 1
    return age

def cvp_files2db(config, datapath = r"Y:\\data\\CVP"):
    # connect to DB
    db = biomesh.biomesh(**config)
    
    # add the experiments
    db.experiments.add({'name': 'T1', 'goals': 'ECG acquisition.', 'description': 'First session; ECG acquired from hands, in recumbent and sitting positions.'})
    db.experiments.add({'name': 'T2', 'goals': 'ECG acquisition.', 'description': 'Second session; ECG acquired from hands, in recumbent and sitting positions.'})
    
    
    user_file_dict = {}
    
    # add subjects
    filename = '\\Lista de Participantes %s.xls'
    for experiment in ['Cardiopneumologia', 'Enfermagem', 'Fisioterapia']:
        # print experiment

        if experiment == 'Cardiopneumologia': idx = 0
        else: idx = -1
        
        #add subjects
        wb = xlrd.open_workbook(datapath+filename%experiment)
        sheet = wb.sheet_by_index(0)
        for rownum in range(1,sheet.nrows):
            # 'N Aluno', 'Sexo', 'Nome', 'D.N.', 'Altura (cm)', 'Peso (Kg)', \
            # 'Raca', 'E-Mail*', 'Telemovel*', 'Desporto', 'Desporto (2o teste)',\
            # '1o teste Sentado', '1o teste Deitado', '2o Teste Sentado', '2o teste Deitado'
            
            # print sheet.row_values(rownum), len(sheet.row_values(rownum))
            
            birthdate = sheet.cell_value(rownum,3)
            year, month, day, hour, minute, sec = xlrd.xldate_as_tuple(birthdate, wb.datemode)
            birthdate = datetime.datetime(year, month, day, hour, minute, sec).isoformat()
            
            name = sheet.cell_value(rownum,2)
            gender = sheet.cell_value(rownum,1)
            fname1, fname2, fname3, fname4 = sheet.cell_value(rownum,11+idx), sheet.cell_value(rownum,12+idx), sheet.cell_value(rownum,13+idx), sheet.cell_value(rownum,14+idx)
            
            
            # MISSING FILEs
            aux = numpy.array(map(lambda pat: fname1.find(pat), ['F2851', 'F2847', 'F2825', 'F2826', 'F2828', 'F2845', 'F2871', 'F2908']))
            if numpy.any(aux>=0):
                print "skipped %s" %fname1
                continue
                
            ext='.txt'
            fname1, fname2, fname3, fname4 = fname1+ext, fname2+ext, fname3+ext, fname4+ext
            if gender == 'M':
                # print gender, fname1, fname2, fname3, fname4
                for timest in ['-T1', '-T2']:
                    rawdatapath = datapath+'\\'+experiment+'\\'+experiment+timest
                    ext = '*.txt'
                    for fname in glob.glob(os.path.join(rawdatapath, ext)):
                        fsplt = fname.split('\\')[-1]
                        # print fsplt
                        # print fname1, fname2, fname3, fname4
                        if fsplt in [fname1, fname2, fname3, fname4]:
                            newfname = fname.replace('F', 'M')
                            try:
                                os.rename(fname, newfname)
                                print "corrected %s to %s"%(fname, newfname)
                            except Exception as e:
                                # pass
                                print fname
                                print e
                fname1 = fname1.replace('F', 'M')
                fname2 = fname2.replace('F', 'M')
                fname3 = fname3.replace('F', 'M')
                fname4 = fname4.replace('F', 'M')
            
            
            info = {
                    'number': sheet.cell_value(rownum,0),
                    'sex': gender,
                    'name': name,
                    'birthdate': birthdate,
                    'heigth': sheet.cell_value(rownum,4),
                    'weigth': sheet.cell_value(rownum,5),
                    # 'ethnicity': sheet.cell_value(rownum,6),
                    'email': sheet.cell_value(rownum,7)}#,
                    # 'mobilenr': sheet.cell_value(rownum,8),
                    # 'sport1': sheet.cell_value(rownum,9),
                    # 'sport2': sheet.cell_value(rownum,10),
                    # 'file': {
                            # '1Sentado': fname1, 
                            # '1Deitado': fname2, 
                            # '2Sentado': fname3, 
                            # '2Deitado': fname4}}
            # add the subject
            subId = db.subjects.add(info)['subjectId']
            # print subId
            
            user_file_dict[str(fname1)] = subId
            user_file_dict[str(fname2)] = subId
            user_file_dict[str(fname3)] = subId
            user_file_dict[str(fname4)] = subId
            
    # add data
    for source in ['Cardiopneumologia', 'Enfermagem', 'Fisioterapia']:
        for timest in ['-T1', '-T2']:
            # data path
            rawdatapath = datapath+'\\'+source+'\\'+source+timest
            ext = '*'+timest+'*.txt'
            
            for fname in glob.glob(os.path.join(rawdatapath, ext)):
                print fname
                fsplt = fname.split('\\')[-1]
                try:
                    subId = user_file_dict[fsplt]
                except Exception as e:
                    print "skipped %s"%fname
                    continue
                
                # Read samples file
                data = plux.loadbpf(fname)
                data = data[:,3]
                
                # datetime
                date = data.header['StartDateTime'].replace(' ', 'T')
                
                # get record
                out = db.records.getAll({'experiment': timest[1:], 'subject': subId})['idList']
                if len(out) == 0:
                    recId = db.records.add({'date': date, 'experiment': timest[1:], 'subject': subId, 'source': source})['recordId']
                else:
                    recId = out[0]
                
                if 'Deit' in fsplt:
                    position = 'Recumbent'
                elif 'Sent' in fsplt:
                    position = 'Sitting'
                
                mdata = {'labels': ['ECG'], 'device':{'Vcc': data.header['Vcc'], 'version': data.header['Version']}, 'units': {'time': 'second', 'sensor': data.header['Units']}, 'sampleRate': data.header['SamplingFrequency'], 'resolution': data.header['SamplingResolution']}
                db.records.addSignal(recId, '/ECG/hand/'+position+'/raw', data, mdata)
    
    # close db
    db.close()



if __name__=='__main__':
    config = {'dbName': 'CVP', 'host': '193.136.222.234', 'port': 27017, 'dstPath': 'D:\\BioMESH'}

    cvp_files2db(config)
    
