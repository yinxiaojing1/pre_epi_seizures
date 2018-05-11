import os
import glob
import cPickle
import traceback

ext = '*.py'
datapath = r'C:\work\biosppy'

# for each module
for module in ['bvp', 'ecg', 'eda', 'emg', 'resp']:#['', 'bvp', 'database', 'ecg', 'eda', 'eeg', 'emg', 'resp']:
	print module
	# get .py files
	for pyfile in glob.glob(os.path.join(datapath+'\\'+module, ext)):
		print '\t'+pyfile,
		module_name = pyfile.split('\\')[-1].split('.')[0]
		if module_name.find('__init__') >= 0 or module_name.find('sandbox') >= 0: continue
		# open and read .py file
		file = open(pyfile, "r")
		lines = file.readlines()
		file.close()
		new_lines = list(lines)
		# backup with .bak extension
		# bak_fn = pyfile.replace('.py', '.bak')
		# bak_file = open(bak_fn, 'wb')
		# bak_file.writelines(lines)
		# bak_file.close()
		# start analysis
		res = {}
		ln = -1
		numberadds = 0
		while True:
			try:
				ln +=1
				l = lines[ln]
				# analyze until main definition is found (no functions after that)
				if l.find('if __name__==\'__main__\':') >= 0: break
				# search for a def
				idx = l.find('def ')
				if idx >= 0:
					# get arguments starting position
					idx2 = l.find('(')
					# get function name
					name = module_name+'.'+l[idx+4:idx2]
					idx3 = l.find('):')
					auxp = l[idx2+1:idx3].split(',')
					# get conf. fields, inputs
					conffields, inputs = {}, []
					for p in auxp:
						p=p.replace(' ', '')
						p=p.strip()
						psl = p.split('=')
						if psl[1].find('None') >= 0 or psl[1] == '{}' or psl[1] == '()':
							# pass
							inputs.append(psl[0])
						else:
							conffields[psl[0]] = psl[1]
					outputs = []
					while True:
						ln +=1
						l = lines[ln]
						# get Kwargs
						idx = l.find('Kwargs:')
						if idx >= 0:
							idx = ln+1
							while True:
								ln +=1
								l = lines[ln]
								# until finding Kwrvals
								idx2 = l.find('Kwrvals:')
								if idx2 < 0:
									pass
									# aux = l.split(':')[0]
									# aux = aux.split('(')[0]
									# aux = aux.replace(' ', '')
									# aux = aux.strip()
									# if aux != '': inputs.append(aux)
								else: 
									break
							break
					# get Kwrvals
					while True:
						ln +=1
						l = lines[ln]
						# until finding See Also
						idx2 = l.find('See Also:')
						if idx2 < 0:
							aux = l.split(':')[0]
							aux = aux.split('(')[0]
							aux = aux.replace(' ', '')
							aux = aux.strip()
							if aux != '': outputs.append(aux)
						else: 
							break
					# get inputs
					aux = 'Configurable fields:{"name": "%s", '%name
					aux += ('"config": %s, '%conffields).replace('\'', '"')
					aux += ('"inputs": %s, '%inputs).replace('\'', '"')
					aux += ('"outputs": %s}'%outputs).replace('\'', '"')
					res[name] = aux
					
					new_lines.insert(ln+numberadds, ' '+' '+' '+' '+aux+'\n\n')
					numberadds += 1
					
			except Exception as e:
				print traceback.format_exc()
				break 

		# file = open(pyfile.replace('.py', '.dictdoc'), "wb")
		# # cPickle.dump(res, file)
		# for k in res: file.write(k+'\n\n'+res[k]+'\n\n\n')
		# file.close()
		
		file = open(pyfile, "wb")
		# cPickle.dump(res, file)
		file.writelines(new_lines)
		file.close()
		

		