"""
.. module:: h5repack
   :platform: Unix, Windows
   :synopsis: Tools for the HDF5 repackaging.
   
.. modeuleauthor:: Carlos Carreiras
"""


import os, fnmatch
import subprocess
import logging
import h5py
import json


def find_files(directory, pattern):
	"""
	
	Find files in a directory given a search pattern.
	
	Kwargs:
		directory (str): Path to search.
		
		pattern (str): Search pattern
	
	Kwrvals:
		
	
	See Also:
		
	
	Notes:
		
	
	Example:
		
	
	References:
		.. [1]
		
	"""
	
	for root, dirs, files in os.walk(directory):
		for basename in files:
			if fnmatch.fnmatch(basename, pattern):
				filename = os.path.join(root, basename)
				yield filename


def repack(path):
	"""
	
	Script to repack the HDF5 files (clear unused space) on the BioMESH directory.
	
	Kwargs:
		path (str): The directory with the HDF5 files to repack.
	
	Kwrvals:
		
	
	See Also:
		
	
	Notes:
		
	
	Example:
		
	
	References:
		.. [1]
		
	"""
	
	# setup logging
	logging.basicConfig(filename=os.path.join(os.path.expanduser('~'), 'h5repack', 'h5repack.log'),level=logging.INFO, format='%(asctime)s: %(message)s', filemode='w')
	logging.info("Repackaging started.")
	
	tst = []
	
	try:
		path = os.path.normpath(path)
		
		for fname in find_files(path, '*.hdf5'):
			# logging
			logging.info("Repacking file %s" % fname)
			tst.append(fname)
			
			# check repack flag
			fid = h5py.File(fname)
			try:
				flag = fid.attrs['repack']
			except KeyError:
				flag = True
			if not flag:
				fid.close()
				continue
			else:
				fid.attrs['repack'] = False
				fid.attrs['delete'] = json.dumps({'list': []})
			fid.close()
			
			# names
			root = os.path.split(fname)[0]
			name = os.path.split(fname)[1].split('.')[0]
			fname2 = os.path.join(root, name) + '_tmp.hdf5'
			if os.path.exists(fname2):
				os.remove(fname2)
			fname3 =  os.path.join(root, name)+'.old'
			if os.path.exists(fname3):
				os.remove(fname3)
			
			# repack
			command = 'h5repack -v "' + fname + '" "' + fname2 + '"'
			p = subprocess.Popen(unicode(command).encode('utf-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
			stdout, stderr = p.communicate()
			ret = p.returncode
			
			# logging
			logging.info(stdout)
			logging.info(stderr)
			
			# rename
			os.rename(fname, fname3)
			os.rename(fname2, fname)
	except Exception, e:
		logging.error(e)
	
	# logging
	logging.info("Repackaging finished.")
	
	return tst
	
	
if __name__ == '__main__':
	repack('/home/biomesh/BioMESH')
