"""
.. module:: repackScript
   :platform: Unix, Windows
   :synopsis: Script to run the HDF5 repackaging. Appropriately configure the global variables.
   
.. modeuleauthor:: Carlos Carreiras
"""


# globals
BIOSPPY = '/home/biomesh/biosppy'
BIOMESH = '/home/biomesh/BioMESH'


import sys
sys.path.append(BIOSPPY)
from database import h5repack as rpk


def run():
	"""
	
	Script to repack the HDF5 files (clear unused space) on the BioMESH directory.
	
	Kwargs:
		
	
	Kwrvals:
		
	
	See Also:
		
	
	Notes:
		
	
	Example:
		
	
	References:
		.. [1]
		
	"""
	
	rpk.repack(BIOMESH)
	
	
if __name__ == '__main__':
	run()
