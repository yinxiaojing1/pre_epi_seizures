"""
	This module provides various functions to ...

	Functions:

	loadbpf()

	"""	
import numpy as np
import copy as c
import pylab as pl

#----------------------------------------------------------------------------------------------------------------------------	
# Load data
def loadbpf(fname):
	"""
		Loads biosignal data.

		Parameters
		__________
		fname : string
			input filename

		Returns
		_______
		bpd : bparray
			output biosignal array

		See Also
		________
			np.loadtxt

		Notes
		_____
		

		Example
		_______


		References
		__________
		.. [1] 

		http://
	"""
	# Check
	hdr={}
	hdr['Version']=0
	bpf=open(fname,'r')
	try:
		for l in bpf:
			if l[0]!='#':
				break
			ddp=l.find(':')
			if ddp>=0:
				val=l[(ddp+1):].strip()
				if val.isdigit():
					val=int(val)
				hdr[l[1:ddp].strip()]=val
	except:
		bpf.close()
		raise
	if hdr['Version']==0:
		hdr['SamplingResolution']=12.
		hdr['SamplingFrequency']=1000.0
	if hdr['Version']<=1:
		hdr['Vcc']=5.
	hdr['Units']='ADC'
	bpd=bparray(np.loadtxt(fname, comments='#', delimiter='\t'), hdr)
	
	dseq=pl.find((pl.diff(bpd[:,0])!=1.) & (pl.diff(bpd[:,0])!=-127.))
	
	if (len(dseq)!=0): print("Warning: sample loss was detected at indexes '"+str(dseq+1)+"'.")

	return bpd
#----------------------------------------------------------------------------------------------------------------------------	

#----------------------------------------------------------------------------------------------------------------------------	
class bparray(np.ndarray):
    
    def __new__(cls, ndarr, hdr={}):
        obj=np.asarray(ndarr).view(cls)
        obj.header=hdr.copy()
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return
        self.header=getattr(obj, 'header', None)

    def toADC(self):
        unit='ADC'
        
        cself=bparray(self,self.header)
        
        if cself.header['Units']==unit: return cself

        cself=(cself.toV()*2.**cself.header['SamplingResolution'])/cself.header['Vcc']
        cself.header['Units']=unit
        
        return cself
    
    def toV(self):
        unit='V'
        
        cself=bparray(self,self.header)
        
        if cself.header['Units']==unit: return cself
        elif cself.header['Units']=='ADC': 
            cself=cself.header['Vcc']*cself/2.**cself.header['SamplingResolution']
        elif cself.header['Units']=='uS':
            cself=cself*cself.header['Vcc']*0.2
        #    cself=cself*cself.header['Vcc']+0.2
        elif cself.header['Units']=='mV':
            cself=cself*1000.
        else:
            raise TypeError("conversion from "+cself.header['Units']+" is not currently supported")
        
        cself.header['Units']=unit
        return cself
    
    def tomV(self):
        unit='mV'
        
        cself=bparray(self,self.header)
        
        if cself.header['Units']==unit: return cself

        cself=cself.toV()/1000.
        cself.header['Units']=unit
        
        return cself
    
    def touS(self):
        unit='uS'
        
        cself=bparray(self,self.header)
        
        if cself.header['Units']==unit: return cself
        
        #cself=(cself.toV()-0.2)/cself.header['Vcc']
        cself=cself.toV()/0.2
        cself.header['Units']=unit
        
        return cself
#----------------------------------------------------------------------------------------------------------------------------	
