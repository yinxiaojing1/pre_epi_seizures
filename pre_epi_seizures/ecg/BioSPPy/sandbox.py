import pylab as pl
import sys
import os

#sys.path.append('/Users/hsilva/Work/IT/code/python/biosppy')
sys.path.append('C:/Users/Filipe/Desktop/biosppy')
sys.path.append('bvp')
sys.path.append('eda')

import bvp
import edafc as edafc
import eda
import plux
import sync
import tools as tls

reload(bvp)
reload(eda)
reload(edafc)

pl.ioff()

fname='../data/PN11 12.12.txt'
#fname='../data/PN12 13.12.txt'
#fname='../data/PN13 13.12.txt'

print "loading "+fname

Raw=plux.loadbpf(fname)

SamplingRate=10.
# SamplingRate=20

DownSampling=Raw.header['SamplingFrequency']/SamplingRate
Data=Raw[::DownSampling,:]

print "switch events"
Switch=sync.step(Raw[:,1])#Switch=sync.step(Data[:,1])
Switch['Event']/=Raw.header['SamplingFrequency']/SamplingRate

print "eda events"
EDA=eda.scr(Signal=Data[:,3].touS(), SamplingRate=SamplingRate, Filter={'UpperCutoff':0.05})
# EDA=edafc.scr(Signal=Raw[:,3].touS(), SamplingRate=1000., Filter={'UpperCutoff':0.05})
# EDA=edafc.scrKBK(Signal=Raw[:,3].touS(), SamplingRate=1000.)

da=pl.zeros(len(EDA['Amplitude']))
da[1:]=abs(pl.diff(EDA['Amplitude']))

idx=(da>=pl.std(da)*.5)

print "bvp events"
BVP=bvp.pulse(Raw[:,4].toV())

# BVP['HR']=pl.zeros(len(BVP['ZeroDerivative']))

# BVP['HR'][1:]=60./(pl.diff(BVP['ZeroDerivative'])/1000.)
# BVP['HR'][pl.find(abs(pl.diff(BVP['HR']))>10)+1]=0
# BVP['HR'][pl.find(abs(pl.diff(BVP['HR']))>10)+1]=0

#pl.figure()
#pl.plot(EDA['Signal'])

#pl.plot(EDA['Onset'][idx]*SamplingRate, EDA['Signal'][(EDA['Onset'][idx]*SamplingRate).astype('int')],'g.')
#pl.vlines(Switch['Event'],0,1,'k','--')
#pl.title(fname)
#pl.show()

#pl.figure()
#pl.plot(BVP['Signal'])
#pl.vlines(BVP['ZeroDerivative'],0,1,'k','--')
#pl.title(fname)
#pl.show()

Results={}

Results['Time']=pl.array([])
Results['ItemDuration']=pl.array([])

ds=pl.diff(Switch['Event'])
idx=pl.find(ds<pl.mean(ds)/4.)

for i in pl.arange(len(idx)):
	di=100 if i<5 else 67
	blbl='Block'+str(i+1)
	Results[blbl]=Switch['Event'][(idx[i]-di):idx[i]]
	Results['Time']=pl.append(Results['Time'], Results[blbl])
	Results[blbl]=pl.append(Results[blbl], Switch['Event'][idx[i]+1])
	Results['ItemDuration']=pl.append(Results['ItemDuration'],pl.diff(Results[blbl]))
	
ts=float(SamplingRate)#/60.

# Time (m)
Results['Time']=Results['Time']/ts/60.
# Item Duration (s)
Results['ItemDuration']=Results['ItemDuration']/ts
# Item #
Results['Item']=pl.arange(1,len(Results['Time'])+1)

nitems=len(Results['Item'])

Results['Activation']=pl.zeros(nitems)
Results['Amplitude']=pl.zeros(nitems)
Results['MeanHR']=pl.zeros(nitems)
Results['StdHR']=pl.zeros(nitems)

onset=EDA['Onset']*SamplingRate
beat=copy(BVP['Onset'])# beat=BVP['ZeroDerivative']/DownSampling
beat/=DownSampling

fparts=tls.fileparts(fname)

rdir=tls.fullfile('../results', fparts[-2])

if not os.path.isdir(rdir): os.mkdir(rdir)

t=pl.arange(len(EDA['Signal']))/ts/60.

item=0
for i in pl.arange(len(idx)):
	blbl='Block'+str(i+1)

	pl.figure()

	y=EDA['Signal'][Results[blbl][0]:Results[blbl][-1]]
	ti=t[Results[blbl][0]:Results[blbl][-1]]

	plmin=np.min(y)#0
	plmax=np.max(y)#1
	
	pl.fill_between(ti,plmin,plmax,color=(0,0,0,.25))
	pl.plot(ti,y,'k')
	pl.vlines(Results[blbl]/ts/60.,plmin,plmax,(.25,.25,.25),'--')
	pl.title(blbl)
	pl.xlabel('t (min)')
	pl.ylabel('SC (uS)')
	
	# pl.figure()

	# y=BVP['Signal'][::DownSampling]
	# y = (y-min(y))/(max(y)-min(y))
	# y=y[Results[blbl][0]:Results[blbl][-1]]

	# pl.fill_between(ti,0,1,color=(0,0,0,.25))
	# pl.plot(ti,y,'k')
	# hidx=pl.find((beat>Results[blbl][0])*(beat<Results[blbl][-1]))
	# pl.vlines(ti[beat[hidx]-Results[blbl][0]],0,1,'g')
	# pl.vlines(Results[blbl]/ts/60.,0,1,(.25,.25,.25),'--')
	# pl.title(blbl)
	# pl.xlabel('t (min)')
	# pl.ylabel('BVP (V)')		
	
	for j in pl.arange(1,len(Results[blbl])):
		aidx=pl.find((Results[blbl][j-1]<onset)*(Results[blbl][j]>onset))
		if len(aidx)!=0: 
			Results['Activation'][item]=1
			Results['Amplitude'][item]=EDA['Amplitude'][aidx[-1]]
			pl.fill_between(t[Results[blbl][(j-1):(j+1)]],plmin,plmax,color=(1,0,0,.25))
		hidx=pl.find((beat>Results[blbl][j-1])*(beat<Results[blbl][j]))
		
		hr = 1000.*(60.0/(np.diff(BVP['Onset'][hidx])))
		good_beats = np.intersect1d(pl.find(hr>60),pl.find(hr<120))
		if(np.any(hr)):
			good_beats2,j,mhr=[],0,np.mean(hr)
			for i in hr:
				if(i>mhr-15 and i<mhr+15):
					good_beats2+=[j]
				j+=1
			try:
				hr=hr[np.intersect1d(good_beats,good_beats2)]
				meanHR = np.mean(hr)
				stdHR = np.std(hr)
			except IndexError:
				meanHR, stdHR =0,0
		else:
			meanHR, stdHR =0,0
		
		Results['MeanHR'][item]=meanHR if not isnan(meanHR) else 0
		Results['StdHR'][item]=stdHR if not isnan(stdHR) else 0
		item+=1
	pl.savefig(tls.fullfile(rdir, fparts[-2]+'-'+blbl, 'pdf'))

albls=pl.array(['N', 'A'])

(m,s) = divmod(Results['Time']*60,60)
time = m+s/100

pl.savetxt(tls.fullfile(rdir, fparts[-2], fparts[-1]), 
	pl.array([time, Results['Item'], Results['ItemDuration'], \
	Results['Activation'],Results['Amplitude'],Results['MeanHR'], \
	Results['StdHR']]).transpose(),
	['%f', '%d', '%f', '%d', '%f', '%0.f', '%0.f'],'\t')#'%f %d %f %s %f')