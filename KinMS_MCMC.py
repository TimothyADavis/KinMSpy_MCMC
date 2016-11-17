import numpy as np
import emcee
from KinMS import *
import pdb
import matplotlib.pyplot as plt
import time
import multiprocessing
from KinMS_testsuite import makeplots

def gaussian(x,x0,sigma):
  return np.exp(-np.power((x - x0)/(sigma), 2.)/2.)

def make_model(param,obspars,rad):
    sbprof = gaussian(rad,param[7],param[8])
    velfunc = interpolate.interp1d([0.0,1,10,200],[0,1,1,1], kind='linear')
    vel=velfunc(rad)*param[6]
    return KinMS(obspars['xsize'],obspars['ysize'],obspars['vsize'],obspars['cellsize'],obspars['dv'],obspars['beamsize'],param[2],sbprof=sbprof,sbrad=rad,velrad=rad,velprof=vel,nsamps=obspars['nsamps'],intflux=param[0],posang=param[1],gassigma=1.,phasecen=[param[3],param[4]],voffset=param[5],fixseed=True)


def lnlike(param,obspars,rad,fdata):
    modout = make_model(param,obspars,rad)
    chiconv = (((fdata-modout)**2)/((obspars['rms'])**2)).sum() 
    like= -0.5*(chiconv-fdata.size)
    return like

def priors(param,priorarr):
    outofrange=0
    for i in range(0,priorarr[:,0].size):
        outofrange+=1-(priorarr[i,0] <= param[i] <= priorarr[i,1])
    if outofrange:
        return -np.inf
    else: 
        return 0.0

def lnprob(param,obspars,rad,fdata,priorarr):
    checkprior = priors(param,priorarr)
    if not np.isfinite(checkprior):
        return -np.inf
    return lnlike(param,obspars,rad,fdata)
    


hdulist = fits.open('test_suite/NGC4324.fits')
fdata = hdulist[0].data.T
fdata=fdata[18:82,18:82,:,0]
rms=np.std(fdata[:,:,0])

#scidata = hdulist[0].data.T
#scidata=scidata[18:82,18:82,:,0]
#scidata[np.where(scidata < rms*4.)]=0.0


# ## Setup cube parameters ##
obspars={}
obspars['xsize']=64.0
obspars['ysize']=64.0
obspars['vsize']=420.0
obspars['cellsize']=1.0
obspars['dv']=20.0
obspars['beamsize']=np.array([4.68,3.85,15.54])
obspars['nsamps']=5e5
obspars['rms']=rms
rad=np.arange(0.,64.)


ndim, nwalkers = 9, 18
nsamps=3000

param=np.zeros(ndim)
priorarr=np.zeros((ndim,2))

labels=["intflux","posang","inc",'centx','centy','voffset',"vflat","r-ring","rrad"]
intflux= 30        # Best fitting total flux
minintflux= 1.             # lower bound total flux
maxintflux= 50.            # upper bound total flux
posang= 50.              # Best fit posang.
minposang= 0.           # Min posang.
maxposang= 360.           # Max posang.
inc= 65.                   # degrees
mininc=50.                # Min inc
maxinc=89.                # Max inc
centx=0.0                 # Best fit x-pos for kinematic centre
mincentx=-5.0             # min cent x
maxcentx=5.0              # max cent x
centy=0.0               # Best fit y-pos for kinematic centre
mincenty=-5.0             # min cent y
maxcenty=5.0              # max cent y
voffset= 0.0               # Best fit velocity centroid
minvoffset=-20.0          # min velocity centroid
maxvoffset=+20.0          # max velocity centroid
vflat =  175.41546             # vflat
min_vflat=10              # Lower range vflat
max_vflat=300             # Upper range vflat
r_ring=20.0               # ring radius
minr_ring=10.              # Lower ring radius
maxr_ring=25.              # Upper ring radius
rrad=2.
minrrad=0.
maxrrad=6.

param=[intflux,posang,inc,centx,centy,voffset,vflat,r_ring,rrad]
priorarr[:,0]=[minintflux,minposang,mininc,mincentx,mincenty,minvoffset,min_vflat,minr_ring,minrrad]
priorarr[:,1]=[maxintflux,maxposang,maxinc,maxcentx,maxcenty,maxvoffset,max_vflat,maxr_ring,maxrrad]
param=[  28.67937687,   48.18815941,   64.90943316,   -0.59164278,
         -0.26750091,   -3.27557838,  177.07797686,   21.1840498 ,
          3.83618746]
t0=time.time()
for i in range(0,10): fsim=make_model(param,obspars,rad)
t1=time.time()
print("Model takes",((t1-t0)/10.),"seconds")
print("Total runtime expected with",multiprocessing.cpu_count(),"processors:",(((t1-t0)/10.)*nsamps)/(0.6*multiprocessing.cpu_count()))

# print(lnprob(param,obspars,rad,fdata,priorarr))


#makeplots(fsim,obspars['xsize'],obspars['ysize'],obspars['vsize'],obspars['dx'],obspars['dy'],obspars['dv'],obspars['beamsize'],rms=obspars['rms'],overcube=fdata,posang=270.)
# print("[Initial model - close plot to continue]")


print(lnprob(param,obspars,rad,fdata,priorarr))



pos = [param + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

t0=time.time()
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(obspars,rad,fdata,priorarr),threads=multiprocessing.cpu_count())
sampler.run_mcmc(pos, nsamps/nwalkers)
t1=time.time()
print("It took",t1-t0,"seconds")

samples = sampler.chain[:, 30:, :].reshape((-1, ndim))
import corner
fig = corner.corner(samples, labels=labels, truths=param)
plt.show()

mymap = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84],axis=0)))
bestfit=np.array(list(mymap))
print(bestfit)
print(bestfit[:,0])
pdb.set_trace()
