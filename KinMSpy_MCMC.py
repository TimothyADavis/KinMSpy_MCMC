import numpy as np
import emcee
from KinMS import *
import os.path
import sys
import matplotlib.pyplot as plt
import time
import multiprocessing
from makeplots import makeplots
from make_corner import make_corner

def make_model(param,obspars,rad):
    ### This function takes in the `param` array (along with obspars; the observational setup, and a radius vector `rad`) and uses it to create a KinMS model.
    
    ### Here we use an exponential disk model for the surface brightness of the gas ###
    sbprof = np.exp((-1)*rad/param[7]) 
    
    ### We use a very simple rotation curve model that peaks at 3 arcseconds then is flat. Vflat is a free parameter here. ###
    velfunc = interpolate.interp1d([0.0,3,10,200],[0,1,1,1], kind='linear')
    vel=velfunc(rad)*param[6]
    
    ### This returns the model
    return KinMS(obspars['xsize'],obspars['ysize'],obspars['vsize'],obspars['cellsize'],obspars['dv'],obspars['beamsize'],param[2],sbprof=sbprof,sbrad=rad,velrad=rad,velprof=vel,nsamps=obspars['nsamps'],intflux=param[0],posang=param[1],gassigma=1.,phasecen=[param[3],param[4]],voffset=param[5],fixseed=True)
    ###


def lnlike(param,obspars,rad,fdata):
    ### This function calculates the log-likelihood, comparing model and data
    
    ### Run make_model to produce a model cube ###
    modout = make_model(param,obspars,rad)
        
    ### calculate the chi^2 ###
    chiconv = (((fdata-modout)**2)/((obspars['rms'])**2)).sum() 
    
    ### covert to log-likelihood ###
    like= -0.5*(chiconv-fdata.size)
    return like


def priors(param,priorarr):
    ### This function checks if any guess is out of range of our priors.
    
    ### initally assume all guesses in range ### 
    outofrange=0 
    
    ### Loop over each parameter ###
    for i in range(0,priorarr[:,0].size):
        ### If the parameter is out of range of the prior then add one to out of range, otherwise add zero
        outofrange+=1-(priorarr[i,0] <= param[i] <= priorarr[i,1])
        
    if outofrange:
        ### If outofrange NE zero at the end of the loop then at least oen parameter is bad, return -inf.
        return -np.inf
    else: 
        ### Otherwise return zero
        return 0.0


def lnprob(param,obspars,rad,fdata,priorarr):
    ### This function calls the others above, first checking that params are valid, and if so returning the log-likelihood.
    checkprior = priors(param,priorarr)
    if not np.isfinite(checkprior):
        return -np.inf
    return lnlike(param,obspars,rad,fdata)
    


################################################# Main code body starts here #######################################################################

### Setup cube parameters ###
obspars={}
obspars['xsize']=64.0 # arcseconds
obspars['ysize']=64.0 # arcseconds
obspars['vsize']=500.0 # km/s
obspars['cellsize']=1.0 # arcseconds/pixel
obspars['dv']=20.0 # km/s/channel
obspars['beamsize']=np.array([4.0,4.0,0]) # [bmaj,bmin,bpa] in (arcsec, arcsec, degrees)
obspars['nsamps']=5e5 # Number of cloudlets to use for KinMS models
obspars['rms']=1e-3 # RMS of data. Here we have a `perfect` model so this is arbitary - when fitting real data this should be the observational RMS

### Setup a radius vector ###
rad=np.arange(0.,64.)


### Make guesses for the parameters, and set prior ranges ###
labels=["intflux","posang","inc",'centx','centy','voffset',"vflat","discscale"] #name of each variable, for plot
intflux= 30        # Best fitting total flux
minintflux= 1.             # lower bound total flux
maxintflux= 50.            # upper bound total flux
posang= 50.              # Best fit posang.
minposang= 30.           # Min posang.
maxposang= 70.           # Max posang.
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
discscale=10              # Disc exponential scale length
mindiscscale=1            # Min Disc exponential scale length
maxdiscscale=20.          # Max Disc exponential scale length

### starting best guess ###
param=np.array([intflux,posang,inc,centx,centy,voffset,vflat,discscale]) 

### Setup array for priors - in this code all priors are uniform ###
priorarr=np.zeros((param.size,2))
priorarr[:,0]=[minintflux,minposang,mininc,mincentx,mincenty,minvoffset,min_vflat,mindiscscale] # Minimum
priorarr[:,1]=[maxintflux,maxposang,maxinc,maxcentx,maxcenty,maxvoffset,max_vflat,maxdiscscale] # Maximum


### Setup MCMC ###
ndim = param.size # How many parameters to fit
nwalkers = ndim*2 # Minimum of 2 walkers per free parameter
mcmc_steps=3000 # How many sample you want to take of the PDF. 3000 is reasonable for a test, larger the better for actual parameter estimation.
nsteps = mcmc_steps/nwalkers # Each walker must take this many steps to give a total of mcmc_steps steps


### Create a model to fit. Here this is made using the make_model function. In reality you would load in your observational data ###
real= param  # save the parameters used to make the real model
fdata=make_model(real,obspars,rad)


### How many CPUs to use. Here I am allowing half of the CPUs. Change this if you want more/less.
cpus2use=np.int(multiprocessing.cpu_count()/2)

### Do a test to estimate how long it will take to run the whole code ###
t0=time.time()
for i in range(0,10): fsim=make_model(param,obspars,rad)
t1=time.time() 
print("One model takes",((t1-t0)/10.),"seconds")
print("Total runtime expected with",cpus2use,"processors:",(((t1-t0)/10.)*mcmc_steps)/(0.6*cpus2use)) #This formula is a rough empirical estimate that works on my system. Your mileage may vary!


### Show what the initial model and data look like
makeplots(fsim,obspars['xsize'],obspars['ysize'],obspars['vsize'],obspars['cellsize'],obspars['dv'],obspars['beamsize'],rms=obspars['rms'],posang=real[1],overcube=fdata)
print("[Initial model - close plot to continue]")




### Code to run the MCMC
t0=time.time()
pos = [param + 1e-4*np.random.randn(ndim) for i in range(nwalkers)] # walkers start in tight ball around initial guess 
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(obspars,rad,fdata,priorarr),threads=cpus2use) # Setup the sampler

### Create a new output file, with the next free number ###
num=0
chainstart="KinMS_MCMCrun"
chainname=chainstart+str(num)+".dat"
while os.path.isfile(chainname):
    num+=1
    chainname=chainstart+str(num)+".dat"
f = open(chainname, "w")
f.close()


### Run the samples, while outputing a progress bar to the screen, and writing progress to the file.
width = 30
for i, result in enumerate(sampler.sample(pos, iterations=nsteps)):
    position = result[0]
    n = int((width+1) * float(i) / nsteps)
    sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
    f = open(chainname, "a")
    for k in range(position.shape[0]):
        f.write("{0:4d} {1:s}\n".format(k, " ".join(map(str, position[k]))))
    f.close()
sys.stdout.write("\n")         
t1=time.time()
print("It took",t1-t0,"seconds")

### Make corner plot (note that this includes all the samples - without a burn-in removed! Set discard to remove the first N samples)
corner=make_corner(filename=chainname,discard=0,labels=labels,truths=real)
