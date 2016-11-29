import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate,ndimage
from sauron_colormap import sauron
import sys
from makebeam import makebeam

def makeplots(f,xsize,ysize,vsize,cellsize,dv,beamsize,posang=0,overcube=False,pvdthick=2,nconts=11., **kwargs):
    
# ;;;; Create plot data from cube ;;;;
    mom0rot=f.sum(axis=2)
    if np.any(overcube): mom0over=overcube.sum(axis=2)
    x1=np.arange(-xsize/2.,xsize/2.,cellsize)
    y1=np.arange(-ysize/2.,ysize/2.,cellsize)
    v1=np.arange(-vsize/2.,vsize/2.,dv)

    mom1=(mom0rot*0.0)-10000.0
    for i in range(0,int(xsize/cellsize)):
         for j in range(0,int(ysize/cellsize)):
             if mom0rot[i,j] > 0.1*np.max(mom0rot):
                 mom1[i,j]=(v1*f[i,j,:]).sum()/f[i,j,:].sum()

    pvdcube=f 
    
    pvdcube=ndimage.interpolation.rotate(f, 90-posang, axes=(1, 0), reshape=False)
    if np.any(overcube): pvdcubeover=ndimage.interpolation.rotate(overcube, 90-posang, axes=(1, 0), reshape=False)
        
    pvd=pvdcube[:,np.int((ysize/2.)-pvdthick):np.int((ysize/2.)+pvdthick),:].sum(axis=1)
    if np.any(overcube): pvdover=pvdcubeover[:,np.int((ysize/2.)-pvdthick):np.int((ysize/2.)+pvdthick),:].sum(axis=1)
    
    if not isinstance(beamsize, (list, tuple, np.ndarray)):
        beamsize=np.array([beamsize,beamsize,0])
    beamtot=(makebeam(xsize,ysize,[beamsize[0]/cellsize,beamsize[1]/cellsize],rot=beamsize[2])).sum()
    spec=f.sum(axis=0).sum(axis=0)/beamtot
    if np.any(overcube): specover=overcube.sum(axis=0).sum(axis=0)/beamtot
     
# ;;;;

# ;;;; Plot the results ;;;;
    levs=v1[np.min(np.where(spec != 0)):np.max(np.where(spec != 0))]
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    ax1 = fig.add_subplot(221, aspect='equal')
    plt.xlabel('Offset (")')
    plt.ylabel('Offset (")')
    ax1.contourf(x1,y1,mom0rot.T,levels=np.linspace(1,0,num=10,endpoint=False)[::-1]*np.max(mom0rot), cmap="YlOrBr")
    if np.any(overcube): ax1.contour(x1,y1,mom0over.T,colors=('black'),levels=np.arange(0.1, 1.1, 0.1)*np.max(mom0over))
    if 'yrange' in kwargs: ax1.set_ylim(kwargs['yrange'])
    if 'xrange' in kwargs: ax1.set_xlim(kwargs['xrange'])
    ax2 = fig.add_subplot(222, aspect='equal')
    plt.xlabel('Offset (")')
    plt.ylabel('Offset (")')
    ax2.contourf(x1,y1,mom1.T,levels=levs, cmap=sauron)
    if 'yrange' in kwargs: ax2.set_ylim(kwargs['yrange'])
    if 'xrange' in kwargs: ax2.set_xlim(kwargs['xrange'])
    ax3 = fig.add_subplot(223)
    plt.xlabel('Offset (")')
    plt.ylabel(r'Velocity (km s$^{-1}$)')
    ax3.contourf(x1,v1,pvd.T,levels=np.linspace(1,0,num=10,endpoint=False)[::-1]*np.max(pvd), cmap="YlOrBr" ,aspect='auto')
    if np.any(overcube): ax3.contour(x1,v1,pvdover.T,colors=('black'),levels=np.arange(0.1, 1.1, 0.1)*np.max(pvdover))
    if 'vrange' in kwargs: ax3.set_ylim(kwargs['vrange'])
    if 'xrange' in kwargs: ax3.set_xlim(kwargs['xrange'])
    ax4 = fig.add_subplot(224)
    plt.ylabel('Flux')
    plt.xlabel(r'Velocity (km s$^{-1}$)')
    ax4.plot(v1,spec, drawstyle='steps')
    if np.any(overcube): ax4.plot(v1,specover,'r', drawstyle='steps')
    if 'vrange' in kwargs: ax4.set_xlim(kwargs['vrange'])
    plt.show()
# ;;;;