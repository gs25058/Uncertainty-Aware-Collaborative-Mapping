"""Controlled source-local smoke field with solid-masked lattice diffusion.

This is a dimensionless visibility degradation field, not CFD/heat/gas safety.
It remains simulator-only. No true density/transmission is given to the policy.
"""
import os
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates


class FireSmokeField:
    def __init__(self,building,cfg):
        c=cfg['sensor']['fire_smoke'];self.res=c['resolution'];self.samples=c['ray_samples']
        cache=Path(c['cache'])/(building.id+'.npz')
        if cache.exists():
            q=np.load(cache);self.field=q['density'];self.ijk=q['ijk'];return
        labels,self.ijk=building.labels(self.res);air=labels==1
        x,y,z=np.ogrid[tuple(slice(0,n) for n in labels.shape)]
        x=(x+self.ijk[0]+.5)*self.res;y=(y+self.ijk[1]+.5)*self.res;z=(z+self.ijk[2]+.5)*self.res
        source=np.zeros(labels.shape,np.float32)
        for fire in building.fires:
            px,py,_=fire['position'];floor=fire['floor']*3.3;x0,y0,x1,y1=fire['room_bounds'];radius=fire['radius']*1.15
            radial=np.exp(-((x-px)**2+(y-py)**2)/(2*radius**2))
            vertical=.12+.88*np.exp(-((z-floor-2.65)/.72)**2)
            inside=(x>=x0)&(x<=x1)&(y>=y0)&(y<=y1)&(z>floor+.05)&(z<floor+3.04)
            source+=((fire['source_strength_relative']/3)*radial*vertical*inside).astype(np.float32)
        source*=air;density=np.zeros_like(source)
        for _ in range(c['iterations']):
            neighbour=np.zeros_like(density)
            neighbour[1:]+=density[:-1];neighbour[:-1]+=density[1:]
            neighbour[:,1:]+=density[:,:-1];neighbour[:,:-1]+=density[:,1:]
            neighbour[:,:,1:]+=density[:,:,:-1];neighbour[:,:,:-1]+=density[:,:,1:]
            density=((1-c['decay']-6*c['diffusion'])*density+c['diffusion']*neighbour+c['source_gain']*source)*air
        self.field=density;cache.parent.mkdir(parents=True,exist_ok=True)
        temporary=cache.with_name(cache.stem+f'_{os.getpid()}.npz')
        np.savez_compressed(temporary,density=density,ijk=self.ijk,res=self.res);temporary.replace(cache)

    def line_mean(self,origin,endpoints,t):
        total=np.zeros(len(endpoints));origin=np.asarray(origin)
        for fraction in (np.arange(self.samples)+.5)/self.samples:
            points=origin+(endpoints-origin)*fraction
            coords=(points/self.res-self.ijk-.5).T
            values=map_coordinates(self.field,coords,order=1,mode='constant',cval=0,prefilter=False)
            modulation=1+.08*np.sin(points[:,0]*.35+points[:,1]*.2-t*.25)+.04*np.sin(points[:,2]*.3+t*.1)
            total+=values*modulation
        return total/self.samples
