"""Observed endpoint bounds local to the measured foot XY, not a whole column.

The full coarse-cell upper bound is retained for torso/obstacle checks. This
layer only makes the centre-leg contact query spatially consistent with the
already measured surface_x/y. No endpoint or obstacle is removed.
"""
import numpy as np


class ContactGeometry:
    def __init__(self,resolution=.1,xy_resolution=.025,radius=.01):
        self.resolution=resolution;self.xy_resolution=xy_resolution;self.radius=radius
        self.upper={}

    def integrate_world(self,world):
        if not len(world):return
        scale=np.array([self.xy_resolution,self.xy_resolution,self.resolution])
        keys,inv=np.unique(np.floor(world/scale).astype(np.int64),axis=0,return_inverse=True)
        high=np.full(len(keys),-np.inf);np.maximum.at(high,inv,world[:,2])
        for key,z in zip(map(tuple,keys),high):self.upper[key]=max(self.upper.get(key,-np.inf),float(z))

    def dense(self,layer,ijk):
        result=np.full(layer['height'].shape,np.nan,np.float32)
        mask=np.isfinite(layer['surface_x'])&np.isfinite(layer['surface_y'])&np.isfinite(layer['height'])
        cells=np.argwhere(mask)
        if not len(cells):return result
        points=np.c_[layer['surface_x'][mask],layer['surface_y'][mask]]
        base=np.floor(points/self.xy_resolution).astype(int);z=cells[:,2]+ijk[2]
        bound=np.full(len(cells),-np.inf);n=int(np.ceil(self.radius/self.xy_resolution))+1
        for dx in range(-n,n+1):
            for dy in range(-n,n+1):
                q=base+[dx,dy];centre=(q+.5)*self.xy_resolution
                distance=np.linalg.norm(np.maximum(abs(centre-points)-self.xy_resolution/2,0),axis=1)
                selected=np.flatnonzero(distance<=self.radius+1e-9)
                values=np.array([self.upper.get((int(q[i,0]),int(q[i,1]),int(z[i])),-np.inf) for i in selected])
                bound[selected]=np.maximum(bound[selected],values)
        fallback=layer['hit_z_max'][mask]
        result[tuple(cells.T)]=np.where(np.isfinite(bound),bound,fallback)
        return result
