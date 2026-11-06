"""Observed support surfaces, separate from volumetric occupancy.

Only locally planar, upward facing depth patches viewed from above contribute.
Votes are capped once per voxel per frame; free air above a surface cannot erase
it. Later depth rays demonstrably continuing *below* it supply contradictions.
No hole filling, reference geometry, or unobserved floor inference is used.
"""
from dataclasses import dataclass, asdict
import json
import numpy as np


@dataclass(frozen=True)
class SurfaceCfg:
    resolution: float = .1
    stride: int = 4
    max_range_m: float = 5.
    max_slope_deg: float = 30.
    plane_residual_m: float = .025
    min_frames: int = 2
    min_evidence: float = 1.2
    max_height_std_m: float = .05
    contradiction_margin_m: float = .05
    depth_beta: float = .5
    dyn_radius_m: float = .35


class SupportSurfaces:
    def __init__(self, cfg=None):
        self.cfg = cfg or SurfaceCfg()
        # values: positive weight, weighted z, weighted z², frame votes, negative
        self.cells = {}
        self.n_frames = 0

    def integrate(self, sd, Z, sigma_Z, valid, T, teammates=None):
        c=self.cfg;ds=c.stride
        z=np.asarray(Z)[::ds,::ds];sz=np.asarray(sigma_Z)[::ds,::ds]
        ok=np.asarray(valid)[::ds,::ds]&np.isfinite(z)&np.isfinite(sz)
        v,u=np.mgrid[:z.shape[0],:z.shape[1]]
        rect=np.stack(((u*ds-sd.cx)*z/sd.fx,(v*ds-sd.cy)*z/sd.fy,z),-1)
        raw=rect@sd.R1
        world=raw@T[:3,:3].T+T[:3,3]
        # Symmetric local tangent/plane checks reject depth discontinuities.
        p=world[1:-1,1:-1]
        dx=world[1:-1,2:]-world[1:-1,:-2]
        dy=world[2:,1:-1]-world[:-2,1:-1]
        n=np.cross(dx,dy);norm=np.linalg.norm(n,axis=-1)
        n=n/np.maximum(norm[...,None],1e-12)
        good=ok[1:-1,1:-1]&ok[1:-1,2:]&ok[1:-1,:-2]&ok[2:,1:-1]&ok[:-2,1:-1]
        good&=(norm>1e-8)&(np.abs(n[...,2])>=np.cos(np.deg2rad(c.max_slope_deg)))
        residual=np.zeros(good.shape)
        for neighbour in (world[1:-1,2:],world[1:-1,:-2],world[2:,1:-1],world[:-2,1:-1]):
            residual=np.maximum(residual,np.abs(np.sum((neighbour-p)*n,axis=-1)))
        good&=(residual<=c.plane_residual_m)&(T[2,3]>p[...,2]+c.contradiction_margin_m)
        good&=np.linalg.norm(p-T[:3,3],axis=-1)<=c.max_range_m
        if teammates is not None and len(teammates):
            for mate in teammates:good&=np.linalg.norm(p-np.asarray(mate),axis=-1)>=c.dyn_radius_m
        points=p[good];weights=np.exp(-sz[1:-1,1:-1][good]**2/c.depth_beta)
        # Verify old surfaces against this frame BEFORE adding its positive vote.
        self._contradictions(sd,Z,sigma_Z,valid,T)
        if len(points):
            vox=np.floor(points/c.resolution).astype(int)
            unique,inv=np.unique(vox,axis=0,return_inverse=True)
            sums=np.bincount(inv,weights=weights)
            means=np.bincount(inv,weights=weights*points[:,2])/np.maximum(sums,1e-12)
            seconds=np.bincount(inv,weights=weights*points[:,2]**2)/np.maximum(sums,1e-12)
            # One correlated image patch is one vote, never thousands of beams.
            strength=np.zeros(len(unique));np.maximum.at(strength,inv,weights)
            for key,z_,z2_,w_ in zip(map(tuple,unique),means,seconds,strength):
                a=self.cells.setdefault(key,np.zeros(5))
                a[:4]+=[w_,w_*z_,w_*z2_,1]
        self.n_frames+=1
        return len(points)

    def _contradictions(self,sd,Z,sigma_Z,valid,T):
        if not self.cells:return
        keys=np.array(list(self.cells));a=np.array(list(self.cells.values()))
        p=(keys+.5)*self.cfg.resolution;p[:,2]=a[:,1]/np.maximum(a[:,0],1e-12)
        rect=((p-T[:3,3])@T[:3,:3])@sd.R1.T
        with np.errstate(divide='ignore',invalid='ignore'):
            u=np.rint(sd.fx*rect[:,0]/rect[:,2]+sd.cx)
            v=np.rint(sd.fy*rect[:,1]/rect[:,2]+sd.cy)
        h,w=Z.shape
        inside=np.isfinite(u)&np.isfinite(v)&(rect[:,2]>.3)&(u>=0)&(u<w)&(v>=0)&(v<h)
        ids=np.flatnonzero(inside)
        if not len(ids):return
        uu=u[ids].astype(int);vv=v[ids].astype(int);z=Z[vv,uu];sz=sigma_Z[vv,uu]
        endpoints=np.c_[(uu-sd.cx)*z/sd.fx,(vv-sd.cy)*z/sd.fy,z]@sd.R1@T[:3,:3].T+T[:3,3]
        margin=self.cfg.contradiction_margin_m
        bad=valid[vv,uu]&np.isfinite(z)&np.isfinite(sz)
        bad&=(z>rect[ids,2]+np.maximum(.1,3*sz))&(endpoints[:,2]<p[ids,2]-margin)
        bad&=np.linalg.norm(endpoints-T[:3,3],axis=1)<=self.cfg.max_range_m
        for i,weight in zip(ids[bad],np.exp(-sz[bad]**2/self.cfg.depth_beta)):
            self.cells[tuple(keys[i])][4]+=float(weight)

    def dense(self,ijk,shape):
        support=np.zeros(shape,bool);height=np.full(shape,np.nan,np.float32)
        evidence=np.zeros(shape,np.float32);votes=np.zeros(shape,np.uint16)
        c=self.cfg
        for key,a in self.cells.items():
            index=np.array(key)-ijk
            if not ((index>=0)&(index<shape)).all():continue
            i=tuple(index);mean=a[1]/max(a[0],1e-12)
            var=max(0.,a[2]/max(a[0],1e-12)-mean**2)
            evidence[i]=a[0]-a[4];votes[i]=min(int(a[3]),65535);height[i]=mean
            support[i]=(a[3]>=c.min_frames and a[0]-a[4]>=c.min_evidence and var<=c.max_height_std_m**2)
        return dict(support=support,height=height,evidence=evidence,votes=votes)

    def save(self,path):
        np.savez_compressed(path,keys=np.array(list(self.cells),int).reshape(-1,3),
                            values=np.array(list(self.cells.values())).reshape(-1,5),
                            cfg=json.dumps(asdict(self.cfg)),n_frames=self.n_frames)

    @classmethod
    def load(cls,path):
        q=np.load(path);obj=cls(SurfaceCfg(**json.loads(str(q['cfg']))))
        obj.cells={tuple(k):v.copy() for k,v in zip(q['keys'],q['values'])}
        obj.n_frames=int(q['n_frames']);return obj
