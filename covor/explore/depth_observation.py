"""Lightweight camera-visible depth with stateless, correlated observation errors.

Truth is internal to raycasting/corruption. Public arrays contain ONLY observed
Z, a configuration/observation-based uncertainty, and validity. No stereo image
or SGBM object is constructed. The audit path is simulator/evaluation only.
"""
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter
import numpy as np
from scipy import ndimage, special
from covor.occupancy import StereoCalib, StereoDepth
from covor.smoke import line_mean_density


def backend_name(cfg):
    return cfg['sensor'].get('backend', 'legacy_sgbm')


def camera_for(cfg):
    c=cfg['sensor'];w,h=c['width'],c['height']
    fx=w/(2*np.tan(np.deg2rad(c['horizontal_fov_deg'])/2))
    fy=h/(2*np.tan(np.deg2rad(c['vertical_fov_deg'])/2))
    K=np.array([[fx,0,w/2],[0,fy,h/2],[0,0,1.]])
    cal=StereoCalib(K,np.zeros(4),K.copy(),np.zeros(4),np.eye(3),np.array([-c['baseline'],0.,0.]),(w,h))
    sd=SimpleNamespace(calib=cal,fx=fx,fy=fy,cx=w/2,cy=h/2,R1=np.eye(3),
                       cfg=SimpleNamespace(z_min=c['min_depth'],z_max=c['range']),range_metric='optical_z')
    sd.backproject=lambda z,s,v,downsample=4:StereoDepth.backproject(sd,z,s,v,downsample)
    return sd


def ray_limit(cfg):
    c=cfg['sensor']
    if backend_name(cfg)=='legacy_sgbm':return c['range']
    return float(c['range']*np.sqrt(1+np.tan(np.deg2rad(c['horizontal_fov_deg'])/2)**2+np.tan(np.deg2rad(c['vertical_fov_deg'])/2)**2))


def _mix(x):
    x=np.asarray(x,np.uint64)
    with np.errstate(over='ignore'):
        x=(x^(x>>30))*np.uint64(0xbf58476d1ce4e5b9)
        x=(x^(x>>27))*np.uint64(0x94d049bb133111eb)
    return x^(x>>31)


def _normal(key):
    a=(_mix(key)>>11).astype(float)/2**53
    b=(_mix(key^np.uint64(0x9e3779b97f4a7c15))>>11).astype(float)/2**53
    return np.sqrt(-2*np.log(np.maximum(a,1e-16)))*np.cos(2*np.pi*b)


def correlated_field(points,t,seed,channel,spacing=.15,time_scale=.75):
    """Gaussian lattice interpolation, normalized to unit marginal variance.

    Eight spatial nodes and two temporal nodes; no RNG state/call counter.
    Each node is a deterministic independent standard Gaussian hash sample.
    """
    p=np.asarray(points,float)/spacing;cell=np.floor(p).astype(np.int64);f=p-cell
    time=t/time_scale;it=int(np.floor(time));ft=time-it
    result=np.zeros(len(p));variance=np.zeros(len(p))
    constants=[np.uint64(v) for v in (0x9e3779b185ebca87,0xc2b2ae3d27d4eb4f,0x165667b19e3779f9,0x85ebca77c2b2ae63)]
    base=np.uint64((int(seed)*1000003+int(channel)*9176+137)&((1<<64)-1))
    for dx in (0,1):
        for dy in (0,1):
            for dz in (0,1):
                w=np.prod(np.where(np.array([dx,dy,dz]),f,1-f),axis=1)
                q=(cell+[dx,dy,dz]).astype(np.uint64)
                with np.errstate(over='ignore'):key=base^(q[:,0]*constants[0])^(q[:,1]*constants[1])^(q[:,2]*constants[2])
                for dt in (0,1):
                    weight=w*(ft if dt else 1-ft)
                    with np.errstate(over='ignore'):k=key^(np.uint64(it+dt)*constants[3])
                    result+=weight*_normal(k);variance+=weight*weight
    return result/np.sqrt(np.maximum(variance,1e-20))


class RayDepthSensor:
    def __init__(self,building,cfg,seed,condition):
        self.cfg=cfg;self.c=cfg['sensor'];self.profile=self.c['error_model'];self.seed=int(seed)
        self.condition='medium' if condition=='smoke' else condition
        self.degradation=self.profile['degradation'][self.condition]
        self.sd=camera_for(cfg);self.backend=backend_name(cfg);self.scene,_=building.render_scene(textured=False)
        v,u=np.indices((self.c['height'],self.c['width']),dtype=np.float32)
        self.dirs=np.stack(((u-self.sd.cx)/self.sd.fx,(v-self.sd.cy)/self.sd.fy,np.ones_like(u)),-1)
        self.radius=cfg['motion']['radius']
        self.fire_smoke=None
        if cfg['sensor'].get('fire_smoke',{}).get('enabled'):
            from .fire_smoke import FireSmokeField
            self.fire_smoke=FireSmokeField(building,cfg)

    def _cast(self,origin,directions,teammates):
        import open3d as o3d
        shape=directions.shape[:-1];d=directions.reshape(-1,3).astype(np.float32);o=np.asarray(origin,np.float32)
        rays=np.empty((len(d),6),np.float32);rays[:,:3]=o;rays[:,3:]=d
        ans=self.scene.cast_rays(o3d.core.Tensor(rays));z=ans['t_hit'].numpy();normal=ans['primitive_normals'].numpy()
        aa=(d*d).sum(1)
        for mate in teammates:
            delta=o-np.asarray(mate);b=(d*delta).sum(1);disc=b*b-aa*(delta@delta-self.radius**2)
            hit=np.where(disc>=0,(-b-np.sqrt(np.maximum(disc,0)))/aa,np.inf)
            mask=(hit>.0001)&(hit<z);z[mask]=hit[mask]
            normal[mask]=(o+d[mask]*hit[mask,None]-mate)/self.radius
        return z.reshape(shape),normal.reshape(*shape,3)

    def measure(self,T,t,drone_id,teammates=(),audit=False,ideal=False):
        """Explicit T allows exact recorded-pose replay (no Euler roundtrip)."""
        p=self.profile;shape=self.dirs.shape[:2];world_dirs=self.dirs@T[:3,:3].T
        true_z,normals=self._cast(T[:3,3],world_dirs,teammates)
        geometric=np.isfinite(true_z)&(true_z>=self.c['min_depth'])&(true_z<=self.c['range'])
        world=T[:3,3]+world_dirs*np.where(geometric,true_z,0)[...,None]
        left_geometric=geometric.copy()
        if p['binocular_visibility']:
            right=T[:3,3]+T[:3,:3]@np.array([self.c['baseline'],0,0])
            raw=self.dirs*np.where(geometric,true_z,1)[...,None];raw[...,0]-=self.c['baseline']
            u=self.sd.fx*raw[...,0]/raw[...,2]+self.sd.cx;v=self.sd.fy*raw[...,1]/raw[...,2]+self.sd.cy
            in_right=(u>=0)&(u<self.c['width'])&(v>=0)&(v<self.c['height'])
            ids=np.flatnonzero(geometric&in_right)
            seen=np.zeros(shape,bool)
            if len(ids):
                direction=world.reshape(-1,3)[ids]-right
                hit,_=self._cast(right,direction,teammates)
                seen.flat[ids]=abs(hit-1)*np.linalg.norm(direction,axis=1)<=p['right_visibility_tolerance_m']
            geometric&=seen
        incidence=np.rad2deg(np.arccos(np.clip(abs(np.sum(normals*world_dirs,axis=-1))/np.maximum(np.linalg.norm(normals,axis=-1)*np.linalg.norm(world_dirs,axis=-1),1e-20),0,1)))
        edge=np.zeros(shape,bool)
        for axis in (0,1):
            a=np.take(true_z,range(shape[axis]-1),axis=axis);b=np.take(true_z,range(1,shape[axis]),axis=axis)
            discontinuity=(np.isfinite(a)!=np.isfinite(b))|(abs(np.where(np.isfinite(a),a,0)-np.where(np.isfinite(b),b,0))>np.maximum(p['edge_absolute_threshold_m'],p['edge_relative_threshold']*np.minimum(a,b)))
            sl=[slice(None),slice(None)];sl[axis]=slice(None,-1);edge[tuple(sl)]|=discontinuity
            sl[axis]=slice(1,None);edge[tuple(sl)]|=discontinuity
        if p['edge_dilation_pixels']:edge=ndimage.binary_dilation(edge,iterations=p['edge_dilation_pixels'])
        grazing=np.clip((incidence-p['grazing_start_deg'])/(p['grazing_full_deg']-p['grazing_start_deg']),0,1)
        severity=np.maximum(edge.astype(float),grazing)
        valid=geometric.copy();z=np.full(shape,np.nan,np.float32);sigma=np.full(shape,np.nan,np.float32)
        ids=np.flatnonzero(geometric);points=world.reshape(-1,3)[ids];zt=true_z.flat[ids].astype(float)
        transmission=np.ones(len(ids));outlier=np.zeros(len(ids),bool);drop=outlier.copy();normal_error=np.zeros(len(ids));sigma_true=np.zeros(len(ids));bias=0.
        if self.backend=='ideal_depth' or ideal:z.flat[ids]=zt;sigma.flat[ids]=0.
        elif len(ids):
            digest=hashlib.sha256(np.asarray(T,np.float64).tobytes()+np.asarray([self.seed,drone_id,t],np.float64).tobytes()).digest()
            rng=np.random.default_rng(int.from_bytes(digest[:8],'little'))
            iid=rng.normal(size=(2,np.prod(shape)))[:,ids]
            fields=[correlated_field(points,t,self.seed,ch,p['correlation_spacing_m'],p['correlation_time_s']) for ch in (11,1000+drone_id*31,29,2000+drone_id*31)]
            w=np.sqrt(p['noise_variance_weights']);noise=w[0]*fields[0]+w[1]*fields[1]+w[2]*iid[0]
            w=np.sqrt(p['patch_variance_weights']);uniform=special.ndtr(w[0]*fields[2]+w[1]*fields[3]+w[2]*iid[1])
            bias=float(np.random.default_rng(self.seed*100003+drone_id*997+771).normal(0,p['drone_bias_std_m']))
            sigma_clean=(p['sigma_constant_m']+p['sigma_quadratic_per_m']*zt**2)*(1+p['geometry_sigma_increase']*severity.flat[ids])
            drop_clean=p['drop_probability']+p['geometry_drop_increase']*severity.flat[ids]
            k,a,q=(self.degradation[x] for x in ('k','a','q'))
            if k or a or q:
                length=np.linalg.norm(points-T[:3,3],axis=1)
                density=self.fire_smoke.line_mean(T[:3,3],points,t) if self.fire_smoke is not None else line_mean_density(T[:3,3],points,t,self.seed)
                tau=p['smoke_beta_per_m']*length*density
                transmission=np.exp(-tau)
            sigma_true=sigma_clean*(1+k*(1-transmission))
            pd=np.clip(drop_clean+(1-drop_clean)*a*(1-transmission),0,1)
            po=np.clip(p['outlier_probability']+q*(1-transmission),0,1)
            drop=uniform<pd;outlier=uniform>1-po
            amplitude=p['outlier_min_m']+(p['outlier_max_m']-p['outlier_min_m'])*special.ndtr(fields[0])
            error=np.where(outlier,np.where(fields[1]>=0,1,-1)*amplitude,sigma_true*noise)+bias
            measured=zt+error;keep=~drop&np.isfinite(measured)&(measured>=self.c['min_depth'])&(measured<=self.c['range'])
            valid.flat[ids]=keep;z.flat[ids[keep]]=measured[keep]
            # Public quality is a conservative known-profile function of OBSERVED Z.
            m=measured[keep];known=(p['sigma_constant_m']+p['sigma_quadratic_per_m']*m*m)*(1+p['geometry_sigma_increase'])*(1+k)
            ea2=(p['outlier_min_m']**2+p['outlier_min_m']*p['outlier_max_m']+p['outlier_max_m']**2)/3
            sigma.flat[ids[keep]]=np.sqrt(known*known+(p['outlier_probability']+q)*ea2+p['drone_bias_std_m']**2)
            normal_error=sigma_true*noise
        public=(z,sigma,valid)
        if not audit:return public
        private=dict(true_z=true_z,geometric=geometric,left_geometric=left_geometric,edge=edge,incidence=incidence,severity=severity,ids=ids,bias=bias,sigma_true=sigma_true,normal_error=normal_error,outlier=outlier,drop=drop,transmission=transmission)
        return public,private

    def capture(self,position,yaw,pitch,t,teammates,directory,frame_id,ideal=False,drone_id=0):
        from .sensing import camera_pose
        start=perf_counter();T=camera_pose(position,yaw,pitch,self.c['baseline'])
        (z,sigma,valid),audit=self.measure(T,t,drone_id,teammates,audit=True,ideal=ideal)
        wall=perf_counter()-start;out=Path(directory);out.mkdir(parents=True,exist_ok=True)
        # Primary input caches deliberately exclude all simulator truth fields.
        np.savez_compressed(out/(frame_id+'.npz'),Z=z,sigma=sigma,valid=valid,T=T,time=t,drone_id=drone_id,teammates=np.asarray(teammates).reshape(-1,3))
        g=audit['geometric'];normal=g&~audit['edge']&(audit['incidence']<p_grazing(self.profile))
        err=z[g&valid]-audit['true_z'][g&valid]
        record=dict(time=t,drone=drone_id,frame_id=frame_id,n_grid=z.size,n_left_geometry=int(audit['left_geometric'].sum()),n_geometry=int(g.sum()),n_valid=int(valid.sum()),n_normal_geometry=int(normal.sum()),n_normal_valid=int((normal&valid).sum()),n_outlier=int(audit['outlier'].sum()),n_random_drop=int(audit['drop'].sum()),mean_error=float(np.mean(err)) if len(err) else None,rms_error=float(np.sqrt(np.mean(err*err))) if len(err) else None)
        target=out.parent/'simulator_audit.csv';exists=target.exists()
        with target.open('a') as f:
            writer=csv.DictWriter(f,fieldnames=list(record));
            if not exists:writer.writeheader()
            writer.writerow(record)
        return T,z,sigma,valid,dict(valid_ratio=float(valid.mean()),render_wall=wall,smoke_wall=0.,stereo_wall=0.,backend='ideal_depth' if ideal else self.backend,profile=self.profile['name'])


def p_grazing(profile):return profile['grazing_start_deg']
