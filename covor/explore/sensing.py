"""Current-time stereo observation and shared map; no candidate oracle access."""
from pathlib import Path
from time import perf_counter
import numpy as np
import cv2
from covor.occupancy import StereoCalib, DepthCfg, OccupancyBuilder, OccCfg
from covor.synth.render import DepthRenderer, _depth_rectified, _shade
from covor.smoke import stereo_smoke, SmokeCfg
from covor.entry.surfaces import SupportSurfaces, SurfaceCfg


def camera_pose(position,yaw,pitch,baseline=.08):
    right=np.array([np.sin(yaw),-np.cos(yaw),0.])
    forward=np.array([np.cos(yaw)*np.cos(pitch),np.sin(yaw)*np.cos(pitch),np.sin(pitch)])
    T=np.eye(4);T[:3,:3]=np.stack([right,np.cross(forward,right),forward],axis=1)
    T[:3,3]=position-right*baseline/2
    return T


class Sensor:
    def __init__(self,building,cfg,seed,condition):
        c=cfg['sensor'];K=np.array([[c['fx'],0,c['width']/2],[0,c['fx'],c['height']/2],[0,0,1.]])
        cal=StereoCalib(K,np.zeros(4),K.copy(),np.zeros(4),np.eye(3),np.array([-c['baseline'],0.,0.]),(c['width'],c['height']))
        scene,textures=building.render_scene()
        self.renderer=DepthRenderer(scene,None,DepthCfg(z_max=c['range'],num_disp=c['num_disp'],block=c['block']),textures,cal)
        self.sd=self.renderer.sd;self.radius=cfg['motion']['radius'];self.condition=condition
        self.smoke=SmokeCfg(**cfg['smoke'],seed=int(seed));self.cfg=c

    def capture(self,position,yaw,pitch,t,teammates,directory,frame_id,ideal=False):
        start=perf_counter();T=camera_pose(position,yaw,pitch,self.cfg['baseline']);images=[];depths=[]
        for dx in (0.,self.cfg['baseline']):
            z,extra=self.renderer._cast(T,dx,True)
            im=_shade(extra,self.renderer.textures,z.shape)
            # Ray/sphere intersection handles team occlusion BEFORE SGBM. These
            # endpoints are excluded from static surface/map evidence below.
            origin=T[:3,3]+T[:3,:3]@np.array([dx,0,0]);d=self.renderer.dir_rect@T[:3,:3].T
            a=(d*d).sum(-1)
            for mate in teammates:
                q=origin-np.asarray(mate);b=(d*q).sum(-1);disc=b*b-a*(q@q-self.radius**2)
                hit=np.where(disc>=0,(-b-np.sqrt(np.maximum(disc,0)))/a,np.inf)
                mask=(hit>.01)&(hit<z);z=np.where(mask,hit,z);im[mask]=96
            images.append(im);depths.append(z)
        render_wall=perf_counter()-start;start=perf_counter()
        meta=dict(fill_left=0.,fill_right=0.,transmission_mean=1.)
        if self.condition=='smoke':
            images[0],images[1],meta=stereo_smoke(self.sd,*images,*depths,T,t,self.smoke)
        smoke_wall=perf_counter()-start;start=perf_counter()
        if ideal:
            z=depths[0];valid=np.isfinite(z)&(z>=.3)&(z<=self.cfg['range']);z=np.where(valid,z,np.nan).astype(np.float32)
            sz=(z*z/(self.sd.fx*self.cfg['baseline'])).astype(np.float32)
        else:z,sz,valid=_depth_rectified(self.sd,*images)
        stereo_wall=perf_counter()-start;meta.update(render_wall=render_wall,smoke_wall=smoke_wall,stereo_wall=stereo_wall)
        out=Path(directory);out.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(out/(frame_id+'_left.png')),images[0]);cv2.imwrite(str(out/(frame_id+'_right.png')),images[1])
        np.savez_compressed(out/(frame_id+'.npz'),Z=z,sigma=sz,valid=valid,T=T,time=t,teammates=np.asarray(teammates).reshape(-1,3))
        return T,z,sz,valid,dict(valid_ratio=float(valid.mean()),**meta)


class SharedMap:
    def __init__(self,prior,cfg):
        self.res=cfg['map_resolution'];bounds=np.asarray(prior['bounds']);self.ijk=np.floor(bounds[0]/self.res).astype(int)
        self.shape=tuple(np.ceil(bounds[1]/self.res).astype(int)-self.ijk)
        self.builder=OccupancyBuilder(OccCfg(resolution=self.res,weighted=False,max_ray=cfg['sensor']['range'],dyn_radius=cfg['sensor']['teammate_exclusion_radius']))
        self.surfaces=SupportSurfaces(SurfaceCfg(resolution=self.res,max_range_m=cfg['sensor']['range'],dyn_radius_m=cfg['sensor']['teammate_exclusion_radius']))
        self.prior=prior;self.version=0

    def integrate(self,sd,T,z,sz,valid,teammates):
        P,s=sd.backproject(z,sz,valid,downsample=2)
        self.builder.integrate_frame(T,0.,P,s,teammates=teammates)
        n=self.surfaces.integrate(sd,z,sz,valid,T,teammates=teammates);self.version+=1
        return n

    def dense(self):
        lab=np.zeros(self.shape,np.uint8)
        occ,free=self.builder.classify_points()
        for points,value in ((free,1),(occ,2)):
            ind=np.floor(points/self.res).astype(int)-self.ijk
            good=((ind>=0)&(ind<np.array(self.shape))).all(1)
            lab[tuple(ind[good].T)]=value
        lo,hi=np.array(self.prior['launch_free_box'])
        a=np.ceil(lo/self.res-.5).astype(int)-self.ijk;b=np.floor(hi/self.res-.5).astype(int)-self.ijk+1
        sl=tuple(slice(x,y) for x,y in zip(a,b));part=lab[sl];part[part==0]=1
        return lab,self.surfaces.dense(self.ijk,self.shape)

    def save(self,path,lab=None,layer=None):
        if lab is None:lab,layer=self.dense()
        np.savez_compressed(path,labels=lab,ijk=self.ijk,res=self.res,version=self.version,**layer)
