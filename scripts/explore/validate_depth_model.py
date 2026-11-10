#!/usr/bin/env python
"""Fixed sensor fixtures, error statistics and no-oracle/no-invalid-ray controls."""
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore.depth_observation import RayDepthSensor,correlated_field
from covor.explore.sensing import Sensor,SharedMap
from covor.explore.runner import csv_write,write_json

class Fixture:
    def __init__(self,kind,d):
        import open3d as o3d
        self.scene=o3d.t.geometry.RaycastingScene()
        def quad(x0,x1,y0,y1,z):
            vertices=np.array([[x0,y0,z],[x1,y0,z],[x1,y1,z],[x0,y1,z]],np.float32)
            self.scene.add_triangles(o3d.core.Tensor(vertices),o3d.core.Tensor(np.array([[0,1,2],[0,2,3]],np.uint32)))
        if kind in ('plane','floor'):quad(-20,20,-20,20,0 if kind=='floor' else d)
        elif kind=='step':quad(-20,0,-20,20,d);quad(0,20,-20,20,d+.18)
        elif kind=='door':
            quad(-20,-.5,-20,20,d);quad(.5,20,-20,20,d);quad(-.5,.5,-20,-.7,d);quad(-.5,.5,.7,20,d);quad(-20,20,-20,20,d+1.2)
        elif kind=='occlusion':quad(-.1,.1,-20,20,d);quad(-20,20,-20,20,d+2)
        elif kind=='wall':quad(-20,20,-20,20,d);quad(-20,20,-20,20,d+1)
        else:raise ValueError(kind)
    def render_scene(self,textured=False):return self.scene,None


def validate(config,out):
 cfg=json.load(open(config));out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];stds=[];normal_valid=normal_n=0;repro=True
 for kind in ('plane','floor','step','door'):
  for distance in (1.,2.,4.,6.):
   s=RayDepthSensor(Fixture(kind,distance),cfg,11000,'clean');T=np.eye(4)
   if kind=='floor':T[:3,:3]=np.diag([1.,-1.,-1.]);T[2,3]=distance
   errors=[];normal_generated=[];n_geometry=n_valid=0;n_core=valid_core=0
   for frame in range(8):
    t=frame*.25;(z,sz,v),a=s.measure(T,t,0,audit=True);g=a['geometric'];ids=a['ids'];normal=(~a['edge']&(a['incidence']<75)).flat[ids]
    # Far-plane censoring is reported, not hidden by clamping or probability changes.
    core=g&~a['edge']&(a['incidence']<75)&(a['true_z']<5.8)&(a['true_z']>.8)
    n_core+=int(core.sum());valid_core+=int((core&v).sum());n_geometry+=int(g.sum());n_valid+=int(v.sum());errors.extend((z[g&v]-a['true_z'][g&v]).tolist())
    if len(ids):normal_generated.extend((a['normal_error'][normal]/np.maximum(a['sigma_true'][normal],1e-20)).tolist())
    if frame==0:
     z2,_,v2=s.measure(T,t,0);repro&=np.array_equal(z,z2,equal_nan=True)&np.array_equal(v,v2)
     if distance==2 and kind in ('plane','step','door'):
      np.savez_compressed(out/(kind+'_representative.npz'),ideal=a['true_z'],observed=z,valid=v,geometry=g,sigma=sz,edge=a['edge'],incidence=a['incidence'])
   e=np.array(errors);standard=np.array(normal_generated)
   rows.append(dict(kind=kind,z_m=distance,n_geometry=n_geometry,n_valid=n_valid,conditional_valid=n_valid/max(n_geometry,1),n_core=n_core,core_valid_ratio=valid_core/n_core if n_core else None,error_mean_m=float(e.mean()) if len(e) else None,error_median_m=float(np.median(e)) if len(e) else None,error_std_m=float(e.std()) if len(e) else None,abs_error_p95_m=float(np.percentile(abs(e),95)) if len(e) else None,over_10cm=float((abs(e)>.1).mean()) if len(e) else None,generated_normal_standardized_std=float(standard.std()) if len(standard) else None))
   if len(standard):stds.extend(standard.tolist())
   normal_n+=n_core;normal_valid+=valid_core
   print(kind,distance,'valid',rows[-1]['conditional_valid'],'std/sigma',rows[-1]['generated_normal_standardized_std'],flush=True)
 # Spatial/time covariance of the Gaussian field, including common and individual parts.
 rng=np.random.default_rng(771);points=rng.uniform(-5,5,(15000,3));p=cfg['sensor']['error_model'];spacing=p['correlation_spacing_m'];ts=p['correlation_time_s']
 x=correlated_field(points,0.,11000,11,spacing,ts);cor=[]
 for dt in (0.,.25,.5,.75,1.):
  y=correlated_field(points,dt,11000,11,spacing,ts);cor.append(dict(axis='time',offset=dt,correlation=float(np.corrcoef(x,y)[0,1])))
 for dx in (.05,.1,.15,.3):
  y=correlated_field(points+[dx,0,0],0.,11000,11,spacing,ts);cor.append(dict(axis='space',offset=dx,correlation=float(np.corrcoef(x,y)[0,1])))
 oc=RayDepthSensor(Fixture('wall',2.),cfg,11000,'clean');(_,_,_),a=oc.measure(np.eye(4),0,0,audit=True);assert np.allclose(a['true_z'][a['geometric']],2.)
 oc=RayDepthSensor(Fixture('occlusion',2.),cfg,11000,'clean');(_,_,_),a=oc.measure(np.eye(4),0,0,audit=True);assert a['geometric'].sum()<a['left_geometric'].sum()
 # Backprojection Z/range and edge-of-search control.
 assert (a['geometric'][:,50]).any(),'no artificial 64-pixel SGBM search strip'
 z=np.ones((180,320),np.float32)*2.;v=np.zeros(z.shape,bool);v[90,200]=True
 pts,_=oc.sd.backproject(z,z*.001,v,downsample=1);assert abs(pts[0,2]-2)<1e-9;assert np.linalg.norm(pts[0])>2
 # Invalid measurements cannot create free rays.
 from covor.explore.scene import Building
 m=SharedMap(Building(11).prior(),cfg);before=m.dense()[0];invalid=np.full((180,320),np.nan,np.float32);m.integrate(oc.sd,np.eye(4),invalid,invalid,np.zeros(invalid.shape,bool),[]);assert np.array_equal(before,m.dense()[0]) and not m.surfaces.cells
 # Public inputs contain no true depth/error/mask/T. New backend does not build SGBM.
 from unittest.mock import patch
 with patch('cv2.StereoSGBM_create',side_effect=AssertionError('SGBM must not be built')):
  sensor=Sensor(Fixture('plane',2.),cfg,11000,'clean');sensor.capture([0,0,0],0,0,0,[],out/'public','d0_t00000',drone_id=0)
 keys=set(np.load(out/'public/d0_t00000.npz').files);assert not keys&{'true_z','outlier','drop','transmission','normal_error','sigma_true'};assert not list((out/'public').glob('*.png'))
 combined_std=float(np.std(stds));fill=normal_valid/normal_n
 assert repro;assert .90<combined_std<1.10,(combined_std,'correlated variance not calibrated');assert fill>.99,(fill,'normal core fill below 99%')
 csv_write(out/'sensor_errors.csv',rows);csv_write(out/'correlations.csv',cor)
 write_json(out/'checks.json',dict(deterministic_replay=repro,normal_core_conditional_valid=fill,generated_normal_standardized_std=combined_std,behind_wall_excluded=True,binocular_occlusion_excluded=True,invalid_carves_no_free=True,no_SGBM_created=True,no_PNG_created=True,public_keys=sorted(keys),far_range_boundary_note='z=6m positive Gaussian errors become invalid; no clipping to boundary',normal_core_definition='geometric, incidence<75deg, non-edge, .8<Z<5.8'))
 print('VALIDATION PASS',fill,combined_std,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/depth_model_20261106.json');p.add_argument('--out',required=True);a=p.parse_args();validate(a.config,a.out)
