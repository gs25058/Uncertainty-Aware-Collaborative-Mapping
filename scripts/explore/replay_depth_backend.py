#!/usr/bin/env python
"""Replay identical recorded BODY poses/times with two new camera backends."""
import sys,json,csv,argparse,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore.scene import Building
from covor.explore.sensing import Sensor,SharedMap
from covor.explore.human import Human
from covor.explore.runner import write_json,csv_write,evaluate
from covor.entry.path_eval import ReferenceValidator,PathEvalCfg
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--config',default='configs/depth_model_20261106.json');p.add_argument('--out',required=True);p.add_argument('--duration',type=float,default=48);p.add_argument('--source-config',action='store_true');p.add_argument('--fire-recon',action='store_true');p.add_argument('--backend',choices=['both','ideal_depth','depth_error_model'],default='both');p.add_argument('--dense-final-only',action='store_true');a=p.parse_args();source=Path(a.source);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);seed=json.load(open(source/'config.json'));cfg=seed['configuration'] if a.source_config else json.load(open(a.config))
if a.fire_recon:
 from covor.explore.fire_recon_scene import ReconFireWorld
 b=ReconFireWorld(seed['scene_seed'],cfg)
else:b=Building(seed['scene_seed'])
b.render_scene(textured=False);prior=b.prior();gt,ijk=b.labels(cfg['map_resolution']);truth=json.load(open(source/'reference_targets.json'));records=list(csv.DictReader(open(source/'trajectories.csv')));groups={}
if a.source_config:assert prior==json.load(open(source/'prior.json')),'replay world/prior differs from recorded run'
for r in records:
 t=float(r['time'])
 if t<=a.duration:groups.setdefault(t,{})[int(r['drone'])]=r
outputs=[]
for backend in (('ideal_depth','depth_error_model') if a.backend=='both' else (a.backend,)):
 target=out/backend
 if (target/'summary.json').exists():raise FileExistsError(target)
 target.mkdir(exist_ok=True);c=copy.deepcopy(cfg);c['sensor']['backend']=backend;sensor=Sensor(b,c,seed['scene_seed']*1000+seed['run_seed'],seed.get('condition','clean') if a.source_config else 'clean');m=SharedMap(prior,c);frames=[];write_json(target/'config.json',c)
 for t,states in sorted(groups.items()):
  for drone,r in sorted(states.items()):
   if r['active']!='True' or r['failed']=='True':continue
   if abs(t/cfg['sensor_period']-round(t/cfg['sensor_period']))>1e-8:continue
   body=np.array([float(r[k]) for k in ('x','y','z')]);est=np.array([float(r[k]) for k in ('estimated_x','estimated_y','estimated_z')]);mates=[np.array([float(q[k]) for k in ('x','y','z')]) for j,q in sorted(states.items()) if j!=drone];emates=[np.array([float(q[k]) for k in ('estimated_x','estimated_y','estimated_z')]) for j,q in sorted(states.items()) if j!=drone]
   fid=f'd{drone}_t{int(round(t/cfg["dt"])):05d}';T,z,sz,v,meta=sensor.capture(body,float(r['yaw']),float(r['gimbal_pitch']),t,mates,target/'frames',fid,drone_id=drone);Ti=T.copy();Ti[:3,3]+=est-body;m.integrate(sensor.sd,Ti,z,sz,v,emates);frames.append(dict(frame_id=fid,time=t,drone=drone,map_version=m.version,camera_pose=Ti.tolist(),camera_pose_truth=T.tolist(),**meta))
  if a.dense_final_only:m.builder.tree.updateInnerOccupancy()
  else:lab,layer=m.dense()
 if a.dense_final_only:lab,layer=m.dense()
 if a.source_config:
  expected=[q['frame_id'] for q in csv.DictReader(open(source/'frames.csv')) if float(q['time'])<=a.duration]
  assert [q['frame_id'] for q in frames]==expected,'replay must use exactly the actual original capture order'
 m.save(target/'final_map.npz',lab,layer);human=Human(prior,c);analysis=human.analyze(lab,layer,m.ijk,False);validator=ReferenceValidator(gt,ijk,m.res,b.scene,cfg=PathEvalCfg(step_limit_m=c['human']['exact_step_limit'],contact_epsilon_m=c['human']['contact_epsilon_m']));rows,summary=evaluate(analysis,truth,validator,prior,a.duration,m.version);summary.update(backend=backend,source=str(source),mode='recorded_body_trajectory_replay_no_planning');write_json(target/'summary.json',summary);csv_write(target/'frames.csv',frames);csv_write(target/'target_paths.csv',rows);outputs.append(summary);print(backend,summary,flush=True)
csv_write(out/'comparison.csv',outputs)
