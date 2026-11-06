#!/usr/bin/env python
"""Sensor-factor diagnostic: ideal depth at EXACT recorded camera poses/times.

No planning or candidate selection is rerun. This is a replay control, not a
performance run or a policy allowed to use a ground-truth depth sensor.
"""
import sys,json,csv,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore.scene import Building
from covor.explore.sensing import SharedMap,Sensor
from covor.explore.human import Human
from covor.explore.runner import write_json,csv_write,evaluate
from covor.entry.path_eval import ReferenceValidator,PathEvalCfg
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.run);out=Path(a.out)
if (out/'summary.json').exists():raise FileExistsError('use new output path')
out.mkdir(parents=True,exist_ok=True);(out/'depths').mkdir(exist_ok=True);cc=json.load(open(root/'config.json'));cfg=cc['configuration'];b=Building(cc['scene_seed']);prior=json.load(open(root/'prior.json'));sensor=Sensor(b,cfg,0,'clean');m=SharedMap(prior,cfg);frames=list(csv.DictReader(open(root/'frames.csv')))
for f in frames:
 q=np.load(root/'frames'/(f['frame_id']+'.npz'));T=q['T'];z=sensor.renderer._cast(T);d=sensor.renderer.dir_rect@T[:3,:3].T;aa=(d*d).sum(-1)
 for mate in q['teammates']:
  delta=T[:3,3]-mate;bb=(d*delta).sum(-1);disc=bb*bb-aa*(delta@delta-cfg['motion']['radius']**2);hit=np.where(disc>=0,(-bb-np.sqrt(np.maximum(disc,0)))/aa,np.inf);mask=(hit>.01)&(hit<z);z=np.where(mask,hit,z)
 valid=np.isfinite(z)&(z>=.3)&(z<=cfg['sensor']['range']);z=np.where(valid,z,np.nan).astype(np.float32);sz=(z*z/(sensor.sd.fx*cfg['sensor']['baseline'])).astype(np.float32);Ti=np.asarray(json.loads(f['camera_pose']));m.integrate(sensor.sd,Ti,z,sz,valid,q['teammates'])
 np.savez_compressed(out/'depths'/(f['frame_id']+'.npz'),Z=z,sigma=sz,valid=valid,T_render=T,T_integrate=Ti,time=float(f['time']))
lab,layer=m.dense();m.save(out/'map_ideal_at_recorded_poses.npz',lab,layer);analysis=Human(prior,cfg).analyze(lab,layer,m.ijk,False);gt,ijk=b.labels();ref=ReferenceValidator(gt,ijk,m.res,b.scene,cfg=PathEvalCfg(step_limit_m=cfg['human']['exact_step_limit'],contact_epsilon_m=cfg['human']['contact_epsilon_m']));truth=json.load(open(root/'reference_targets.json'));rows,summary=evaluate(analysis,truth,ref,prior,cfg['duration'],m.version)
summary.update(mode='fixed_trajectory_ideal_depth_replay',source_run=str(root),same_recorded_frames=len(frames),no_candidate_selection=True);csv_write(out/'target_paths.csv',rows);write_json(out/'summary.json',summary);write_json(out/'input_contract.json',dict(source_run=str(root),frames='same frame IDs, times, true camera poses, teammate occlusions and integration poses as source frames.csv',changed='depth sensor only; ideal ray depth replaces SGBM depth; same uncertainty formula and map/support/human rules',future_pose_use='recorded replay evaluation only; no closed-loop selection'))
print(summary,flush=True)
