#!/usr/bin/env python3
import sys,argparse,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore.fire_world import FireWorld
from covor.explore.depth_observation import RayDepthSensor
from covor.explore.sensing import camera_pose
from covor.explore.runner import csv_write,write_json

p=argparse.ArgumentParser();p.add_argument('--config',default='configs/fire_closed_loop_20261107.json');p.add_argument('--out',required=True);a=p.parse_args();cfg=json.load(open(a.config));out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
b=FireWorld(0,cfg);fire=b.fires[0];x,y,z=fire['position'];position=np.array([x+1.2,y+1.8,1.15]);yaw=np.arctan2(y-position[1],x-position[0]);T=camera_pose(position,yaw,-.35,.095);rows=[];public={}
for condition in ('clean','strong'):
 s=RayDepthSensor(b,cfg,0,condition);(depth,sigma,valid),private=s.measure(T,0,0,audit=True);g=private['geometric'];e=depth[g&valid]-private['true_z'][g&valid]
 rows.append(dict(condition=condition,n_geometry=int(g.sum()),n_valid=int(valid.sum()),conditional_valid=float(valid.sum()/g.sum()),error_std_m=float(e.std()),abs_error_p95_m=float(np.percentile(abs(e),95)),mean_transmission=float(private['transmission'].mean()),public_truth_fields=False))
 public[condition]=(depth,valid);np.savez_compressed(out/(condition+'.npz'),Z=depth,sigma=sigma,valid=valid,T=T,time=0.,drone_id=0)
 if condition=='strong':
  labels,ijk=b.labels(s.fire_smoke.res);assert not s.fire_smoke.field[labels==2].any();assert np.max(s.fire_smoke.field)>0
  again=s.measure(T,0,0);assert np.array_equal(depth,again[0],equal_nan=True)
assert rows[1]['mean_transmission']<.9 and rows[1]['error_std_m']>rows[0]['error_std_m']
assert rows[1]['conditional_valid']<rows[0]['conditional_valid']
csv_write(out/'paired_sensor.csv',rows);write_json(out/'checks.json',dict(source_smoke_connected=True,solid_field_zero=True,same_pose_geometry=True,deterministic=True,no_private_truth_to_mapping=True));print(rows,flush=True)
