#!/usr/bin/env python
"""Rebuild maps from ordered saved observations, without scene/GT/planning."""
import sys,csv,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,cv2
from covor.occupancy import StereoCalib,StereoDepth,DepthCfg
from covor.synth.render import _depth_rectified
from covor.explore.sensing import SharedMap
from covor.explore.runner import write_json
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);p.add_argument('--sgbm',action='store_true');a=p.parse_args();root=Path(a.run);out=Path(a.out)
if (out/'verification.json').exists():raise FileExistsError('use a new output directory')
out.mkdir(parents=True,exist_ok=True);cfg=json.load(open(root/'config.json'))['configuration'];c=cfg['sensor'];K=np.array([[c['fx'],0,c['width']/2],[0,c['fx'],c['height']/2],[0,0,1.]])
sd=StereoDepth(StereoCalib(K,np.zeros(4),K.copy(),np.zeros(4),np.eye(3),np.array([-c['baseline'],0.,0.]),(c['width'],c['height'])),DepthCfg(z_max=c['range'],num_disp=c['num_disp'],block=c['block']))
m=SharedMap(json.load(open(root/'prior.json')),cfg);poses={}
for r in csv.DictReader(open(root/'trajectories.csv')):poses.setdefault(float(r['time']),{})[int(r['drone'])]=np.array([float(r[k]) for k in ('estimated_x','estimated_y','estimated_z')])
frames=list(csv.DictReader(open(root/'frames.csv')));previous=None;depth_exact=True
for f in frames:
 t=float(f['time'])
 if previous is not None and t!=previous:m.dense()
 q=np.load(root/'frames'/(f['frame_id']+'.npz'));z,sz,valid=q['Z'],q['sigma'],q['valid']
 if a.sgbm:
  il=cv2.imread(str(root/'frames'/(f['frame_id']+'_left.png')),0);ir=cv2.imread(str(root/'frames'/(f['frame_id']+'_right.png')),0);z,sz,valid=_depth_rectified(sd,il,ir);depth_exact&=np.array_equal(z,q['Z'],equal_nan=True)
 mates=[p for i,p in sorted(poses[t].items()) if i!=int(f['drone'])];m.integrate(sd,np.array(json.loads(f['camera_pose'])),z,sz,valid,mates);assert m.version==int(f['map_version']);previous=t
lab,layer=m.dense();m.save(out/'replayed_map.npz',lab,layer);original=np.load(root/'maps/final.npz');check=dict(labels=np.array_equal(lab,original['labels']))
for key,value in layer.items():check[key]=np.array_equal(value,original[key],equal_nan=True)
assert all(check.values()) and depth_exact,check
write_json(out/'verification.json',dict(source_run=str(root),mode='rerun_SGBM_from_saved_mono8' if a.sgbm else 'reuse_saved_depth',frames=len(frames),exact_depth=depth_exact,exact_final_map=check,scene_or_truth_read=False));print(len(frames),'frames; exact depth/final labels/support/height/votes/evidence PASS',flush=True)
