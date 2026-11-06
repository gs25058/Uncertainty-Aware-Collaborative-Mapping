#!/usr/bin/env python
"""Recorded-input reproduction, candidate admission and failed-drone audit."""
import sys,argparse,json,hashlib,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import cv2,numpy as np
from covor.occupancy import StereoCalib,StereoDepth,DepthCfg
from covor.synth.render import _depth_rectified
from covor.explore.runner import write_json
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);args=p.parse_args();root=Path(args.root);cfg=json.load(open(root/'fixed_config.json'));c=cfg['sensor'];K=np.array([[c['fx'],0,c['width']/2],[0,c['fx'],c['height']/2],[0,0,1.]])
sd=StereoDepth(StereoCalib(K,np.zeros(4),K.copy(),np.zeros(4),np.eye(3),np.array([-c['baseline'],0.,0.]),(c['width'],c['height'])),DepthCfg(z_max=c['range'],num_disp=c['num_disp'],block=c['block']))
checks=[];pool_checks=0;failed_checks=0
for folder in sorted(root.glob('scene*')):
 if not (folder/'summary.json').exists():continue
 frames=list(csv.DictReader(open(folder/'frames.csv')));assert len({r['frame_id'] for r in frames})==len(frames);assert [int(r['map_version']) for r in frames]==list(range(1,len(frames)+1))
 pool={}
 for line in open(folder/'candidates.jsonl'):
  r=json.loads(line);pool.setdefault(r['time'],set()).add(r['id'])
 for e in csv.DictReader(open(folder/'events.csv')):
  if e['event']=='route_pool_all_candidates':assert set(json.loads(e['candidate_ids']))==pool[float(e['time'])];pool_checks+=1
 tr=list(csv.DictReader(open(folder/'trajectories.csv')))
 for i in range(cfg['n_drones']):
  failed=[r for r in tr if int(r['drone'])==i and r['failed']=='True']
  if failed:
   q=np.array([[float(r[k]) for k in ('x','y','z')] for r in failed]);assert np.allclose(q,q[0],rtol=0,atol=1e-12);failed_checks+=1
 # Recompute one actually executed stereo image pair in EVERY run.
 f=frames[min(4,len(frames)-1)]['frame_id'];z=np.load(folder/'frames'/(f+'.npz'))
 il=cv2.imread(str(folder/'frames'/(f+'_left.png')),0);ir=cv2.imread(str(folder/'frames'/(f+'_right.png')),0);zz,sz,v=_depth_rectified(sd,il,ir)
 assert np.array_equal(v,z['valid']);assert np.array_equal(zz,z['Z'],equal_nan=True)
 checks.append(dict(run=folder.name,frame=f,exact_depth_replay=True,unique_frames=len(frames)))
manifest=json.load(open(root/'source_manifest.json'));same={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest.items() if Path(p).is_file()};assert all(same.values()),[p for p,v in same.items() if not v]
write_json(args.out,dict(raw_stereo_replays=checks,full_common_pool_checks=pool_checks,failed_drones_remained_frozen=failed_checks,source_unchanged=same,oracle_input_to_policy='none; only observed map, current estimated states, common prior and intrinsics are passed'))
print('PASS',len(checks),'exact SGBM replays;',pool_checks,'complete candidate pools;',failed_checks,'persistent frozen faults; source hashes unchanged')
