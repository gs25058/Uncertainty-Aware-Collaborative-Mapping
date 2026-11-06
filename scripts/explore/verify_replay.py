#!/usr/bin/env python
"""Recompute actual saved stereo and compare a fully repeated closed loop."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,cv2
from covor.occupancy import StereoCalib,StereoDepth,DepthCfg
from covor.synth.render import _depth_rectified
from covor.explore.runner import write_json
root=Path('results/closed_loop_20261105');main=root/'main';cfg=json.load(open(main/'fixed_config.json'));c=cfg['sensor'];K=np.array([[c['fx'],0,c['width']/2],[0,c['fx'],c['height']/2],[0,0,1.]])
sd=StereoDepth(StereoCalib(K,np.zeros(4),K.copy(),np.zeros(4),np.eye(3),np.array([-c['baseline'],0.,0.]),(c['width'],c['height'])),DepthCfg(z_max=c['range'],num_disp=c['num_disp'],block=c['block']))
checks=[]
for condition in ('clean','smoke'):
 folder=main/f'scene11_seed0_{condition}_unknown'
 for fid in ('d0_t00000','d0_t00064'):
  q=np.load(folder/'frames'/(fid+'.npz'));il=cv2.imread(str(folder/'frames'/(fid+'_left.png')),0);ir=cv2.imread(str(folder/'frames'/(fid+'_right.png')),0)
  z,sz,v=_depth_rectified(sd,il,ir);assert np.array_equal(v,q['valid']);assert np.array_equal(z,q['Z'],equal_nan=True)
  checks.append(dict(condition=condition,frame=fid,sgbm_exact_replay=True))
a=np.load(root/'validation/four_clean/maps/final.npz');b=np.load(main/'scene11_seed0_clean_unknown/maps/final.npz')
assert np.array_equal(a['labels'],b['labels']);assert np.array_equal(a['support'],b['support'])
pa=json.load(open(root/'validation/four_clean/summary.json'));pb=json.load(open(main/'scene11_seed0_clean_unknown/summary.json'));assert pa['travel_distance']==pb['travel_distance']
manifest=json.load(open(main/'source_manifest.json'));unchanged={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest.items() if Path(p).is_file()}
assert all(unchanged.values()),unchanged
write_json(root/'audit/replay_checks.json',dict(stereo_checks=checks,independent_repeat_map_equal=True,independent_repeat_distance=pa['travel_distance'],source_unchanged_during_primary_matrix=unchanged))
print('Exact replay: four raw stereo pairs, repeated final labels/support/distance, frozen source hashes PASS')
