#!/usr/bin/env python
"""Posthoc synthetic sensor check against static shape, after frame selection."""
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore.scene import Building
from covor.explore.sensing import Sensor
from covor.explore.runner import csv_write,write_json
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.run);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);configuration=json.load(open(root/'config.json'));cfg=configuration['configuration'];sensor=Sensor(Building(configuration['scene_seed']),cfg,0,'clean');rows=[]
for fid in ('d0_t00000','d0_t00032','d0_t00064'):
 q=np.load(root/'frames'/(fid+'.npz'));T=q['T'];true=sensor.renderer._cast(T);dynamic=np.zeros(true.shape,bool);d=sensor.renderer.dir_rect@T[:3,:3].T;aa=(d*d).sum(-1)
 for mate in q['teammates']:
  delta=T[:3,3]-mate;bb=(d*delta).sum(-1);dis=bb*bb-aa*(delta@delta-cfg['motion']['radius']**2);hit=np.where(dis>=0,(-bb-np.sqrt(np.maximum(dis,0)))/aa,np.inf);dynamic|=(hit>.01)&(hit<true)
 mask=q['valid']&np.isfinite(true)&(true>=.3)&(true<=cfg['sensor']['range'])&~dynamic;err=abs(q['Z'][mask]-true[mask]);rows.append(dict(frame=fid,n_static_compared=int(mask.sum()),left_dynamic_pixels_excluded=int(dynamic.sum()),median_abs_depth_error_m=float(np.median(err)),mean_abs_depth_error_m=float(err.mean()),fraction_error_over_10cm=float((err>.1).mean()),fraction_error_over_50cm=float((err>.5).mean())))
csv_write(out/'stereo_quality.csv',rows);write_json(out/'definition.json',dict(frames='fixed t=0,8,16 s of drone0',reference='raycast static complete shape; left teammate occlusions excluded',scope='posthoc sensor diagnostic, no reference supplied to policy'));print(rows)
