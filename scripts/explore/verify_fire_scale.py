#!/usr/bin/env python3
"""Verify that a fire-scale change preserves all geometric experiment factors."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def footprint(scene,res=.05):
    """Union of declared fire discs clipped to their rooms, not real burnt area."""
    w,h=scene['footprint_m'];x,y=np.meshgrid(np.arange(res/2,w,res),np.arange(res/2,h,res),indexing='ij');area=0.
    for floor in range(scene['levels']):
        mask=np.zeros(x.shape,bool)
        for f in scene['fires']:
            if f['floor']!=floor:continue
            px,py,_=f['position'];region=f.get('room_bounds')
            if region is None:
                room=next(r for r in scene['rooms'] if r['floor']==floor and r['bounds'][0]<=px<=r['bounds'][2] and r['bounds'][1]<=py<=r['bounds'][3]);region=room['bounds']
            x0,y0,x1,y1=region
            mask|=((x-px)**2+(y-py)**2<=f['radius']**2)&(x>=x0)&(x<=x1)&(y>=y0)&(y<=y1)
        area+=float(mask.sum())*res**2
    return area


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',default='results/fire_buildings_20261107');p.add_argument('--large',default='results/fire_buildings_large_fire_20261107');p.add_argument('--out',default='reports/fire_buildings_large_fire_20261107');a=p.parse_args()
    base,large,out=Path(a.base),Path(a.large),Path(a.out);out.mkdir(parents=True,exist_ok=True);rows=[]
    for c in json.loads((large/'catalog.json').read_text()):
        old,new=base/c['id'],large/c['id'];s=json.loads((old/'scene.json').read_text());t=json.loads((new/'scene.json').read_text())
        colliders=lambda p:[c for c in json.loads((p/'collision_geometry.json').read_text()) if c['solid']]
        assert colliders(old)==colliders(new),c['id']
        assert s['prior']==t['prior'] and s['stairs']==t['stairs'] and s['slab_openings']==t['slab_openings']
        assert (old/'scene.ply').read_bytes()==(new/'scene.ply').read_bytes(),'solid mesh including colors changed'
        with np.load(old/'gt_voxel.npz') as x,np.load(new/'gt_voxel.npz') as y:
            assert all(np.array_equal(x[k],y[k]) for k in ('labels','ijk','res'))
        for f in t['fires']:
            assert f['visual_only'] and f['position'][2]+f['height']<=f['floor']*3.3+3.02
        assert all(not c['solid'] for c in json.loads((new/'collision_geometry.json').read_text()) if c['kind']=='fire_visual')
        before,after=footprint(s),footprint(t)
        rows.append(dict(scene=c['id'],old_sources=len(s['fires']),new_sources=len(t['fires']),old_projected_source_area_m2=before,new_projected_source_area_m2=after,area_ratio=after/before,solid_geometry_identical=True,voxel_labels_identical=True,prior_entry_goals_identical=True,fire_not_solid=True,ceiling_height_respected=True))
    with (out/'fire_scale_comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(scenes=len(rows),old_sources=sum(r['old_sources'] for r in rows),new_sources=sum(r['new_sources'] for r in rows),old_projected_source_area_m2=sum(r['old_projected_source_area_m2'] for r in rows),new_projected_source_area_m2=sum(r['new_projected_source_area_m2'] for r in rows),all_geometry_and_targets_unchanged=True,observation_error_profile_changed=False,sensor_fire_coupling_implemented=False)
    (out/'invariants.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
