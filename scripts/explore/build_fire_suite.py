#!/usr/bin/env python3
"""Generate ten immutable mesh/voxel fire scenes and an offline WebGL gallery."""
import argparse
import base64
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore.fire_buildings import FireBuilding, CATALOG


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode('ascii')


def payload(building):
    groups=defaultdict(list)
    for c in building.components:groups[(c['floor'],c['kind'])].append(c)
    packed=[]
    for (floor,kind),components in sorted(groups.items()):
        p=[];col=[]
        for c in components:
            triangle=c['vertices'][c['faces']].reshape(-1,3);p.append(triangle)
            col.append(np.tile(np.round(c['color']*255),(len(triangle),1)))
        p=np.concatenate(p).astype('<f4');col=np.concatenate(col).astype('u1')
        packed.append(dict(floor=floor,kind=kind,positions=b64(p),colors=b64(col),count=len(p)))
    return dict(**building.metadata(),groups=packed)


def validate(building,lab,ijk,res):
    from covor.explore.depth_observation import RayDepthSensor
    scene,_=building.render_scene(False)
    import open3d as o3d
    assert len(building.fires)>=building.levels
    assert all(not c['solid'] for c in building.components if c['kind']=='fire_visual')
    assert any(f['kind']=='locked_door' for f in building.features)
    assert any(f['kind']=='wall_collapse' for f in building.features)
    assert len(building.stairs)==2*(building.levels-1)
    assert all(s['rise_m']<=.18 for s in building.stairs)
    assert np.isfinite(building.mesh.vertices).all()
    assert all(len(c['faces'])>=4 for c in building.components)
    # Actual mesh rays, independent of voxel raster: floor, hole and both stair flights.
    queries=[]
    def ray(name,origin,direction,expect=None):
        ans=scene.cast_rays(o3d.core.Tensor(np.array([[*origin,*direction]],np.float32)))['t_hit'].numpy()[0]
        hit=float(ans);queries.append(dict(test=name,origin=origin,direction=direction,hit=hit,expected=expect))
        if expect is not None:assert np.isfinite(hit) and abs(hit-expect)<.04,(name,hit,expect)
        return hit
    c=building.depth/2
    # A clear corridor support is deliberately outside all debris/fire furniture.
    for f in range(building.levels):ray(f'floor_L{f+1}',[building.width*.44,c,f*3.3+.7],[0,0,-1],.7)
    # Closed locked leaf versus a genuine exterior window opening.
    locked=next(f for f in building.features if f['kind']=='locked_door')
    x,y,z=locked['position'];ray('locked_door_closed',[x,y-.5,z],[0,1,0],.4675)
    x=(building.room_cuts[0]+building.room_cuts[1])/2+.34
    hit=ray('broken_window_opening',[x,-.6,1.65],[0,1,0])
    assert hit>.8,('window still filled with wall',hit)
    for s in building.stairs:
        for flight in range(2):
            i=2;xx=s['x']+flight*1.45+.625;yy=s['y']+(i if flight==0 else 9-i)*.29+.145
            top=s['floor']*3.3+(i+1+flight*10)*.165
            ray(f'stair_{s["floor"]}_{s["x"]}_{flight}',[xx,yy,top+.08],[0,0,-1],.08)
    # Confirm at least one clear through-opening in each upper slab. Avoid its
    # adjacent furniture/debris by testing a deterministic interior grid.
    for f in range(1,building.levels):
        x,y,rx,ry=building.holes[f][-1]['ellipse'];hits=[]
        for dx in (-.35,0,.35):
            for dy in (-.3,0,.3):
                h=ray(f'floor_hole_L{f+1}',[x+dx,y+dy,f*3.3+.05],[0,0,-1]);hits.append(h)
        assert max(hits)>.5,('floor opening covered by intact slab',f,hits)
    # RealSense-inspired backend can consume this mesh without changing policy.
    cfg=json.loads(Path('configs/depth_model_20261106.json').read_text())
    sensor=RayDepthSensor(building,cfg,building.seed,'clean')
    from covor.explore.sensing import camera_pose
    T=camera_pose([building.width/2,c,1.1],0,-.25,.095)
    z,sigma,valid=sensor.measure(T,0,0)
    assert valid.any() and np.isfinite(z[valid]).all()
    return dict(passed=True,mesh_components=len(building.components),mesh_triangles=len(building.mesh.faces),
                occupied_voxels=int((lab==2).sum()),air_voxels=int((lab==1).sum()),
                mesh_ray_tests=queries,depth_backend_valid_pixels=int(valid.sum()),
                reference_routes_certified=False,seed=building.seed)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/fire_buildings_20261107')
    p.add_argument('--viewer',default='web/fire_buildings_3d.html');p.add_argument('--resolution',type=float,default=.15)
    p.add_argument('--indices',type=int,nargs='+',default=list(range(10)));p.add_argument('--fire-scale',type=float,default=1.);a=p.parse_args()
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    if (root/'catalog.json').exists():raise FileExistsError('Use a new output path; existing scene suites are preserved')
    library=[];catalog=[];checks=[]
    for index in a.indices:
        b=FireBuilding(index,fire_scale=a.fire_scale);folder=root/b.id
        print('building',b.id,len(b.components),'components',flush=True)
        meta=b.export(folder,a.resolution)
        q=np.load(folder/'gt_voxel.npz');check=validate(b,q['labels'],q['ijk'],a.resolution)
        (folder/'validation.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n')
        checks.append(dict(id=b.id,passed=check['passed'],triangles=check['mesh_triangles'],components=len(b.components)))
        catalog.append(dict(id=b.id,title=b.title,seed=b.seed,levels=b.levels,footprint_m=[b.width,b.depth],rooms=len(b.rooms),
                            windows=sum(c['name'].endswith('_sill') for c in b.components)//2,
                            locked_doors=sum(f['kind']=='locked_door' for f in b.features),fires=len(b.fires),fire_scale=a.fire_scale,stairs=len(b.stairs),
                            floor_holes=b.levels,description=b.description,files={n:str(folder/n) for n in ('scene.glb','scene.ply','scene.json','gt_voxel.npz','collision_geometry.json','validation.json')}))
        library.append(payload(b));print('validated',b.id,'triangles',check['mesh_triangles'],flush=True)
    (root/'catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n')
    with (root/'catalog.csv').open('w') as f:
        rows=[{k:v for k,v in x.items() if k!='files'} for x in catalog];w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    manifest=dict(source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('covor/explore/fire_buildings.py'),Path(__file__),Path('web/fire_buildings_3d.template.html')]},validation=checks)
    (root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    viewer=Path(a.viewer)
    html=Path('web/fire_buildings_3d.template.html').read_text().replace('/*__SCENES__*/[]',json.dumps(library,ensure_ascii=False,separators=(',',':'))).replace('__ASSET_ROOT__',os.path.relpath(root,viewer.parent))
    viewer.parent.mkdir(parents=True,exist_ok=True);viewer.write_text(html)
    print('DONE',len(library),'scenes;',viewer,round(viewer.stat().st_size/1e6,2),'MB',flush=True)


if __name__=='__main__':main()
