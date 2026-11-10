#!/usr/bin/env python3
"""Verify generated base solids against the actual preserved PLY/metadata files."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import trimesh
from covor.explore.fire_world import FireWorld
from covor.explore.fire_recon_scene import ReconFireWorld
from covor.explore.runner import csv_write,write_json
p=argparse.ArgumentParser();p.add_argument('--assets',default='results/fire_buildings_large_fire_20261107');p.add_argument('--config',default='configs/fire_recon_20261107.preflight.json');p.add_argument('--out',default='results/fire_recon_20261107/intake');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);cfg=json.load(open(a.config));rows=[]
for index in range(10):
 b=FireWorld(index,cfg);r=ReconFireWorld(index,cfg);folder=Path(a.assets)/b.id;geo=json.load(open(folder/'collision_geometry.json'));meta=json.load(open(folder/'scene.json'));assert len(geo)==len(b.components)
 for x,y,z in zip(b.components,r.components,geo):
  assert x['name']==y['name']==z['name'];assert x['solid']==z['solid'];assert np.array_equal(x['vertices'],y['vertices']);assert np.array_equal(x['faces'],y['faces']);assert np.allclose(x['vertices'],np.array(z['vertices']),atol=5.01e-7,rtol=0);assert np.array_equal(x['faces'],np.array(z['faces']))
 vertices=[];faces=[];offset=0
 for c in b.components:
  if not c['solid']:continue
  vertices.append(c['vertices']);faces.append(c['faces']+offset);offset+=len(c['vertices'])
 mesh=trimesh.load_mesh(folder/'scene.ply',process=False);assert np.array_equal(mesh.vertices.astype(np.float32),np.concatenate(vertices).astype(np.float32));assert np.array_equal(mesh.faces,np.concatenate(faces));assert b.fires==r.fires==meta['fires'];assert b.levels==meta['levels'];assert [b.width,b.depth]==meta['footprint_m']
 rows.append(dict(scene=index,id=b.id,base_ply_vertices_exact=True,base_ply_faces_exact=True,base_collision_geometry_matches=True,original_components_unchanged=True,fire_sources_unchanged=True,added_landings=len(r.geometry_repairs),base_geometry_sha256=ReconFireWorld(index,{**cfg,'scene':{**cfg['scene'],'repair_bottom_landings':False}}).geometry_hash(),repaired_geometry_sha256=r.geometry_hash()))
 print('MATCH',b.id,'added landings',len(r.geometry_repairs),flush=True)
csv_write(out/'asset_geometry_checks.csv',rows);write_json(out/'asset_geometry_checks.json',dict(scenes=len(rows),all_base_ply_exact=True,all_original_components_preserved=True,added_landings=sum(q['added_landings'] for q in rows)));print('ALL ASSETS MATCH',flush=True)
