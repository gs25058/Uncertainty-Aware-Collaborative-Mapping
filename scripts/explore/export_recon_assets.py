#!/usr/bin/env python3
"""Export a new scene revision without touching the approved base assets."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from covor.explore.fire_recon_scene import ReconFireWorld
from covor.explore.runner import write_json,csv_write
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/fire_recon_20261107.preflight.json');p.add_argument('--out',required=True);p.add_argument('--scene',type=int,action='append');a=p.parse_args();cfg=json.load(open(a.config));root=Path(a.out);root.mkdir(parents=True,exist_ok=True);rows=[]
for index in a.scene if a.scene is not None else range(10):
 b=ReconFireWorld(index,cfg);out=root/b.id
 if (out/'scene.json').exists():raise FileExistsError(out)
 meta=b.export(out,cfg['map_resolution']);rows.append(dict(scene=index,id=b.id,levels=b.levels,geometry_sha256=b.geometry_hash(),added_landings=len(b.geometry_repairs),source_assets='results/fire_buildings_large_fire_20261107/'+b.id,reference_labels='not supplied in asset export; computed independently by validation gate'))
 print('EXPORTED',b.id,'landings',len(b.geometry_repairs),flush=True)
csv_write(root/'catalog.csv',rows);write_json(root/'config.json',cfg)
