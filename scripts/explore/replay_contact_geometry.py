#!/usr/bin/env python3
"""Same raw hits, support, body rules; resolve foot contact at measured XY."""
import argparse,csv,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore.contact_geometry import ContactGeometry
from covor.explore.depth_observation import camera_for,ray_limit
from covor.explore.human import Human
from covor.explore.fire_recon_scene import ReconFireWorld
from covor.explore.runner import csv_write,write_json,evaluate
from covor.entry.path_eval import ReferenceValidator,PathEvalCfg


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--xy-resolution',type=float,default=.025);p.add_argument('--radius',type=float,default=.01);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();meta=json.loads((a.run/'config.json').read_text());cfg=meta['configuration'];res=cfg['map_resolution'];sd=camera_for(cfg)
    prior=json.loads((a.run/'prior.json').read_text());contact=ContactGeometry(res,a.xy_resolution,a.radius)
    frames=list(csv.DictReader((a.run/'frames.csv').open()))
    for n,row in enumerate(frames,1):
        with np.load(a.run/'frames'/(row['frame_id']+'.npz')) as q:
            T=np.array(json.loads(row['camera_pose']));points,_=sd.backproject(q['Z'],q['sigma'],q['valid'],downsample=2)
            world=points@T[:3,:3].T+T[:3,3];distance=np.linalg.norm(world-T[:3,3],axis=1)
            keep=(distance>res)&(distance<=ray_limit(cfg))
            for mate in q['teammates']:keep&=np.linalg.norm(world-mate,axis=1)>=cfg['sensor']['teammate_exclusion_radius']
            contact.integrate_world(world[keep])
        if n%200==0:print('HITS',n,'/',len(frames),'wall',round(time.monotonic()-start,1),flush=True)
    with np.load(a.run/'maps/final.npz') as q:
        lab=q['labels'];ijk=q['ijk'];layer={k:q[k] for k in q.files if k not in ('labels','ijk','res','version')}
    # Prove that refining spatial indexing did not remove ANY old upper hit.
    high=np.full(lab.shape,-np.inf,np.float32);ratio=round(res/a.xy_resolution)
    if abs(ratio*a.xy_resolution-res)>1e-10:raise ValueError('XY resolution must divide coarse resolution')
    keys=np.array(list(contact.upper));values=np.array(list(contact.upper.values()))
    coarse=np.c_[np.floor_divide(keys[:,:2],ratio),keys[:,2]]-ijk
    good=((coarse>=0)&(coarse<lab.shape)).all(1);np.maximum.at(high,tuple(coarse[good].T),values[good])
    original=layer['hit_z_max'];expected=np.where(np.isfinite(original),original,-np.inf)
    exact=np.array_equal(high,expected);write_json(a.out/'hits_preserved.json',dict(all_original_upper_hits_exact=exact,actual_frames=len(frames),fine_columns=len(contact.upper)))
    assert exact,'endpoint replay differs from original hit bounds'
    layer['contact_hit_z_max']=contact.dense(layer,ijk)
    cfg['human'].update(contact_xy_resolution=a.xy_resolution,contact_query_radius=a.radius)
    analysis=Human(prior,cfg).analyze(lab,layer,ijk,False);write_json(a.out/'paths_before_reference.json',analysis['decisions'])
    b=ReconFireWorld(meta['scene_seed'],cfg);gt,gtijk=b.labels(res);b.render_scene(False)
    v=ReferenceValidator(gt,gtijk,res,b.scene,cfg=PathEvalCfg(step_limit_m=cfg['human']['exact_step_limit'],contact_epsilon_m=cfg['human']['contact_epsilon_m']))
    truth=json.loads((a.run/'reference_targets.json').read_text());rows,summary=evaluate(analysis,truth,v,prior,cfg['duration'],len(frames))
    summary.update(source=str(a.run),xy_resolution=a.xy_resolution,contact_radius=a.radius,wall_seconds=time.monotonic()-start,all_hits_preserved=exact)
    write_json(a.out/'summary.json',summary);write_json(a.out/'config.json',cfg);csv_write(a.out/'targets.csv',rows)
    np.savez_compressed(a.out/'map.npz',labels=lab,ijk=ijk,res=res,version=len(frames),**layer)
    print('RESULT',json.dumps(summary),flush=True)


if __name__=='__main__':main()
