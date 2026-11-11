#!/usr/bin/env python3
"""Compare repair utility on an identical observed map/state/candidate pool."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore.human import Human
from covor.explore.motion import FlightMap
from covor.explore.policy import candidates,score_pool,marginal
from covor.explore.candidate_views import refine_body_views
from covor.explore.depth_observation import camera_for
from covor.explore.runner import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--configs',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    prior=json.loads((a.run/'prior.json').read_text())
    with np.load(a.run/'maps/final.npz') as q:
        lab=q['labels'];ijk=q['ijk'];layer={k:q[k] for k in q.files if k not in ('labels','ijk','res','version')}
    all_rows=list(csv.DictReader((a.run/'trajectories.csv').open()));last=max(float(q['time']) for q in all_rows)
    states=[dict(id=int(q['drone']),p=np.array([float(q[k]) for k in ('estimated_x','estimated_y','estimated_z')]),failed=q['failed']=='True',active=q['active']=='True') for q in all_rows if float(q['time'])==last]
    results=[];pool_hashes=[]
    for path in a.configs:
        cfg=json.loads(path.read_text());human=Human(prior,cfg);analysis=human.analyze(lab,layer,ijk,True)
        flight=FlightMap(lab,ijk,cfg['map_resolution'],cfg['motion']);sd=camera_for(cfg)
        pool=candidates(lab,ijk,cfg['map_resolution'],layer,flight,states,cfg,round(last/cfg['assignment_period']),0,prior)
        pool=refine_body_views(pool,analysis,lab,layer,ijk,cfg['map_resolution'],sd,cfg,prior,flight=flight,states=states)
        digest=hashlib.sha256(json.dumps(pool,sort_keys=True).encode()).hexdigest();pool_hashes.append(digest)
        items=score_pool(pool,analysis,lab,layer,ijk,cfg['map_resolution'],sd,cfg,prior)
        results.append(dict(config=str(path),pool_hash=digest,candidates=[dict(**q['candidate'],unknown=len(q['unknown']),entry_gain=marginal(q,'entry',[],cfg,True)[0],goals=q['targets'],diagnostics=q['objective_diagnostics']) for q in items]))
        print(path.name,'pool',digest[:12],'repair_candidates',sum(c['kind']=='path_repair' for c in pool),'positive',sum(bool(q['targets']) for q in items),flush=True)
    assert len(set(pool_hashes))==1,'common candidate procedure disagrees at identical map/state'
    write_json(a.out,dict(map_only=True,truth_or_actual_candidate_observations_read=False,identical_candidate_pool=True,results=results))


if __name__=='__main__':main()
